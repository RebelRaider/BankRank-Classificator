import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from typing import Any, Dict, List, Tuple
from torch.utils.data import Dataset
import json
from ..core.config import settings

class RatingDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class RatingClassifier:
    def __init__(self):
        # Используем mps, если доступно, иначе cuda, иначе cpu
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model_name = settings.RATING_MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        self.num_labels = 0
        
    def load_pretrained_model(self):
        """Load the base BERT model for fine-tuning"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"Loaded tokenizer: {self.model_name}")
    
    def prepare_data(self, csv_path: str) -> Tuple[List[str], List[int], Dict]:
        """
        Prepare training data from CSV file
        Expected CSV format: text, Категория
        """
        df = pd.read_csv(csv_path)
        
        # Use only the category column as specified
        texts = df.iloc[:, 0].astype(str).tolist()  # First column (text)
        categories = df['Категория'].astype(str).tolist()  # Category column
        
        # Create label mappings
        unique_labels = sorted(list(set(categories)))
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        # Ensure id_to_label keys are int
        self.id_to_label = {int(k): v for k, v in self.id_to_label.items()}
        self.num_labels = len(unique_labels)
        
        # Convert labels to ids
        label_ids = [self.label_to_id[label] for label in categories]
        
        # Data statistics
        stats = {
            "total_samples": len(texts),
            "num_labels": self.num_labels,
            "label_distribution": dict(pd.Series(categories).value_counts()),
            "label_mappings": self.label_to_id
        }
        
        return texts, label_ids, stats
    
    def train_model(self, csv_path: str, output_dir: str = "./rating_model", 
                   test_size: float = 0.2, epochs: int = 3, batch_size: int = 16):
        """Train the rating classification model"""
        
        # Load and prepare data
        texts, labels, stats = self.prepare_data(csv_path)
        print(f"Dataset statistics: {stats}")
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Load model with correct number of labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        
        # Create datasets
        train_dataset = RatingDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = RatingDataset(val_texts, val_labels, self.tokenizer)
        
        # Training arguments (use TrainingArguments, not dict)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_accuracy',
        )
        
        # Compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
            accuracy = accuracy_score(labels, predictions)
            
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        with open(f"{output_dir}/label_mappings.json", "w", encoding="utf-8") as f:
            json.dump({
                "label_to_id": self.label_to_id,
                "id_to_label": self.id_to_label,
                "num_labels": self.num_labels
            }, f, ensure_ascii=False, indent=2)
        
        # Evaluate on test set
        test_results = trainer.evaluate()
        
        # Generate classification report
        predictions = trainer.predict(val_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = val_labels
        
        report = classification_report(
            y_true, y_pred, 
            target_names=[self.id_to_label[i] for i in range(self.num_labels)],
            output_dict=True
        )
        
        return {
            "training_stats": stats,
            "test_results": test_results,
            "classification_report": report,
            "model_path": output_dir
        }
    
    def load_trained_model(self, model_path: str):
        """Load a trained rating classification model"""
        try:
            # Load label mappings
            with open(f"{model_path}/label_mappings.json", "r", encoding="utf-8") as f:
                mappings = json.load(f)
                self.label_to_id = mappings["label_to_id"]
                # Ensure id_to_label keys are int
                self.id_to_label = {int(k): v for k, v in mappings["id_to_label"].items()}
                self.num_labels = mappings["num_labels"]
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Loaded trained rating model from: {model_path}")
            
        except Exception as e:
            print(f"Error loading trained model: {e}")
            raise e
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict rating for a single text"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        # Переносим на нужное устройство
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get predicted class and confidence
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        predicted_label = self.id_to_label[predicted_class_id]
        confidence = probabilities[0][predicted_class_id].item()
        
        # Get all probabilities
        all_probs = {
            self.id_to_label[i]: probabilities[0][i].item() 
            for i in range(self.num_labels)
        }
        
        return {
            "category": predicted_label,
            "confidence": confidence,
            "probabilities": all_probs
        }
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict ratings for multiple texts"""
        results = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            # Переносим на нужное устройство
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Process results
            for j in range(len(batch_texts)):
                predicted_class_id = torch.argmax(probabilities[j]).item()
                predicted_label = self.id_to_label[predicted_class_id]
                confidence = probabilities[j][predicted_class_id].item()
                
                all_probs = {
                    self.id_to_label[k]: probabilities[j][k].item() 
                    for k in range(self.num_labels)
                }
                
                results.append({
                    "category": predicted_label,
                    "confidence": confidence,
                    "probabilities": all_probs
                })
        
        return results
