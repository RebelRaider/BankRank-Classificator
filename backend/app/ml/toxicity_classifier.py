import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List
import numpy as np
from ..core.config import settings

class ToxicityClassifier:
    def __init__(self):
        self.model_name = settings.TOXICITY_MODEL_NAME
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the toxicity classification model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded toxicity model: {self.model_name}")
        except Exception as e:
            print(f"Error loading toxicity model: {e}")
            raise e
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Predict toxicity for a single text
        Returns: Dict with toxicity_score, is_toxic, confidence
        """
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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
        # Extract results
        # Assuming binary classification: [non_toxic, toxic]
        toxic_prob = probabilities[0][1].item()
        non_toxic_prob = probabilities[0][0].item()
        
        is_toxic = toxic_prob > 0.5
        confidence = max(toxic_prob, non_toxic_prob)
        
        return {
            "toxicity_score": toxic_prob,
            "is_toxic": is_toxic,
            "confidence": confidence,
            "probabilities": {
                "non_toxic": non_toxic_prob,
                "toxic": toxic_prob
            }
        }
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict toxicity for multiple texts"""
        results = []
        
        # Process in batches to manage memory
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Process results
            for j in range(len(batch_texts)):
                toxic_prob = probabilities[j][1].item()
                non_toxic_prob = probabilities[j][0].item()
                
                is_toxic = toxic_prob > 0.5
                confidence = max(toxic_prob, non_toxic_prob)
                
                results.append({
                    "toxicity_score": toxic_prob,
                    "is_toxic": is_toxic,
                    "confidence": confidence,
                    "probabilities": {
                        "non_toxic": non_toxic_prob,
                        "toxic": toxic_prob
                    }
                })
        
        return results
