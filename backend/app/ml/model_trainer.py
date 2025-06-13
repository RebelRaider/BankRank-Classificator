import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import json
from .rating_classifier import RatingClassifier
from .data_processor import TextProcessor

class ModelTrainer:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.rating_classifier = RatingClassifier()
        
    def analyze_dataset(self, csv_path: str) -> Dict:
        """Analyze the dataset and generate visualizations"""
        df = pd.read_csv(csv_path)

        # Определяем имя текстовой колонки
        text_col = [col for col in df.columns if col not in ['Категория', 'Id', 'id', 'ID', "Уровень рейтинга"]][0]

        # Очистка текста только в текстовой колонке
        df[text_col] = df[text_col].astype(str).apply(self.text_processor.preprocess_for_training)
        print(df[text_col].head(10))  # Добавьте для проверки
        print(df['Категория'].value_counts())  # Проверка баланса классов
        
        # Basic statistics
        stats = {
            "total_samples": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict()
        }
        
        # Category distribution
        category_counts = df['Категория'].value_counts()
        stats["category_distribution"] = category_counts.to_dict()
        
        # Text length analysis
        df['text_length'] = df[text_col].astype(str).apply(len)
        df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
        
        stats["text_statistics"] = {
            "avg_text_length": df['text_length'].mean(),
            "median_text_length": df['text_length'].median(),
            "max_text_length": df['text_length'].max(),
            "min_text_length": df['text_length'].min(),
            "avg_word_count": df['word_count'].mean(),
            "median_word_count": df['word_count'].median(),
            "max_word_count": df['word_count'].max(),
            "min_word_count": df['word_count'].min()
        }
        
        # Generate visualizations
        self._create_visualizations(df, category_counts)
        
        return stats
    
    def _create_visualizations(self, df: pd.DataFrame, category_counts: pd.Series):
        """Create and save visualizations"""
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Category distribution
        plt.figure(figsize=(12, 6))
        category_counts.plot(kind='bar')
        plt.title('Распределение категорий рейтингов банков', fontsize=14, pad=20)
        plt.xlabel('Категория рейтинга', fontsize=12)
        plt.ylabel('Количество образцов', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Text length distribution
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(df['text_length'], bins=30, alpha=0.7, color='skyblue')
        plt.title('Распределение длины текста (символы)')
        plt.xlabel('Количество символов')
        plt.ylabel('Частота')
        
        plt.subplot(1, 3, 2)
        plt.hist(df['word_count'], bins=30, alpha=0.7, color='lightgreen')
        plt.title('Распределение количества слов')
        plt.xlabel('Количество слов')
        plt.ylabel('Частота')
        
        plt.subplot(1, 3, 3)
        plt.boxplot(df['text_length'])
        plt.title('Boxplot длины текста')
        plt.ylabel('Количество символов')
        
        plt.tight_layout()
        plt.savefig('text_length_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Category vs text length
        plt.figure(figsize=(12, 6))
        df.boxplot(column='text_length', by='Категория', ax=plt.gca())
        plt.title('Длина текста по категориям')
        plt.suptitle('')  # Remove default title
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('category_vs_length.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_rating_model(self, csv_path: str, model_output_dir: str = "./trained_rating_model",
                          epochs: int = 3, batch_size: int = 16, test_size: float = 0.2):
        """Train the rating classification model with comprehensive analysis"""

        print("Analyzing dataset...")
        dataset_stats = self.analyze_dataset(csv_path)

        print("Training rating classification model...")
        # Очистка текста для обучения
        # Создаем временный очищенный CSV
        cleaned_csv_path = csv_path.replace('.csv', '_cleaned.csv')
        df = pd.read_csv(csv_path)
        text_col = [col for col in df.columns if col not in ['Категория', 'Id', 'id', 'ID']][0]
        df[text_col] = df[text_col].astype(str).apply(self.text_processor.preprocess_for_training)
        df.to_csv(cleaned_csv_path, index=False)

        self.rating_classifier.load_pretrained_model()

        training_results = self.rating_classifier.train_model(
            csv_path=cleaned_csv_path,
            output_dir=model_output_dir,
            epochs=epochs,
            batch_size=batch_size,
            test_size=test_size
        )

        # Create training visualization
        self._create_training_visualizations(training_results)
        
        # Save complete results
        complete_results = {
            "dataset_analysis": dataset_stats,
            "training_results": training_results,
            "model_path": model_output_dir
        }
        
        with open(f"{model_output_dir}/complete_training_results.json", "w", encoding="utf-8") as f:
            json.dump(complete_results, f, ensure_ascii=False, indent=2, default=str)
        
        return complete_results
    
    def _create_training_visualizations(self, training_results: Dict):
        """Create visualizations for training results"""
        classification_report = training_results["classification_report"]
        
        # Extract metrics for visualization
        categories = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for category, metrics in classification_report.items():
            if category not in ['accuracy', 'macro avg', 'weighted avg']:
                categories.append(category)
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1_scores.append(metrics['f1-score'])
        
        # Create metrics visualization
        plt.figure(figsize=(15, 5))
        
        x = np.arange(len(categories))
        width = 0.25
        
        plt.subplot(1, 2, 1)
        plt.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x, recalls, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Категории')
        plt.ylabel('Значение метрики')
        plt.title('Метрики классификации по категориям')
        plt.xticks(x, categories, rotation=45)
        plt.legend()
        
        # Overall metrics
        plt.subplot(1, 2, 2)
        overall_metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
        overall_values = [
            classification_report['accuracy'],
            classification_report['macro avg']['f1-score'],
            classification_report['weighted avg']['f1-score']
        ]
        
        bars = plt.bar(overall_metrics, overall_values, color=['skyblue', 'lightgreen', 'salmon'])
        plt.title('Общие метрики модели')
        plt.ylabel('Значение метрики')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, overall_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_training_report(self, results_path: str):
        """Create a comprehensive training report"""
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        report = f"""
# Отчет об обучении модели классификации рейтингов банков

## Анализ датасета

### Основная статистика
- Общее количество образцов: {results['dataset_analysis']['total_samples']}
- Количество категорий: {len(results['dataset_analysis']['category_distribution'])}
- Средняя длина текста: {results['dataset_analysis']['text_statistics']['avg_text_length']:.0f} символов
- Среднее количество слов: {results['dataset_analysis']['text_statistics']['avg_word_count']:.0f} слов

### Распределение категорий
"""
        
        for category, count in results['dataset_analysis']['category_distribution'].items():
            percentage = (count / results['dataset_analysis']['total_samples']) * 100
            report += f"- {category}: {count} образцов ({percentage:.1f}%)\n"
        
        report += f"""

## Результаты обучения

### Общие метрики
- Accuracy: {results['training_results']['classification_report']['accuracy']:.3f}
- Macro F1-Score: {results['training_results']['classification_report']['macro avg']['f1-score']:.3f}
- Weighted F1-Score: {results['training_results']['classification_report']['weighted avg']['f1-score']:.3f}

### Метрики по категориям
"""
        
        for category, metrics in results['training_results']['classification_report'].items():
            if category not in ['accuracy', 'macro avg', 'weighted avg']:
                report += f"""
#### {category}
- Precision: {metrics['precision']:.3f}
- Recall: {metrics['recall']:.3f}
- F1-Score: {metrics['f1-score']:.3f}
- Support: {metrics['support']}
"""
        
        # Save report
        with open("training_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        return report
