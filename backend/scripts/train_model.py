#!/usr/bin/env python3
"""
Скрипт для обучения модели классификации рейтингов банков
"""
import argparse
import sys
import os

# Добавляем корень проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.model_trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description='Обучение модели классификации рейтингов банков')
    parser.add_argument('--csv_path', required=True, help='Путь к CSV файлу с данными')
    parser.add_argument('--model_output_dir', default='./trained_rating_model', 
                       help='Директория для сохранения обученной модели')
    parser.add_argument('--epochs', type=int, default=3, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=16, help='Размер батча')
    parser.add_argument('--test_size', type=float, default=0.2, help='Доля тестовой выборки')
    
    args = parser.parse_args()
    
    print("🚀 Запуск обучения модели классификации рейтингов банков")
    print(f"📁 Путь к данным: {args.csv_path}")
    print(f"💾 Модель будет сохранена в: {args.model_output_dir}")
    print(f"⚙️ Параметры: эпохи={args.epochs}, батч={args.batch_size}, тест={args.test_size}")
    
    # Создаем тренер
    trainer = ModelTrainer()
    
    try:
        # Обучаем модель
        results = trainer.train_rating_model(
            csv_path=args.csv_path,
            model_output_dir=args.model_output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            test_size=args.test_size
        )
        
        # Создаем отчет
        report = trainer.create_training_report(
            f"{args.model_output_dir}/complete_training_results.json"
        )
        
        print("\n✅ Обучение успешно завершено!")
        print(f"📊 Точность модели: {results['training_results']['classification_report']['accuracy']:.3f}")
        print(f"📈 F1-Score (weighted): {results['training_results']['classification_report']['weighted avg']['f1-score']:.3f}")
        print("📋 Отчет сохранен в: training_report.md")
        print("📊 Графики сохранены: category_distribution.png, text_length_analysis.png, training_metrics.png")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении модели: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
