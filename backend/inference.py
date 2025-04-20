import os
import torch
import pandas as pd
import numpy as np
import spacy
from transformers import AutoTokenizer, AutoModel
import joblib
from tqdm.auto import tqdm
import re
from bs4 import BeautifulSoup

# Импортируем конкретные параметры из конфигурации
from config import (
    BERT_MODEL_NAME, MAX_SEQ_LENGTH, INFERENCE_BATCH_SIZE,
    MODELS_DIR, DEVICE, STOP_WORDS, TEXT_COLUMN,
    CATEGORY_MODEL_PATH, RATING_MODEL_PATH,
    CATEGORY_ENCODER_PATH, RATING_ENCODER_PATH
)

# Импортируем класс модели из train.py
from train import CreditRatingModel, TextProcessor

# Загружаем модель spaCy для русского языка
try:
    nlp = spacy.load("ru_core_news_lg")
except OSError:
    # Если модель не установлена, загрузим её
    print("Загрузка языковой модели для spaCy...")
    spacy.cli.download("ru_core_news_lg")
    nlp = spacy.load("ru_core_news_lg")

class CreditRatingPredictor:
    """Класс для прогнозирования кредитных рейтингов с помощью обученных моделей"""
    
    def __init__(self, model_dir=MODELS_DIR):
        """
        Инициализация предиктора с обученными моделями
        
        Аргументы:
            model_dir (str): Директория с обученными моделями
        """
        self.model_dir = model_dir
        self.device = DEVICE
        self.processor = TextProcessor()
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        
        print("Загрузка моделей и кодировщиков...")
        
        # Загрузка кодировщиков меток
        self.label_encoder_cat = joblib.load(CATEGORY_ENCODER_PATH)
        self.label_encoder_rat = joblib.load(RATING_ENCODER_PATH)
        
        # Загрузка модели категорий
        self.model_cat = CreditRatingModel(
            BERT_MODEL_NAME, 
            num_classes=len(self.label_encoder_cat.classes_)
        )
        self.model_cat.load_state_dict(torch.load(
            CATEGORY_MODEL_PATH,
            map_location=self.device
        ))
        self.model_cat.to(self.device)
        self.model_cat.eval()
        
        # Загрузка модели рейтингов
        self.model_rat = CreditRatingModel(
            BERT_MODEL_NAME, 
            num_classes=len(self.label_encoder_rat.classes_)
        )
        self.model_rat.load_state_dict(torch.load(
            RATING_MODEL_PATH,
            map_location=self.device
        ))
        self.model_rat.to(self.device)
        self.model_rat.eval()
        
        print("Модели успешно загружены")
    
    def predict_single(self, text):
        """
        Делает прогноз для одного текста
        
        Аргументы:
            text (str): Входной текст
        
        Возвращает:
            dict: Словарь с предсказанной категорией и рейтингом
        """
        # Предобработка текста
        clean_text = self.processor.clear_text(text)
        features = self.processor.get_features(clean_text)
        
        # Токенизация текста
        encoding = self.tokenizer(
            clean_text,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Перемещение на устройство
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        features_tensor = torch.tensor([features], dtype=torch.float).to(self.device)
        
        # Получение предсказаний
        with torch.no_grad():
            # Предсказание категории
            cat_outputs = self.model_cat(input_ids, attention_mask, features_tensor)
            cat_prediction = torch.argmax(cat_outputs, dim=1).item()
            cat_label = self.label_encoder_cat.inverse_transform([cat_prediction])[0]
            
            # Предсказание рейтинга
            rat_outputs = self.model_rat(input_ids, attention_mask, features_tensor)
            rat_prediction = torch.argmax(rat_outputs, dim=1).item()
            rat_label = self.label_encoder_rat.inverse_transform([rat_prediction])[0]
        
        return {
            'Категория': cat_label,
            'Уровень рейтинга': rat_label
        }
    
    def predict_batch(self, texts, batch_size=INFERENCE_BATCH_SIZE):
        """
        Делает прогнозы для пакета текстов
        
        Аргументы:
            texts (list): Список входных текстов
            batch_size (int): Размер пакета для прогнозирования
        
        Возвращает:
            pd.DataFrame: DataFrame с предсказанными категориями и рейтингами
        """
        # Предобработка текстов
        clean_texts = [self.processor.clear_text(text) for text in texts]
        features_list = [self.processor.get_features(text) for text in tqdm(clean_texts, desc="Извлечение признаков")]
        
        # Инициализация результатов
        categories = []
        ratings = []
        
        # Обработка пакетами
        for i in tqdm(range(0, len(clean_texts), batch_size), desc="Прогнозирование"):
            batch_texts = clean_texts[i:i+batch_size]
            batch_features = features_list[i:i+batch_size]
            
            # Токенизация пакета
            encodings = self.tokenizer(
                batch_texts,
                max_length=MAX_SEQ_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Перемещение на устройство
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            features_tensor = torch.tensor(batch_features, dtype=torch.float).to(self.device)
            
            # Получение предсказаний
            with torch.no_grad():
                # Предсказания категорий
                cat_outputs = self.model_cat(input_ids, attention_mask, features_tensor)
                cat_predictions = torch.argmax(cat_outputs, dim=1).cpu().numpy()
                batch_categories = self.label_encoder_cat.inverse_transform(cat_predictions)
                
                # Предсказания рейтингов
                rat_outputs = self.model_rat(input_ids, attention_mask, features_tensor)
                rat_predictions = torch.argmax(rat_outputs, dim=1).cpu().numpy()
                batch_ratings = self.label_encoder_rat.inverse_transform(rat_predictions)
            
            # Добавление к результатам
            categories.extend(batch_categories)
            ratings.extend(batch_ratings)
        
        # Создание DataFrame с результатами
        results_df = pd.DataFrame({
            'Категория': categories,
            'Уровень рейтинга': ratings
        })
        
        return results_df
    
    def predict_from_dataframe(self, df, text_column=TEXT_COLUMN, output_path=None):
        """
        Обработка DataFrame с текстами и сохранение предсказаний
        
        Аргументы:
            df (pd.DataFrame): DataFrame с текстами для прогнозирования
            text_column (str): Имя столбца, содержащего тексты
            output_path (str, опционально): Путь для сохранения результатов в Excel
            
        Возвращает:
            pd.DataFrame: DataFrame с исходными текстами и предсказаниями
        """
        # Проверка существования столбца с текстом
        if text_column not in df.columns:
            raise ValueError(f"Столбец '{text_column}' не найден в DataFrame")
        
        # Получение текстов из DataFrame
        texts = df[text_column].tolist()
        
        # Получение предсказаний
        predictions_df = self.predict_batch(texts)
        
        # Объединение с исходными текстами
        result_df = pd.concat([df[[text_column]], predictions_df], axis=1)
        
        # Сохранение в Excel, если указан путь вывода
        if output_path:
            print(f"Сохранение предсказаний в {output_path}")
            result_df.to_excel(output_path, index=False)
        
        return result_df

def main():
    """Основная функция для использования в командной строке"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Прогнозирование кредитных рейтингов')
    parser.add_argument('--input', type=str, required=True, 
                       help='Путь к Excel-файлу с текстами для прогнозирования')
    parser.add_argument('--output', type=str, required=True, 
                       help='Путь для сохранения выходного Excel-файла')
    parser.add_argument('--model_dir', type=str, default=MODELS_DIR, 
                       help='Директория с обученными моделями')
    parser.add_argument('--text_column', type=str, default=TEXT_COLUMN, 
                       help='Имя столбца, содержащего тексты')
    
    args = parser.parse_args()
    
    # Загрузка входных данных
    print(f"Загрузка данных из {args.input}")
    df = pd.read_excel(args.input)
    
    # Инициализация предиктора и выполнение предсказаний
    predictor = CreditRatingPredictor(model_dir=args.model_dir)
    predictor.predict_from_dataframe(
        df, 
        text_column=args.text_column, 
        output_path=args.output
    )
    
    print(f"Предсказания сохранены в {args.output}")

if __name__ == "__main__":
    main()
