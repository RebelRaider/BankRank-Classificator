import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from bs4 import BeautifulSoup
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import os
from tqdm.auto import tqdm

# Импортируем конкретные параметры из конфигурации
from config import (
    TRAIN_DATASET_PATH, TEST_DATASET_PATH, DATASET_DIR,
    TEXT_COLUMN, CATEGORY_COLUMN, RATING_COLUMN,
    STOP_WORDS, RANDOM_SEED, FIGURE_SIZE, DPI
)

# Загружаем модель spaCy для русского языка
try:
    nlp = spacy.load("ru_core_news_lg")
except OSError:
    print("Загрузка языковой модели для spaCy...")
    spacy.cli.download("ru_core_news_lg")
    nlp = spacy.load("ru_core_news_lg")

class DataAnalyzer:
    """Класс для анализа набора данных кредитных рейтингов"""
    
    def __init__(self, data_path=None, df=None):
        """
        Инициализация анализатора с набором данных
        
        Аргументы:
            data_path (str, опционально): Путь к файлу с данными
            df (pd.DataFrame, опционально): DataFrame с данными
        """
        if df is not None:
            self.df = df
        elif data_path is not None:
            self.load_data(data_path)
        else:
            self.df = None
        
        self.text_processor = TextProcessor()
    
    def load_data(self, data_path):
        """
        Загрузка набора данных из файла
        
        Аргументы:
            data_path (str): Путь к файлу с данными
        """
        if data_path.endswith('.csv'):
            self.df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(data_path)
        else:
            raise ValueError("Неподдерживаемый формат файла. Используйте CSV или Excel.")
        
        print(f"Загружены данные размерностью: {self.df.shape}")
        return self.df
    
    def get_basic_stats(self):
        """Получение базовой статистики набора данных"""
        if self.df is None:
            raise ValueError("Данные не загружены. Сначала загрузите данные.")
        
        print("=== Обзор набора данных ===")
        print(f"Количество записей: {len(self.df)}")
        print(f"Столбцы: {', '.join(self.df.columns)}")
        print("\n=== Типы данных ===")
        print(self.df.dtypes)
        print("\n=== Пропущенные значения ===")
        print(self.df.isnull().sum())
        
        # Показать образцы строк
        print("\n=== Образец данных ===")
        print(self.df.head())
        
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes,
            "missing_values": self.df.isnull().sum()
        }
    
    def analyze_labels_distribution(self, plot=True):
        """
        Анализ распределения категорий и рейтингов
        
        Аргументы:
            plot (bool): Создавать ли графики
            
        Возвращает:
            dict: Количество каждой категории и рейтинга
        """
        if self.df is None:
            raise ValueError("Данные не загружены. Сначала загрузите данные.")
        
        # Подсчет категорий и рейтингов
        cat_counts = self.df[CATEGORY_COLUMN].value_counts()
        rat_counts = self.df[RATING_COLUMN].value_counts()
        
        print(f"=== Распределение {CATEGORY_COLUMN} ===")
        print(cat_counts)
        print(f"\n=== Распределение {RATING_COLUMN} ===")
        print(rat_counts)
        
        if plot:
            # Создание фигуры с двумя графиками
            fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
            
            # График распределения категорий
            cat_counts.plot(kind='bar', ax=axes[0], color='skyblue')
            axes[0].set_title(f'Распределение {CATEGORY_COLUMN}')
            axes[0].set_ylabel('Количество')
            axes[0].tick_params(axis='x', rotation=45)
            
            # График распределения рейтингов
            rat_counts.plot(kind='bar', ax=axes[1], color='lightgreen')
            axes[1].set_title(f'Распределение {RATING_COLUMN}')
            axes[1].set_ylabel('Количество')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            # Создание круговых диаграмм
            fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
            
            # Круговая диаграмма категорий
            axes[0].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', 
                      explode=[0.05] * len(cat_counts), shadow=True, startangle=90)
            axes[0].set_title(f'Распределение {CATEGORY_COLUMN}')
            
            # Круговая диаграмма рейтингов
            axes[1].pie(rat_counts, labels=rat_counts.index, autopct='%1.1f%%', 
                       explode=[0.05] * len(rat_counts), shadow=True, startangle=90)
            axes[1].set_title(f'Распределение {RATING_COLUMN}')
            
            plt.tight_layout()
            plt.show()
        
        return {
            "category_counts": cat_counts.to_dict(),
            "rating_counts": rat_counts.to_dict()
        }
    
    def analyze_text_length(self, column=TEXT_COLUMN, plot=True):
        """
        Анализ длины текста в указанном столбце
        
        Аргументы:
            column (str): Столбец с текстом
            plot (bool): Создавать ли графики
            
        Возвращает:
            dict: Статистика о длине текста
        """
        if self.df is None:
            raise ValueError("Данные не загружены. Сначала загрузите данные.")
        
        # Вычисление длины текста
        self.df['text_length'] = self.df[column].apply(lambda x: len(str(x)))
        self.df['word_count'] = self.df[column].apply(lambda x: len(str(x).split()))
        
        # Расчет статистики
        length_stats = {
            "mean_length": self.df['text_length'].mean(),
            "median_length": self.df['text_length'].median(),
            "min_length": self.df['text_length'].min(),
            "max_length": self.df['text_length'].max(),
            "std_length": self.df['text_length'].std(),
            "mean_word_count": self.df['word_count'].mean(),
            "median_word_count": self.df['word_count'].median(),
            "min_word_count": self.df['word_count'].min(),
            "max_word_count": self.df['word_count'].max()
        }
        
        print("=== Статистика длины текста ===")
        print(f"Средняя длина: {length_stats['mean_length']:.2f} символов")
        print(f"Медианная длина: {length_stats['median_length']} символов")
        print(f"Минимальная длина: {length_stats['min_length']} символов")
        print(f"Максимальная длина: {length_stats['max_length']} символов")
        print(f"Стандартное отклонение: {length_stats['std_length']:.2f} символов")
        print("\n=== Статистика количества слов ===")
        print(f"Среднее количество слов: {length_stats['mean_word_count']:.2f} слов")
        print(f"Медианное количество слов: {length_stats['median_word_count']} слов")
        print(f"Минимальное количество слов: {length_stats['min_word_count']} слов")
        print(f"Максимальное количество слов: {length_stats['max_word_count']} слов")
        
        if plot:
            # Создание фигуры с гистограммами
            fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
            
            # График гистограммы длины текста
            sns.histplot(self.df['text_length'], bins=30, kde=True, ax=axes[0], color='skyblue')
            axes[0].set_title('Распределение длины текста')
            axes[0].set_xlabel('Количество символов')
            axes[0].set_ylabel('Частота')
            
            # График гистограммы количества слов
            sns.histplot(self.df['word_count'], bins=30, kde=True, ax=axes[1], color='lightgreen')
            axes[1].set_title('Распределение количества слов')
            axes[1].set_xlabel('Количество слов')
            axes[1].set_ylabel('Частота')
            
            plt.tight_layout()
            plt.show()
            
            # Создание диаграмм размаха по категориям
            fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
            
            # Длина текста по категориям
            sns.boxplot(x=CATEGORY_COLUMN, y='text_length', data=self.df, ax=axes[0])
            axes[0].set_title('Длина текста по категориям')
            axes[0].set_xlabel('Категория')
            axes[0].set_ylabel('Количество символов')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Количество слов по категориям
            sns.boxplot(x=CATEGORY_COLUMN, y='word_count', data=self.df, ax=axes[1])
            axes[1].set_title('Количество слов по категориям')
            axes[1].set_xlabel('Категория')
            axes[1].set_ylabel('Количество слов')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
        
        return length_stats
    
    def extract_features(self, column=TEXT_COLUMN, sample_size=None):
        """
        Извлечение лингвистических признаков из текста
        
        Аргументы:
            column (str): Столбец с текстом
            sample_size (int, опционально): Количество образцов для обработки (None для всех)
            
        Возвращает:
            pd.DataFrame: DataFrame с извлеченными признаками
        """
        if self.df is None:
            raise ValueError("Данные не загружены. Сначала загрузите данные.")
        
        # Использовать выборку, если указано
        if sample_size is not None and sample_size < len(self.df):
            df_sample = self.df.sample(sample_size, random_state=RANDOM_SEED)
        else:
            df_sample = self.df
        
        print(f"Извлечение признаков из {len(df_sample)} текстов...")
        
        # Очистка текстов
        clean_texts = [self.text_processor.clear_text(text) for text in tqdm(df_sample[column], desc="Очистка текстов")]
        
        # Извлечение признаков
        features_list = [self.text_processor.get_features(text) for text in tqdm(clean_texts, desc="Извлечение признаков")]
        
        # Создание DataFrame с признаками
        features_df = pd.DataFrame(
            features_list, 
            columns=['count', 'average_word_length', 'unique_percentage', 'org_count', 'loc_count', 'per_count']
        )
        
        # Добавление к DataFrame с выборкой
        result_df = df_sample.copy()
        result_df['clean_text'] = clean_texts
        result_df = pd.concat([result_df, features_df], axis=1)
        
        print("Извлечение признаков завершено.")
        return result_df
    
    def save_processed_data(self, output_path):
        """
        Сохранение обработанного DataFrame в файл
        
        Аргументы:
            output_path (str): Путь для сохранения обработанных данных
        """
        if self.df is None:
            raise ValueError("Нет данных для сохранения.")
        
        if output_path.endswith('.csv'):
            self.df.to_csv(output_path, index=False)
        elif output_path.endswith(('.xlsx', '.xls')):
            self.df.to_excel(output_path, index=False)
        else:
            raise ValueError("Неподдерживаемый формат файла. Используйте CSV или Excel.")
        
        print(f"Данные сохранены в {output_path}")


class TextProcessor:
    """Класс для обработки текста и извлечения признаков"""
    
    def __init__(self):
        """Инициализация обработчика текста"""
        self.stop_words = STOP_WORDS
    
    def clear_text(self, text):
        """Очистка текста от HTML, ссылок и ненужных символов"""
        # Обработка None или пустых строк
        if not text or pd.isna(text):
            return ""
            
        # Удаление HTML-тегов
        soup = BeautifulSoup(str(text), features="html.parser")
        text = soup.get_text()
        
        # Удаление URL-адресов
        text = re.sub(r'(http\S+)|(www\S+)|([\w\d]+www\S+)|([\w\d]+http\S+)', '', text)
        
        # Удаление специальных символов и нормализация пробелов
        text = re.sub(r'[\n\t\«]', ' ', text).strip()
        text = re.sub(r'[^\w\d\s\.\,\"]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Исправление пробелов вокруг знаков пунктуации
        pat = "\s+([{}]+)".format(re.escape(".,"))
        text = re.sub("\s{2,}", " ", re.sub(pat, r"\1", text))
        
        return text
    
    def get_features(self, text):
        """Извлечение NLP-признаков из текста, включая именованные сущности"""
        # Обработка пустых строк
        if not text or text.strip() == "":
            return [0, 0, 0, 0, 0, 0]
            
        # Обработка текста с помощью spaCy
        doc = nlp(text)
        
        # Извлечение релевантных токенов (исключая определенные части речи)
        words = [token.text for token in doc if token.pos_ not in ["ADP", "PUNCT", "NUM", "CCONJ", "PROPN"]]
        
        # Расчет статистики текста
        count = len(words)
        if count > 0:
            average = sum(len(word) for word in words) / count
            uniq = round(100 * len(set(words)) / count)  # % уникальных слов
        else:
            average = 0
            uniq = 0
            
        # Подсчет именованных сущностей по типу
        org_count = sum(1 for ent in doc.ents if ent.label_ == "ORG" and ent.text not in self.stop_words)
        loc_count = sum(1 for ent in doc.ents if ent.label_ == "LOC")
        per_count = sum(1 for ent in doc.ents if ent.label_ == "PER")
        
        return [count, average, uniq, org_count, loc_count, per_count]


def main():
    """Основная функция для запуска анализа данных"""
    # Проверка существования набора данных
    if os.path.exists(TRAIN_DATASET_PATH):
        data_path = TRAIN_DATASET_PATH
    else:
        # Поиск Excel-файлов в директории набора данных
        excel_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(('.xlsx', '.xls'))]
        if excel_files:
            data_path = os.path.join(DATASET_DIR, excel_files[0])
        else:
            print(f"Набор данных не найден по пути {TRAIN_DATASET_PATH}")
            return
    
    # Создание анализатора и загрузка данных
    analyzer = DataAnalyzer(data_path)
    
    # Выполнение анализа
    analyzer.get_basic_stats()
    analyzer.analyze_labels_distribution()
    analyzer.analyze_text_length()
    
    # Извлечение и анализ признаков (ограничимся 100 примерами для быстрой демонстрации)
    features_df = analyzer.extract_features(sample_size=100)
    analyzer.analyze_features(features_df)
    
    # Генерация облаков слов
    analyzer.generate_word_clouds()
    
    # Анализ наиболее часто встречающихся слов
    analyzer.analyze_most_common_words()
    
    # Сохранение обработанных данных
    processed_path = os.path.join(DATASET_DIR, "processed_data.csv")
    analyzer.save_processed_data(processed_path)


if __name__ == "__main__":
    main()
