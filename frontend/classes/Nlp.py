import re
import time

import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup  # Для удаления HTML тегов

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import Embedding, Flatten, Input, concatenate, Reshape
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import tensorflow_text as text

from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    NamesExtractor,
    DatesExtractor,
    MoneyExtractor,
    AddrExtractor,

    Doc
)

from matplotlib import pyplot as plt


class Nlp:
    def __init__(self):
        # Natasha
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        self.names_extractor = NamesExtractor(self.morph_vocab)
        self.dates_extractor = DatesExtractor(self.morph_vocab)
        self.money_extractor = MoneyExtractor(self.morph_vocab)
        self.addr_extractor = AddrExtractor(self.morph_vocab)
        self.model_rat = False  # 
        self.labels_rat = ['A+', 'A', 'A-', 'AA+', 'AA', 'AA-', 'AAA', 'B+', 'B', 'B-', 'BB+', 'BB', 'BB-', 'BBB+', 'BBB', 'BBB-', 'C']
        self.le = LabelEncoder()
        self.le.fit(self.labels_rat)


    # Очищаем текст регулярными выражениями
    def clear_text(self, text):
        soup = BeautifulSoup(text)
        text = soup.get_text()
        text = re.sub(r'(http\S+)|(www\S+)|([\w\d]+www\S+)|([\w\d]+http\S+)', '', text)
        text = re.sub(r'[\n\t]', ' ', text).strip()  # Перенос, табуляция
        text = re.sub(r'[^\w\d\s\.\,]', ' ', text)  # Только слова, цифры, пробелы, точки и запятые
        text = re.sub(r'\s+', ' ', text)  # Удаляем двойные пробелы
        return text


    # Извлекаем фичи
    def get_features(self, text):
        # Облегченная версия, не подтягиваем Natasha
        features_list = []
        words = text.split()
        count = len(words)  # Количество слов в строке
        average = sum(len(word) for word in words) / count
        uniq = round(100*len(set(words))/count)  # % уникальных слов в строке
        features_list.append([count, average, uniq])
        features = pd.DataFrame(features_list, columns=['count', 'average', 'uniq'])
        return features


    # В категорийные признаки
    def to_categorical(self, df, labels):
        le = LabelEncoder()
        le.fit(labels)
        label = le.transform(df)
        return to_categorical(label, num_classes=len(labels), dtype='int')


    # Возвращает список с количеством найденных именованных сущностей [names, dates, LOC, MONEY]
    def get_ner_features(self, text):
        names = len(list(nlp.names_extractor(text)))
        money = len(list(nlp.money_extractor(text)))
        addr = len(list(nlp.addr_extractor(text)))
        # dates = len(list(nlp.dates_extractor(text)))  # Выдаёт ошибку на текст с 'маю'
        # return [names, dates, money, addr]
        return [names, money, addr]


    # Извлекает именнованные сущности - имена, названия
    def names_extractor(self, text):
        return self.names_extractor(text)


    # Извлекает именнованные сущности - даты
    def dates_extractor(self, text):
        return self.dates_extractor(text)


    # Извлекает именнованные сущности - деньги
    def money_extractor(self, text):
        return self.money_extractor(text)


    # Извлекает именнованные сущности - локацию, адреса
    def addr_extractor(self, text):
        return self.addr_extractor(text)