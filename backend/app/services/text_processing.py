import re
import pymorphy3
from bs4 import BeautifulSoup
from typing import List, Union
import string
from nltk.corpus import stopwords
from transformers import AutoTokenizer
import nltk

class TextPreprocessor:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.russian_stopwords = set(stopwords.words('russian'))
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def clean_text(self, text: str) -> str:
        """Полный цикл предобработки текста"""
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        return text

    def remove_html_tags(self, text: str) -> str:
        return BeautifulSoup(text, 'html.parser').get_text()

    def remove_urls(self, text: str) -> str:
        return re.sub(r'http[s]?://\S+', '', text)

    def remove_emails(self, text: str) -> str:
        return re.sub(r'\S+@\S+', '', text)

    def remove_special_chars(self, text: str) -> str:
        return re.sub(r'[^\w\s]|_', '', text)

    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def remove_stopwords(self, text: str) -> str:
        words = text.split()
        return ' '.join([word for word in words if word.lower() not in self.russian_stopwords])

    def lemmatize_text(self, text: str) -> str:
        words = text.split()
        return ' '.join([self.morph.parse(word)[0].normal_form for word in words])

    def prepare_for_bert(self, text: str, max_length: int = 512) -> str:
        processed = self.clean_text(text)
        tokens = self.tokenizer(
            processed,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return tokens
