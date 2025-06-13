import re
import pymorphy3
from bs4 import BeautifulSoup
from typing import List
import string
from nltk.corpus import stopwords
from transformers import AutoTokenizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextProcessor:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.russian_stopwords = set(stopwords.words('russian'))
        self.toxicity_tokenizer = AutoTokenizer.from_pretrained("s-nlp/russian_toxicity_classifier")
        self.rating_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        
    def clean_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def remove_special_chars(self, text: str, keep_punct: bool = False) -> str:
        """Remove special characters"""
        if keep_punct:
            # Keep basic punctuation for BERT
            pattern = r'[^\w\s.,!?;:()-]'
        else:
            pattern = r'[^\w\s]'
        return re.sub(pattern, '', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace - remove extra spaces, tabs, newlines"""
        # Replace multiple whitespace characters with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """Remove Russian stopwords"""
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.russian_stopwords]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize Russian text using pymorphy3"""
        words = text.split()
        lemmatized_words = []
        
        for word in words:
            # Skip short words and punctuation
            if len(word) < 2 or word in string.punctuation:
                lemmatized_words.append(word)
                continue
                
            parsed = self.morph.parse(word)[0]
            lemmatized_words.append(parsed.normal_form)
        
        return ' '.join(lemmatized_words)
    
    def preprocess_for_bert(self, text: str, max_length: int = 512) -> str:
        """Preprocess text for BERT models"""
        # Basic cleaning
        text = self.clean_html(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_special_chars(text, keep_punct=True)
        text = self.normalize_whitespace(text)
        
        # Truncate if too long (leave space for special tokens)
        if len(text) > max_length - 2:
            text = text[:max_length-2]
        
        return text
    
    def preprocess_for_training(self, text: str, remove_stopwords: bool = True, 
                               lemmatize: bool = True) -> str:
        """Comprehensive preprocessing for model training"""
        text = self.clean_html(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_special_chars(text, keep_punct=True)
        text = self.normalize_whitespace(text)
        
        if remove_stopwords:
            text = self.remove_stopwords(text)
            
        if lemmatize:
            text = self.lemmatize_text(text)
            
        return text
    
    def prepare_for_toxicity_model(self, text: str) -> str:
        """Prepare text specifically for toxicity classification"""
        return self.preprocess_for_bert(text, max_length=512)
    
    def prepare_for_rating_model(self, text: str) -> str:
        """Prepare text specifically for rating classification"""
        return self.preprocess_for_bert(text, max_length=512)
    
    def batch_preprocess(self, texts: List[str], preprocessing_type: str = "bert") -> List[str]:
        """Batch preprocessing for multiple texts"""
        if preprocessing_type == "bert":
            return [self.preprocess_for_bert(text) for text in texts]
        elif preprocessing_type == "training":
            return [self.preprocess_for_training(text) for text in texts]
        elif preprocessing_type == "toxicity":
            return [self.prepare_for_toxicity_model(text) for text in texts]
        elif preprocessing_type == "rating":
            return [self.prepare_for_rating_model(text) for text in texts]
        else:
            raise ValueError(f"Unknown preprocessing type: {preprocessing_type}")
