import torch
from torch.utils.data import Dataset

class CreditRatingDataset(Dataset):
    """
    Улучшенный набор данных для прогнозирования кредитных рейтингов,
    поддерживающий дополнительные текстовые признаки
    """
    
    def __init__(self, texts, features, labels, tokenizer, max_length=512):
        """
        Инициализация набора данных
        
        Args:
            texts (list): Список текстов
            features (list): Список признаков или None
            labels (list): Список меток
            tokenizer: Токенизатор для обработки текста
            max_length (int): Максимальная длина последовательности
        """
        self.texts = texts
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_features = features is not None
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Токенизация текста
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Создаем базовый словарь элемента
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        # Если есть дополнительные признаки, добавляем их
        if self.use_features:
            item['features'] = torch.tensor(self.features[idx], dtype=torch.float)
        
        return item
