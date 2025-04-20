import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class CombinedBertForSequenceClassification(nn.Module):
    """
    Расширенная модель BERT для классификации последовательностей с
    интеграцией дополнительных численных признаков
    """
    
    def __init__(self, model_name, num_labels, feature_dim=None, id2label=None, label2id=None):
        """
        Инициализация модели
        
        Args:
            model_name (str): Название предобученной модели BERT
            num_labels (int): Количество классов для классификации
            feature_dim (int, optional): Размерность дополнительных признаков
            id2label (dict, optional): Словарь для преобразования индекса в метку
            label2id (dict, optional): Словарь для преобразования метки в индекс
        """
        super(CombinedBertForSequenceClassification, self).__init__()
        
        # Создаем маппинг для классов, если они предоставлены
        config_kwargs = {
            'num_labels': num_labels,
            'return_dict': True
        }
        
        if id2label and label2id:
            config_kwargs['id2label'] = id2label
            config_kwargs['label2id'] = label2id
        
        # Загружаем предобученную модель BERT
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            **config_kwargs
        )
        
        # Если есть дополнительные признаки, настраиваем их обработку
        self.use_features = feature_dim is not None
        if self.use_features:
            # Извлекаем размерность скрытого состояния BERT
            bert_hidden_size = self.bert.config.hidden_size
            
            # Слой для обработки дополнительных признаков
            self.feature_projection = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # Заменяем классификатор в BERT на наш, который объединяет оба типа признаков
            self.bert.classifier = nn.Sequential(
                nn.Linear(bert_hidden_size + 128, bert_hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(bert_hidden_size, num_labels)
            )
    
    def forward(self, input_ids, attention_mask, features=None, labels=None):
        """
        Прямой проход модели
        
        Args:
            input_ids (torch.Tensor): Индексы токенов
            attention_mask (torch.Tensor): Маска внимания
            features (torch.Tensor, optional): Дополнительные признаки
            labels (torch.Tensor, optional): Метки
            
        Returns:
            dict: Словарь с результатами (потери и логиты) или только логиты
        """
        # Получаем выход BERT (без финального классификатора)
        bert_outputs = self.bert.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled_output = bert_outputs.pooler_output
        
        # Если используем дополнительные признаки, объединяем их с BERT
        if self.use_features and features is not None:
            features_projected = self.feature_projection(features)
            combined_features = torch.cat((pooled_output, features_projected), dim=1)
        else:
            combined_features = pooled_output
        
        # Применяем классификатор к объединенным признакам
        logits = self.bert.classifier(combined_features)
        
        # Если предоставлены метки, вычисляем потери
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.bert.config.num_labels), labels.view(-1))
        
        return {'loss': loss, 'logits': logits} if loss is not None else logits
    
    def save_pretrained(self, path):
        """
        Сохранение модели вместе с конфигурацией
        
        Args:
            path (str): Путь для сохранения модели
        """
        self.bert.save_pretrained(path)
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
    
    @classmethod
    def from_pretrained(cls, path, feature_dim=None):
        """
        Загрузка модели из сохраненной
        
        Args:
            path (str): Путь к сохраненной модели
            feature_dim (int, optional): Размерность дополнительных признаков
            
        Returns:
            CombinedBertForSequenceClassification: Загруженная модель
        """
        # Загружаем конфигурацию BERT
        bert = AutoModelForSequenceClassification.from_pretrained(path)
        
        # Создаем и инициализируем модель
        model = cls(
            model_name=path,
            num_labels=bert.config.num_labels,
            feature_dim=feature_dim,
            id2label=bert.config.id2label if hasattr(bert.config, 'id2label') else None,
            label2id=bert.config.label2id if hasattr(bert.config, 'label2id') else None
        )
        
        # Загружаем веса
        model.load_state_dict(torch.load(f"{path}/pytorch_model.bin"))
        return model
