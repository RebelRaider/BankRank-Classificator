import torch.nn as nn
from transformers import Trainer

class FeatureTrainer(Trainer):
    """
    Расширенный Trainer для работы с моделями, использующими
    дополнительные числовые признаки
    """
    
    def __init__(self, use_features=False, *args, **kwargs):
        """
        Инициализация тренера
        
        Args:
            use_features (bool): Использовать ли дополнительные признаки
            *args, **kwargs: Аргументы для базового класса Trainer
        """
        super(FeatureTrainer, self).__init__(*args, **kwargs)
        self.use_features = use_features
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Вычисление функции потерь с поддержкой дополнительных признаков
        
        Args:
            model: Модель
            inputs (dict): Входные данные
            return_outputs (bool): Возвращать ли выходы модели
            
        Returns:
            torch.Tensor или tuple: Потери или (потери, выходы)
        """
        # Извлекаем данные из входов
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        features = inputs.get("features") if self.use_features else None
        
        # Прямой проход модели
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            features=features,
            labels=labels
        )
        
        # Если модель возвращает словарь с потерями и логитами
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
            logits = outputs.get("logits")
        else:
            # Если модель возвращает только логиты
            logits = outputs
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, {"logits": logits}) if return_outputs else loss
