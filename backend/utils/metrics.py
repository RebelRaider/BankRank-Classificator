from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def compute_metrics(pred):
    """
    Расширенная функция для вычисления метрик во время обучения,
    включая F1-score, точность и детализированный отчет
    
    Args:
        pred: Объект предсказаний от Trainer
        
    Returns:
        dict: Словарь с вычисленными метриками
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Вычисляем базовые метрики
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    precision_macro = precision_score(labels, preds, average='macro')
    recall_macro = recall_score(labels, preds, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision_macro,
        'recall': recall_macro
    }

def get_classification_report(y_true, y_pred, target_names=None, output_dict=True):
    """
    Получение детального отчета о классификации
    
    Args:
        y_true (array-like): Истинные метки
        y_pred (array-like): Предсказанные метки
        target_names (list, optional): Список имен классов
        output_dict (bool): Возвращать словарь (True) или строку (False)
        
    Returns:
        dict или str: Отчет о классификации
    """
    return classification_report(
        y_true, 
        y_pred, 
        target_names=target_names,
        output_dict=output_dict
    )
