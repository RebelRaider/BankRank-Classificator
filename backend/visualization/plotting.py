import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def analyze_and_plot_data(df, category_col, rating_col, text_col, output_dir, logger, figure_size=(12, 8), dpi=100):
    """
    Функция для анализа и визуализации данных перед обучением
    
    Args:
        df (pandas.DataFrame): Датафрейм с данными
        category_col (str): Название столбца с категориями
        rating_col (str): Название столбца с рейтингами
        text_col (str): Название столбца с текстом
        output_dir (str): Директория для сохранения визуализаций
        logger: Логгер
        figure_size (tuple): Размер фигуры
        dpi (int): Разрешение изображения
        
    Returns:
        dict: Статистика данных
    """
    logger.info("Анализ данных...")
    
    # Создаем директорию для отчетов, если она не существует
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Распределение категорий
    plt.figure(figsize=figure_size)
    category_counts = df[category_col].value_counts()
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title('Распределение категорий')
    plt.xlabel('Категория')
    plt.ylabel('Количество')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'category_distribution.png'), dpi=dpi)
    plt.close()
    
    # 2. Распределение рейтингов
    plt.figure(figsize=figure_size)
    rating_counts = df[rating_col].value_counts()
    sns.barplot(x=rating_counts.index, y=rating_counts.values)
    plt.title('Распределение рейтингов')
    plt.xlabel('Рейтинг')
    plt.ylabel('Количество')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'rating_distribution.png'), dpi=dpi)
    plt.close()
    
    # 3. Статистика текста
    text_lengths = df[text_col].str.len()
    plt.figure(figsize=figure_size)
    sns.histplot(text_lengths, bins=30, kde=True)
    plt.title('Распределение длины текстов')
    plt.xlabel('Длина текста (символы)')
    plt.ylabel('Количество')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'text_length_distribution.png'), dpi=dpi)
    plt.close()
    
    # Запись статистики в лог
    logger.info(f"Количество образцов: {len(df)}")
    logger.info(f"Количество уникальных категорий: {len(category_counts)}")
    logger.info(f"Количество уникальных рейтингов: {len(rating_counts)}")
    logger.info(f"Средняя длина текста: {text_lengths.mean():.2f} символов")
    logger.info(f"Медианная длина текста: {text_lengths.median():.2f} символов")
    logger.info(f"Минимальная длина текста: {text_lengths.min():.2f} символов")
    logger.info(f"Максимальная длина текста: {text_lengths.max():.2f} символов")
    
    # Отчет о распределении категорий
    cat_report = category_counts.reset_index()
    cat_report.columns = ['Категория', 'Количество']
    cat_report['Процент'] = cat_report['Количество'] / cat_report['Количество'].sum() * 100
    logger.info("Распределение категорий:")
    for _, row in cat_report.iterrows():
        logger.info(f"  {row['Категория']}: {row['Количество']} ({row['Процент']:.2f}%)")
    
    # Отчет о распределении рейтингов
    rating_report = rating_counts.reset_index()
    rating_report.columns = ['Рейтинг', 'Количество']
    rating_report['Процент'] = rating_report['Количество'] / rating_report['Количество'].sum() * 100
    logger.info("Распределение рейтингов:")
    for _, row in rating_report.iterrows():
        logger.info(f"  {row['Рейтинг']}: {row['Количество']} ({row['Процент']:.2f}%)")
    
    return {
        'category_counts': category_counts,
        'rating_counts': rating_counts,
        'text_lengths': text_lengths
    }

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path, figure_size=(12, 8), dpi=100):
    """
    Построение и сохранение матрицы ошибок
    
    Args:
        y_true (array-like): Истинные метки
        y_pred (array-like): Предсказанные метки
        classes (list): Список названий классов
        title (str): Заголовок графика
        save_path (str): Путь для сохранения графика
        figure_size (tuple): Размер фигуры
        dpi (int): Разрешение изображения
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figure_size)
    
    # Нормализация для процентного отображения
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Создаем тепловую карту
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title(title)
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()

def plot_training_metrics(trainer_state, model_name, output_dir, figure_size=(12, 8), dpi=100):
    """
    Построение графиков метрик обучения из состояния Trainer
    с улучшенной визуализацией
    
    Args:
        trainer_state: Состояние обучения из Trainer
        model_name (str): Название модели для заголовков
        output_dir (str): Директория для сохранения графика
        figure_size (tuple): Размер фигуры
        dpi (int): Разрешение изображения
        
    Returns:
        str: Путь к сохраненному графику
    """
    plt.figure(figsize=figure_size)
    
    # График потерь
    plt.subplot(2, 1, 1)
    train_history = trainer_state.log_history
    
    # Извлекаем данные о потерях
    train_steps = []
    train_losses = []
    
    for entry in train_history:
        if 'loss' in entry:
            train_steps.append(entry.get('step', 0))
            train_losses.append(entry.get('loss'))
    
    plt.plot(train_steps, train_losses, 'b-', label='Потери обучения')
    plt.title(f'Потери модели {model_name}')
    plt.xlabel('Шаг')
    plt.ylabel('Потери')
    plt.legend()
    
    # График метрик
    eval_metrics = [x for x in train_history if 'eval_accuracy' in x]
    if eval_metrics:
        plt.subplot(2, 1, 2)
        steps = [x.get('step', i) for i, x in enumerate(eval_metrics)]
        
        metrics_to_plot = [
            ('eval_accuracy', 'Точность', 'g-'),
            ('eval_f1_macro', 'F1 (macro)', 'r-'),
            ('eval_f1_weighted', 'F1 (weighted)', 'b-'),
            ('eval_precision', 'Precision', 'm-'),
            ('eval_recall', 'Recall', 'c-')
        ]
        
        for metric_name, label, style in metrics_to_plot:
            if any(metric_name in x for x in eval_metrics):
                values = [x.get(metric_name, 0) for x in eval_metrics]
                plt.plot(steps, values, style, label=label)
        
        plt.title(f'Метрики оценки {model_name}')
        plt.xlabel('Шаг')
        plt.ylabel('Значение')
        plt.legend()
    
    plt.tight_layout()
    
    # Сохраняем график
    save_path = os.path.join(output_dir, f'{model_name}_training_metrics.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    
    return save_path

def plot_feature_importance(feature_names, importance_scores, title, save_path, figure_size=(12, 8), dpi=100, top_n=20):
    """
    Построение графика важности признаков
    
    Args:
        feature_names (list): Список названий признаков
        importance_scores (array-like): Оценки важности признаков
        title (str): Заголовок графика
        save_path (str): Путь для сохранения графика
        figure_size (tuple): Размер фигуры
        dpi (int): Разрешение изображения
        top_n (int): Количество наиболее важных признаков для отображения
    """
    # Создаем DataFrame с именами признаков и их важностью
    import pandas as pd
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    
    # Сортируем по важности и берем top_n признаков
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=figure_size)
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(title)
    plt.xlabel('Важность')
    plt.ylabel('Признак')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
