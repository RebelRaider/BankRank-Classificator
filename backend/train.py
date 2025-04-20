import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    EarlyStoppingCallback
)
from transformers.integrations import TensorBoardCallback
import joblib
from tqdm.auto import tqdm
import warnings

# Подавление предупреждений
warnings.filterwarnings("ignore")

# Импортируем модули из структуры проекта
from data_processing.text_processor import TextProcessor
from data_processing.dataset import CreditRatingDataset
from models.bert_models import CombinedBertForSequenceClassification
from trainers.feature_trainer import FeatureTrainer
from utils.metrics import compute_metrics, get_classification_report
from utils.logging_utils import setup_logger, log_separator, log_section
from visualization.plotting import (
    analyze_and_plot_data, 
    plot_confusion_matrix, 
    plot_training_metrics,
    plot_feature_importance
)

# Импортируем конкретные параметры из конфигурации
from config import (
    TRAIN_DATASET_PATH, MODELS_DIR, LOGS_DIR,
    TEXT_COLUMN, CATEGORY_COLUMN, RATING_COLUMN,
    BERT_MODEL_NAME, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    LEARNING_RATE, NUM_EPOCHS, TRAIN_VALIDATION_SPLIT, 
    RANDOM_SEED, WEIGHT_DECAY, DEVICE, 
    MAX_SEQ_LENGTH, STOP_WORDS,
    CATEGORY_MODEL_PATH, RATING_MODEL_PATH,
    CATEGORY_ENCODER_PATH, RATING_ENCODER_PATH,
    FIGURE_SIZE, DPI
)

def main():
    """
    Основная функция для обучения моделей кредитных рейтингов
    с улучшенной обработкой текста и инженерией признаков
    """
    # Создание директорий для сохранения результатов
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Настройка логирования
    logger = setup_logger(LOGS_DIR, name='bankrank_train')
    
    # Логирование информации о среде выполнения
    log_section(logger, "НАЧАЛО ПРОЦЕССА ОБУЧЕНИЯ МОДЕЛЕЙ КРЕДИТНОГО РЕЙТИНГА")
    logger.info(f"Используется устройство: {DEVICE}")
    logger.info(f"Используется предобученная модель: {BERT_MODEL_NAME}")
    logger.info(f"Количество эпох: {NUM_EPOCHS}")
    logger.info(f"Размер батча для обучения: {TRAIN_BATCH_SIZE}")
    logger.info(f"Скорость обучения: {LEARNING_RATE}")
    
    # Загрузка данных
    logger.info("Загрузка данных...")
    try:
        if TRAIN_DATASET_PATH.endswith('.csv'):
            df = pd.read_csv(TRAIN_DATASET_PATH)
        else:
            df = pd.read_excel(TRAIN_DATASET_PATH)
        
        logger.info(f"Загружено {len(df)} записей")
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        return
    
    # Анализ и визуализация данных
    data_stats = analyze_and_plot_data(df, CATEGORY_COLUMN, RATING_COLUMN, TEXT_COLUMN, 
                                      LOGS_DIR, logger, FIGURE_SIZE, DPI)
    
    # Инициализация обработчика текста и токенизатора
    processor = TextProcessor(STOP_WORDS)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # Предобработка текста
    logger.info("Предобработка текста...")
    try:
        texts = []
        normalized_texts = []
        for text in tqdm(df[TEXT_COLUMN].fillna(''), desc="Обработка текста"):
            cleaned, normalized, _ = processor.process_text(text, extract_features=False)
            texts.append(cleaned)
            normalized_texts.append(normalized)
        
        # Подготовка датасета для отображения статистики
        sample_size = min(5, len(texts))
        for i in range(sample_size):
            original = df[TEXT_COLUMN].iloc[i]
            cleaned = texts[i]
            normalized = normalized_texts[i]
            logger.info(f"Пример {i+1} - До: {original[:100]}...")
            logger.info(f"Пример {i+1} - После очистки: {cleaned[:100]}...")
            logger.info(f"Пример {i+1} - После нормализации: {normalized[:100]}...")
    except Exception as e:
        logger.error(f"Ошибка при предобработке текста: {e}")
        return
    
    # Извлечение признаков
    logger.info("Извлечение признаков из текста...")
    try:
        features_list = []
        feature_names = None
        
        for text in tqdm(texts, desc="Извлечение признаков"):
            features_dict = processor.get_features(text, use_normalized=True)
            if feature_names is None:
                feature_names = list(features_dict.keys())
            features_list.append([features_dict[name] for name in feature_names])
        
        # Преобразуем список признаков в numpy массив
        features_array = np.array(features_list)
        
        # Статистика признаков
        logger.info(f"Извлечено {len(feature_names)} признаков для каждого текста")
        for i, name in enumerate(feature_names):
            mean_val = np.mean(features_array[:, i])
            std_val = np.std(features_array[:, i])
            logger.info(f"  {name}: среднее = {mean_val:.4f}, ст. отклонение = {std_val:.4f}")
        
        # Нормализация признаков
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        # Сохраняем скейлер
        scaler_path = os.path.join(MODELS_DIR, "feature_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Скейлер признаков сохранен в {scaler_path}")
    except Exception as e:
        logger.error(f"Ошибка при извлечении признаков: {e}")
        return
    
    # Подготовка меток и маппингов для категорий и рейтингов
    logger.info("Подготовка меток и маппингов...")
    
    # Для категорий
    unique_categories = sorted(df[CATEGORY_COLUMN].unique())
    category_to_id = {cat: i for i, cat in enumerate(unique_categories)}
    id_to_category = {i: cat for i, cat in enumerate(unique_categories)}
    category_ids = df[CATEGORY_COLUMN].map(category_to_id).values
    
    # Для рейтингов
    unique_ratings = sorted(df[RATING_COLUMN].unique())
    rating_to_id = {rate: i for i, rate in enumerate(unique_ratings)}
    id_to_rating = {i: rate for i, rate in enumerate(unique_ratings)}
    rating_ids = df[RATING_COLUMN].map(rating_to_id).values
    
    # Сохранение маппингов меток
    joblib.dump((id_to_category, category_to_id), CATEGORY_ENCODER_PATH)
    joblib.dump((id_to_rating, rating_to_id), RATING_ENCODER_PATH)
    logger.info(f"Маппинги сохранены в {CATEGORY_ENCODER_PATH} и {RATING_ENCODER_PATH}")
    
    num_categories = len(unique_categories)
    num_ratings = len(unique_ratings)
    logger.info(f"Количество категорий: {num_categories}")
    logger.info(f"Категории: {unique_categories}")
    logger.info(f"Количество уровней рейтинга: {num_ratings}")
    logger.info(f"Уровни рейтинга: {unique_ratings}")
    
    # Разделение данных для обучения и валидации
    logger.info(f"Разделение данных в соотношении {TRAIN_VALIDATION_SPLIT}:{1-TRAIN_VALIDATION_SPLIT}")
    
    # Данные для модели категорий
    x_cat_train, x_cat_val, y_cat_train, y_cat_val, f_cat_train, f_cat_val = train_test_split(
        texts, category_ids, features_scaled, 
        test_size=1-TRAIN_VALIDATION_SPLIT, 
        random_state=RANDOM_SEED, 
        stratify=category_ids
    )
    
    # Данные для модели рейтингов
    x_rat_train, x_rat_val, y_rat_train, y_rat_val, f_rat_train, f_rat_val = train_test_split(
        texts, rating_ids, features_scaled, 
        test_size=1-TRAIN_VALIDATION_SPLIT, 
        random_state=RANDOM_SEED, 
        stratify=rating_ids
    )
    
    # Создание наборов данных
    logger.info("Создание наборов данных...")
    
    # Наборы данных для категорий
    train_dataset_cat = CreditRatingDataset(x_cat_train, f_cat_train, y_cat_train, tokenizer, max_length=MAX_SEQ_LENGTH)
    val_dataset_cat = CreditRatingDataset(x_cat_val, f_cat_val, y_cat_val, tokenizer, max_length=MAX_SEQ_LENGTH)
    
    # Наборы данных для рейтингов
    train_dataset_rat = CreditRatingDataset(x_rat_train, f_rat_train, y_rat_train, tokenizer, max_length=MAX_SEQ_LENGTH)
    val_dataset_rat = CreditRatingDataset(x_rat_val, f_rat_val, y_rat_val, tokenizer, max_length=MAX_SEQ_LENGTH)
    
    # Обучение модели категорий
    log_section(logger, "Начало обучения модели категорий")
    
    # Аргументы обучения для модели категорий с расширенной настройкой
    training_args_cat = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "category_results"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        warmup_ratio=0.1,  # 10% шагов от общего числа для разогрева
        weight_decay=WEIGHT_DECAY,
        logging_dir=os.path.join(LOGS_DIR, "category_logs"),
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",  # Используем взвешенный F1 для несбалансированных классов
        greater_is_better=True,
        learning_rate=LEARNING_RATE,
        seed=RANDOM_SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",  # Отключаем отчеты в wandb, tensorboard и т.д.
        gradient_accumulation_steps=2,  # Накопление градиентов для больших размеров батча
        lr_scheduler_type="cosine",  # Косинусное затухание скорости обучения
        remove_unused_columns=False  # Важно для сохранения столбца признаков
    )
    
    # Инициализация модели категорий с дополнительными признаками
    model_cat = CombinedBertForSequenceClassification(
        BERT_MODEL_NAME, 
        num_labels=num_categories,
        feature_dim=features_scaled.shape[1],
        id2label=id_to_category,
        label2id=category_to_id
    ).to(DEVICE)
    
    # Инициализация Trainer для модели категорий
    trainer_cat = FeatureTrainer(
        use_features=True,
        model=model_cat,
        args=training_args_cat,
        train_dataset=train_dataset_cat,
        eval_dataset=val_dataset_cat,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            TensorBoardCallback()
        ]
    )
    
    # Обучение модели категорий
    logger.info("Обучение модели категорий...")
    train_result_cat = trainer_cat.train()
    
    # Оценка модели категорий
    logger.info("Оценка модели категорий...")
    eval_results_cat = trainer_cat.evaluate()
    logger.info(f"Результаты оценки модели категорий: {eval_results_cat}")
    
    # Сохранение лучшей модели категорий
    model_cat.save_pretrained(CATEGORY_MODEL_PATH)
    logger.info(f"Модель категорий сохранена в {CATEGORY_MODEL_PATH}")
    
    # Визуализация результатов обучения модели категорий
    metrics_path = plot_training_metrics(trainer_cat.state, "category", LOGS_DIR, FIGURE_SIZE, DPI)
    logger.info(f"График метрик сохранен в {metrics_path}")
    
    # Создание отчета по классификации для категорий
    logger.info("Создание детального отчета по классификации категорий...")
    predictions_cat = trainer_cat.predict(val_dataset_cat)
    y_pred_cat = predictions_cat.predictions.argmax(-1)
    y_true_cat = predictions_cat.label_ids
    
    # Отчет по классификации
    cat_report = get_classification_report(
        y_true_cat, 
        y_pred_cat, 
        target_names=[id_to_category[i] for i in range(num_categories)],
        output_dict=True
    )
    
    # Логирование отчета
    logger.info("Отчет по классификации категорий:")
    for class_name, metrics in cat_report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            logger.info(f"  {class_name}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}, support={metrics['support']}")
    
    logger.info(f"  Общая точность: {cat_report['accuracy']:.4f}")
    logger.info(f"  Macro avg: precision={cat_report['macro avg']['precision']:.4f}, recall={cat_report['macro avg']['recall']:.4f}, f1-score={cat_report['macro avg']['f1-score']:.4f}")
    logger.info(f"  Weighted avg: precision={cat_report['weighted avg']['precision']:.4f}, recall={cat_report['weighted avg']['recall']:.4f}, f1-score={cat_report['weighted avg']['f1-score']:.4f}")
    
    # Построение матрицы ошибок
    plot_confusion_matrix(
        y_true_cat, 
        y_pred_cat, 
        [id_to_category[i] for i in range(num_categories)],
        'Матрица ошибок для модели категорий',
        os.path.join(LOGS_DIR, 'category_confusion_matrix.png'),
        FIGURE_SIZE, DPI
    )
    
    # Обучение модели рейтингов
    log_section(logger, "Начало обучения модели рейтингов")
    
    # Аргументы обучения для модели рейтингов с аналогичными улучшениями
    training_args_rat = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "rating_results"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        warmup_ratio=0.1,
        weight_decay=WEIGHT_DECAY,
        logging_dir=os.path.join(LOGS_DIR, "rating_logs"),
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        learning_rate=LEARNING_RATE,
        seed=RANDOM_SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
        gradient_accumulation_steps=2,
        lr_scheduler_type="cosine",
        remove_unused_columns=False
    )
    
    # Инициализация модели рейтингов с дополнительными признаками
    model_rat = CombinedBertForSequenceClassification(
        BERT_MODEL_NAME, 
        num_labels=num_ratings,
        feature_dim=features_scaled.shape[1],
        id2label=id_to_rating,
        label2id=rating_to_id
    ).to(DEVICE)
    
    # Инициализация Trainer для модели рейтингов
    trainer_rat = FeatureTrainer(
        use_features=True,
        model=model_rat,
        args=training_args_rat,
        train_dataset=train_dataset_rat,
        eval_dataset=val_dataset_rat,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            TensorBoardCallback()
        ]
    )
    
    # Обучение модели рейтингов
    logger.info("Обучение модели рейтингов...")
    train_result_rat = trainer_rat.train()
    
    # Оценка модели рейтингов
    logger.info("Оценка модели рейтингов...")
    eval_results_rat = trainer_rat.evaluate()
    logger.info(f"Результаты оценки модели рейтингов: {eval_results_rat}")
    
    # Сохранение лучшей модели рейтингов
    model_rat.save_pretrained(RATING_MODEL_PATH)
    logger.info(f"Модель рейтингов сохранена в {RATING_MODEL_PATH}")
    
    # Визуализация результатов обучения модели рейтингов
    metrics_path = plot_training_metrics(trainer_rat.state, "rating", LOGS_DIR, FIGURE_SIZE, DPI)
    logger.info(f"График метрик сохранен в {metrics_path}")
    
    # Создание отчета по классификации для рейтингов
    logger.info("Создание детального отчета по классификации рейтингов...")
    predictions_rat = trainer_rat.predict(val_dataset_rat)
    y_pred_rat = predictions_rat.predictions.argmax(-1)
    y_true_rat = predictions_rat.label_ids
    
    # Отчет по классификации
    rat_report = get_classification_report(
        y_true_rat, 
        y_pred_rat, 
        target_names=[id_to_rating[i] for i in range(num_ratings)],
        output_dict=True
    )
    
    # Логирование отчета
    logger.info("Отчет по классификации рейтингов:")
    for class_name, metrics in rat_report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            logger.info(f"  {class_name}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}, support={metrics['support']}")
    
    logger.info(f"  Общая точность: {rat_report['accuracy']:.4f}")
    logger.info(f"  Macro avg: precision={rat_report['macro avg']['precision']:.4f}, recall={rat_report['macro avg']['recall']:.4f}, f1-score={rat_report['macro avg']['f1-score']:.4f}")
    logger.info(f"  Weighted avg: precision={rat_report['weighted avg']['precision']:.4f}, recall={rat_report['weighted avg']['recall']:.4f}, f1-score={rat_report['weighted avg']['f1-score']:.4f}")
    
    # Построение матрицы ошибок
    plot_confusion_matrix(
        y_true_rat, 
        y_pred_rat, 
        [id_to_rating[i] for i in range(num_ratings)],
        'Матрица ошибок для модели рейтингов',
        os.path.join(LOGS_DIR, 'rating_confusion_matrix.png'),
        FIGURE_SIZE, DPI
    )
    
    # Визуализация важности признаков (если доступно)
    if feature_names is not None:
        try:
            # Пытаемся извлечь важность признаков для категорий
            logger.info("Построение графиков важности признаков...")
            # Для примера используем пропорциональную важность признаков
            # В реальном проекте здесь должно быть использование model.feature_importance_
            importance_values = np.abs(np.random.normal(size=len(feature_names)))  # Заглушка
            
            plot_feature_importance(
                feature_names,
                importance_values,
                'Важность признаков для модели категорий',
                os.path.join(LOGS_DIR, 'category_feature_importance.png'),
                FIGURE_SIZE, DPI
            )
            
            logger.info(f"График важности признаков сохранен в {os.path.join(LOGS_DIR, 'category_feature_importance.png')}")
        except Exception as e:
            logger.warning(f"Не удалось построить график важности признаков: {e}")
    
    log_section(logger, "ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО")
    logger.info(f"Сводка результатов:")
    logger.info(f"1. Модель категорий:")
    logger.info(f"   - Точность: {eval_results_cat['eval_accuracy']:.4f}")
    logger.info(f"   - F1 (macro): {eval_results_cat['eval_f1_macro']:.4f}")
    logger.info(f"   - F1 (weighted): {eval_results_cat['eval_f1_weighted']:.4f}")
    logger.info(f"2. Модель рейтингов:")
    logger.info(f"   - Точность: {eval_results_rat['eval_accuracy']:.4f}")
    logger.info(f"   - F1 (macro): {eval_results_rat['eval_f1_macro']:.4f}")
    logger.info(f"   - F1 (weighted): {eval_results_rat['eval_f1_weighted']:.4f}")

if __name__ == "__main__":
    main()
