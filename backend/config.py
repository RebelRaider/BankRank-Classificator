import os
import torch
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Resource directories
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
DATASET_DIR = os.path.join(RESOURCES_DIR, "dataset")
MODELS_DIR = os.path.join(RESOURCES_DIR, "models")
LOGS_DIR = os.path.join(RESOURCES_DIR, "logs")

# Create directories if they don't exist
for directory in [RESOURCES_DIR, DATASET_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset paths
TRAIN_DATASET_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_DATASET_PATH = os.path.join(DATASET_DIR, "test.csv")

# Model paths
CATEGORY_MODEL_PATH = os.path.join(MODELS_DIR, "model_cat.pt")
RATING_MODEL_PATH = os.path.join(MODELS_DIR, "model_rat.pt")
CATEGORY_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder_cat.joblib")
RATING_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder_rat.joblib")

# Text processing
TEXT_COLUMN = "pr_txt"  # Column name containing text in datasets
CATEGORY_COLUMN = "Категория"  # Column name for category labels
RATING_COLUMN = "Уровень рейтинга"  # Column name for rating labels
MAX_SEQ_LENGTH = 512  # Maximum sequence length for tokenization

# Training parameters
BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"  # Pretrained model for Russian
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 1e-5
LEARNING_RATE_WARMUP_STEPS = 0
NUM_EPOCHS = 10
DROPOUT_RATE = 0.25
TRAIN_VALIDATION_SPLIT = 0.8
RANDOM_SEED = 42
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_VALUE = 1.0

# Hardware settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # Number of workers for data loading

# Category labels (from notebooks)
CATEGORY_LABELS = ['A', 'AA', 'AAA', 'B', 'BB', 'BBB', 'C']

# Rating labels (from notebooks)
RATING_LABELS = ['A+', 'A', 'A-', 'AA+', 'AA', 'AA-', 'AAA', 'B+', 'B', 'B-', 
                 'BB+', 'BB', 'BB-', 'BBB+', 'BBB', 'BBB-', 'C']

# NLP processing settings
STOP_WORDS = ['АО «Эксперт РА', 'АКРА', 'Компания', 'Группа', 'Эксперт РА', 
              'Рейтинговое агентство', 'АО Эксперт РА', 'Кредитные', 
              'Оценка внешнее влияние', 'Группа.']

# Inference settings
INFERENCE_BATCH_SIZE = 16

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 100
