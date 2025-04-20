import os
import logging
import datetime

def setup_logger(logs_dir, name='bankrank', level=logging.INFO):
    """
    Настройка логирования
    
    Args:
        logs_dir (str): Директория для сохранения логов
        name (str): Имя логгера
        level: Уровень логирования
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    # Создаем директорию для логов, если она не существует
    os.makedirs(logs_dir, exist_ok=True)
    
    # Генерируем имя файла лога с датой и временем
    log_file = os.path.join(logs_dir, f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Настраиваем логгер
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Создаем обработчики для файла и консоли
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Устанавливаем форматирование
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Добавляем обработчики к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_separator(logger, length=80, char='='):
    """
    Печать разделителя в лог
    
    Args:
        logger: Логгер
        length (int): Длина разделителя
        char (str): Символ разделителя
    """
    logger.info(char * length)

def log_section(logger, title, length=80, char='='):
    """
    Печать заголовка секции в лог
    
    Args:
        logger: Логгер
        title (str): Заголовок секции
        length (int): Длина разделителя
        char (str): Символ разделителя
    """
    log_separator(logger, length, char)
    logger.info(title)
    log_separator(logger, length, char)
