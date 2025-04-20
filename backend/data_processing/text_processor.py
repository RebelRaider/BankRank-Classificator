import re
import pandas as pd
from bs4 import BeautifulSoup
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
from pymorphy3 import MorphAnalyzer
import warnings

# Подавление предупреждений
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Класс для обработки текста с расширенными возможностями очистки и извлечения признаков.
    Включает нормализацию текста с помощью pymorphy3.
    """
    
    def __init__(self, stop_words, nlp_model=None):
        """
        Инициализация процессора текста
        
        Args:
            stop_words (list): Список стоп-слов
            nlp_model: Загруженная модель spaCy (если None, будет загружена автоматически)
        """
        self.stop_words = set(stop_words)
        
        # Инициализация spaCy
        if nlp_model:
            self.nlp = nlp_model
        else:
            try:
                self.nlp = spacy.load("ru_core_news_lg")
                logger.info("Загружена языковая модель spaCy: ru_core_news_lg")
            except OSError:
                logger.info("Загрузка языковой модели для spaCy...")
                spacy.cli.download("ru_core_news_lg")
                self.nlp = spacy.load("ru_core_news_lg")
        
        # Инициализация морфологического анализатора
        self.morph = MorphAnalyzer()
        
        # Инициализация анализатора тональности
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            logger.info("Загрузка ресурсов NLTK для анализа тональности...")
            nltk.download('vader_lexicon')
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Финансовые термины для поиска в тексте
        self.financial_terms = {
            'risk': ['риск', 'рисков', 'рисками', 'рисковый', 'рискованный'],
            'profit': ['прибыль', 'доход', 'выручка', 'доходность', 'прибыльный', 'рентабельность'],
            'loss': ['убыток', 'потери', 'отрицательный результат', 'убыточный'],
            'debt': ['долг', 'задолженность', 'обязательства', 'кредит', 'займ', 'заемный'],
            'growth': ['рост', 'увеличение', 'развитие', 'повышение', 'возрастание'],
            'decline': ['спад', 'снижение', 'падение', 'ухудшение', 'сокращение'],
            'investment': ['инвестиции', 'вложения', 'капиталовложения', 'инвестирование'],
            'assets': ['активы', 'имущество', 'собственность', 'основные средства'],
            'liabilities': ['обязательства', 'пассивы', 'задолженности', 'долговые обязательства'],
            'equity': ['капитал', 'собственный капитал', 'акционерный капитал', 'уставный капитал'],
            'liquidity': ['ликвидность', 'платежеспособность', 'оборотные средства'],
            'solvency': ['платежеспособность', 'кредитоспособность'],
            'rating': ['рейтинг', 'оценка', 'кредитный рейтинг', 'кредитоспособность'],
            'stability': ['стабильность', 'устойчивость', 'финансовая устойчивость']
        }
        
        # Регулярные выражения для поиска финансовых показателей
        self.money_pattern = re.compile(r'(?:\d+[.,]?\d*\s*(?:млн|млрд|трлн|тыс)?\.?\s*(?:руб|₽|долл|€|\$|евро))|(?:(?:руб|₽|долл|€|\$|евро)\.?\s*\d+[.,]?\d*\s*(?:млн|млрд|трлн|тыс)?\.?)', re.IGNORECASE)
        self.percentage_pattern = re.compile(r'\d+[.,]?\d*\s*(?:%|процент(?:а|ов)?)\b|\b(?:увеличение|снижение|рост|спад|повышение|понижение)\s+на\s+\d+[.,]?\d*\s*(?:%|процент(?:а|ов)?)\b', re.IGNORECASE)
        self.ratio_pattern = re.compile(r'\b(?:коэффициент|показатель|индекс|отношение)\s+[а-яА-Я\s]+\s*[:|]\s*\d+[.,]?\d*', re.IGNORECASE)
        
        # Словарь для нормализации аббревиатур и сокращений
        self.abbreviations = {
            "АО": "акционерное общество",
            "ПАО": "публичное акционерное общество",
            "ООО": "общество с ограниченной ответственностью",
            "НПФ": "негосударственный пенсионный фонд",
            "ЦБ": "центральный банк",
            "ЦБ РФ": "центральный банк Российской Федерации",
            "РФ": "Российская Федерация",
            "млн": "миллион",
            "млрд": "миллиард",
            "трлн": "триллион",
            "тыс": "тысяча",
            "руб": "рублей",
            "долл": "долларов",
            "евро": "евро"
        }

    def clear_text(self, text):
        """
        Улучшенная очистка текста от HTML, ссылок и ненужных символов
        с сохранением важной финансовой информации
        
        Args:
            text (str): Исходный текст
            
        Returns:
            str: Очищенный текст
        """
        # Обработка None или пустых строк
        if not text or pd.isna(text):
            return ""
            
        # Удаление HTML-тегов с сохранением структуры документа
        soup = BeautifulSoup(str(text), features="html.parser")
        for script in soup(["script", "style"]):
            script.extract()  # удаляем скрипты и стили
        
        # Заменяем некоторые теги на их текстовые эквиваленты для сохранения структуры
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            tag.append(' ')
        for tag in soup.find_all('br'):
            tag.replace_with(' ')
        for tag in soup.find_all('li'):
            tag.insert_before('• ')
        
        text = soup.get_text(separator=' ')
        
        # Удаление URL-адресов
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Нормализация кавычек и других парных знаков
        text = re.sub(r'[«»""„"]', '"', text)
        text = re.sub(r'['']', "'", text)
        
        # Улучшенная обработка финансовых показателей (сохраняем суммы, проценты и т.д.)
        # Временно заменяем их маркерами, чтобы не повредить при очистке
        money_markers = {}
        percentages_markers = {}
        ratio_markers = {}
        
        # Сохраняем финансовые показатели
        for i, match in enumerate(self.money_pattern.finditer(text)):
            marker = f"MONEY_MARKER_{i}"
            money_markers[marker] = match.group(0)
            text = text[:match.start()] + marker + text[match.end():]
        
        for i, match in enumerate(self.percentage_pattern.finditer(text)):
            marker = f"PERCENTAGE_MARKER_{i}"
            percentages_markers[marker] = match.group(0)
            text = text[:match.start()] + marker + text[match.end():]
        
        for i, match in enumerate(self.ratio_pattern.finditer(text)):
            marker = f"RATIO_MARKER_{i}"
            ratio_markers[marker] = match.group(0)
            text = text[:match.start()] + marker + text[match.end():]
        
        # Удаление специальных символов с сохранением важных знаков препинания
        text = re.sub(r'[^\w\d\s.,;:!?%()"/\'«»—–-]', ' ', text)
        
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text)
        
        # Нормализация знаков пунктуации (удаление лишних пробелов)
        for punct in '.,;:!?)':
            text = re.sub(f' ([{punct}])', r'\1', text)
        for punct in '(«':
            text = re.sub(f'([{punct}]) ', r'\1', text)
        
        # Возвращаем финансовые показатели обратно
        for marker, value in money_markers.items():
            text = text.replace(marker, value)
        for marker, value in percentages_markers.items():
            text = text.replace(marker, value)
        for marker, value in ratio_markers.items():
            text = text.replace(marker, value)
            
        # Нормализация аббревиатур
        for abbr, full in self.abbreviations.items():
            # Заменяем только полные слова-аббревиатуры, а не части слов
            text = re.sub(rf'\b{abbr}\b', f"{abbr} ({full})", text, count=1)
        
        return text.strip()

    def normalize_text(self, text):
        """
        Нормализация текста с использованием pymorphy3
        для приведения слов к нормальной форме
        
        Args:
            text (str): Исходный очищенный текст
            
        Returns:
            str: Нормализованный текст
        """
        if not text:
            return ""
        
        # Токенизация текста
        words = text.split()
        
        # Нормализация слов
        normalized_words = []
        for word in words:
            # Пропускаем специальные символы и числа
            if re.match(r'^[.,;:!?%()"/\'«»—–-]$', word) or re.match(r'^\d+$', word):
                normalized_words.append(word)
                continue
                
            # Используем pymorphy3 для получения нормальной формы
            parsed = self.morph.parse(word)[0]
            normalized_words.append(parsed.normal_form)
        
        # Объединяем слова обратно в текст
        normalized_text = ' '.join(normalized_words)
        
        return normalized_text
        
    def get_features(self, text, use_normalized=True):
        """
        Расширенное извлечение признаков из текста с возможностью использования
        нормализованного текста для более точного анализа
        
        Args:
            text (str): Исходный очищенный текст
            use_normalized (bool): Использовать ли нормализованный текст для анализа
            
        Returns:
            dict: Словарь с извлеченными признаками
        """
        # Обработка пустых строк
        if not text or text.strip() == "":
            # Возвращаем нулевые значения для всех признаков
            return {
                # Базовая статистика
                'word_count': 0, 'avg_word_len': 0, 'unique_words_ratio': 0,
                'sentence_count': 0, 'avg_sentence_len': 0,
                # Именованные сущности
                'org_count': 0, 'loc_count': 0, 'per_count': 0, 'date_count': 0,
                # Финансовые показатели
                'money_mentions': 0, 'percentage_mentions': 0, 'ratio_mentions': 0,
                # Анализ тональности
                'sentiment_pos': 0, 'sentiment_neg': 0, 'sentiment_neu': 0,
                # Финансовые термины
                'risk_terms': 0, 'profit_terms': 0, 'loss_terms': 0, 
                'debt_terms': 0, 'growth_terms': 0, 'decline_terms': 0,
                'investment_terms': 0, 'asset_terms': 0, 'liability_terms': 0, 'equity_terms': 0,
                'liquidity_terms': 0, 'solvency_terms': 0, 'rating_terms': 0, 'stability_terms': 0,
                # Части речи
                'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0, 'adv_ratio': 0,
                # Формальность
                'formality_score': 0
            }
        
        # Если требуется использовать нормализованный текст, нормализуем его
        analysis_text = self.normalize_text(text) if use_normalized else text
            
        # Обработка текста с помощью spaCy для не-нормализованной версии
        # (для сохранения структуры предложений, частей речи и т.д.)
        doc = self.nlp(text)
        
        # БАЗОВАЯ СТАТИСТИКА ТЕКСТА
        # Считаем только значимые слова (исключаем стоп-слова и знаки препинания)
        words = [token.text.lower() for token in doc 
                if not token.is_stop and token.is_alpha and not token.is_punct]
        
        # Количество слов и уникальных слов
        word_count = len(words)
        unique_words = len(set(words)) if word_count > 0 else 0
        unique_words_ratio = (unique_words / word_count * 100) if word_count > 0 else 0
        
        # Средняя длина слова
        avg_word_len = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        # Количество и средняя длина предложений
        sentences = list(doc.sents)
        sentence_count = len(sentences)
        avg_sentence_len = sum(len(sent) for sent in sentences) / sentence_count if sentence_count > 0 else 0
        
        # ИМЕНОВАННЫЕ СУЩНОСТИ
        org_count = sum(1 for ent in doc.ents if ent.label_ == "ORG" and ent.text.lower() not in self.stop_words)
        loc_count = sum(1 for ent in doc.ents if ent.label_ == "LOC")
        per_count = sum(1 for ent in doc.ents if ent.label_ == "PER")
        date_count = sum(1 for ent in doc.ents if ent.label_ == "DATE")
        
        # ФИНАНСОВЫЕ ПОКАЗАТЕЛИ
        money_mentions = len(re.findall(self.money_pattern, text))
        percentage_mentions = len(re.findall(self.percentage_pattern, text))
        ratio_mentions = len(re.findall(self.ratio_pattern, text))
        
        # АНАЛИЗ ТОНАЛЬНОСТИ
        # Для русского текста используем лексический подход или адаптированную модель
        # Поскольку VADER на английском, это будет приблизительная оценка
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # ФИНАНСОВЫЕ ТЕРМИНЫ
        # Подсчитываем упоминания финансовых терминов по категориям
        # Используем нормализованный текст для более точного поиска
        text_lower = analysis_text.lower()
        financial_term_counts = {}
        for term_type, term_list in self.financial_terms.items():
            # Для нормализованного текста мы также должны нормализовать термины для поиска
            if use_normalized:
                normalized_terms = [self.morph.parse(term)[0].normal_form for term in term_list]
                count = sum(text_lower.count(term) for term in normalized_terms)
            else:
                count = sum(text_lower.count(term) for term in term_list)
                
            financial_term_counts[f'{term_type}_terms'] = count
        
        # ЧАСТИ РЕЧИ
        pos_counts = {}
        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        total_tokens = len(doc)
        noun_ratio = pos_counts.get('NOUN', 0) / total_tokens if total_tokens > 0 else 0
        verb_ratio = pos_counts.get('VERB', 0) / total_tokens if total_tokens > 0 else 0
        adj_ratio = pos_counts.get('ADJ', 0) / total_tokens if total_tokens > 0 else 0
        adv_ratio = pos_counts.get('ADV', 0) / total_tokens if total_tokens > 0 else 0
        
        # ФОРМАЛЬНОСТЬ ТЕКСТА
        # Простой показатель формальности: соотношение существительных к глаголам и наречиям
        formality_numerator = pos_counts.get('NOUN', 0) + pos_counts.get('ADJ', 0) + pos_counts.get('ADP', 0)
        formality_denominator = pos_counts.get('VERB', 0) + pos_counts.get('ADV', 0) + pos_counts.get('PRON', 0) + 1  # +1 чтобы избежать деления на 0
        formality_score = formality_numerator / formality_denominator
        
        # Собираем все признаки в один словарь
        features = {
            # Базовая статистика
            'word_count': word_count,
            'avg_word_len': avg_word_len,
            'unique_words_ratio': unique_words_ratio,
            'sentence_count': sentence_count,
            'avg_sentence_len': avg_sentence_len,
            # Именованные сущности
            'org_count': org_count,
            'loc_count': loc_count,
            'per_count': per_count,
            'date_count': date_count,
            # Финансовые показатели
            'money_mentions': money_mentions,
            'percentage_mentions': percentage_mentions,
            'ratio_mentions': ratio_mentions,
            # Анализ тональности
            'sentiment_pos': sentiment_scores['pos'],
            'sentiment_neg': sentiment_scores['neg'],
            'sentiment_neu': sentiment_scores['neu'],
            # Части речи
            'noun_ratio': noun_ratio,
            'verb_ratio': verb_ratio,
            'adj_ratio': adj_ratio,
            'adv_ratio': adv_ratio,
            # Формальность
            'formality_score': formality_score
        }
        
        # Добавляем финансовые термины
        features.update(financial_term_counts)
        
        return features

    def process_text(self, text, extract_features=True, normalize=True):
        """
        Полная обработка текста: очистка, нормализация и извлечение признаков
        
        Args:
            text (str): Исходный текст
            extract_features (bool): Извлекать ли признаки
            normalize (bool): Нормализовать ли текст
            
        Returns:
            tuple: (очищенный текст, нормализованный текст, признаки)
        """
        cleaned_text = self.clear_text(text)
        
        normalized_text = None
        features = None
        
        if normalize:
            normalized_text = self.normalize_text(cleaned_text)
            
        if extract_features:
            features = self.get_features(cleaned_text, use_normalized=normalize)
            
        return cleaned_text, normalized_text, features
