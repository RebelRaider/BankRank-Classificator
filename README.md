# 🏦 Bank Classification Service

Веб-сервис автоматической классификации текстовых обращений банков с анализом токсичности и прогнозированием рейтингов.

## 🚀 Возможности

- **Анализ токсичности** - определение токсичных высказываний в банковских документах
- **Классификация рейтингов** - прогнозирование рейтингов банков на основе пресс-релизов
- **Поддержка форматов** - TXT, PDF, DOC, DOCX файлы до 10MB
- **Веб-интерфейс** - современный React-интерфейс для загрузки и анализа
- **REST API** - программный доступ к функциональности
- **Аналитика** - детальная статистика и визуализация результатов
- **Кэширование** - Redis для быстрого повторного анализа

## 🏗️ Архитектура

``` structure
├── Backend (FastAPI)
│   ├── ML модели (BERT-based)
│   ├── API endpoints
│   └── Обработка данных
├── Frontend (React)
│   ├── Загрузка файлов
│   ├── Отображение результатов
│   └── Аналитика
├── База данных (PostgreSQL)
└── Кэш (Redis)
```

## 🔧 Установка и запуск

### Предварительные требования

- Docker и Docker Compose
- Python 3.9+ (для локальной разработки)
- Node.js 18+ (для локальной разработки)

### Быстрый старт с Docker

1. Клонируйте репозиторий:
``` sh
git clone 
cd bank-classification-service
```

2. Создайте файл `.env` на основе `.env.example`:
``` sh
cp .env.example .env
# Отредактируйте .env файл с вашими настройками
```

3. Запустите сервисы:
``` sh
docker-compose up -d
```

4. Проверьте работоспособность:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API документация: http://localhost:8000/docs

### Обучение модели

1. Подготовьте CSV файл с данными (формат: текст, Категория)

2. Запустите обучение:
```
python scripts/train_model.py --csv_path path/to/train.csv --epochs 5
```

3. Обученная модель будет сохранена в `./trained_rating_model/`

## 📊 Используемые модели

- **Токсичность**: `s-nlp/russian_toxicity_classifier`
- **Рейтинги**: `DeepPavlov/rubert-base-cased` (fine-tuned)

## 🔗 API Endpoints

### Классификация
- `POST /api/v1/classification/upload` - загрузка файла
- `POST /api/v1/classification/classify/{document_id}` - анализ документа
- `POST /api/v1/classification/classify-text` - анализ текста
- `GET /api/v1/classification/history` - история анализов

### Аналитика
- `GET /api/v1/analytics/dashboard` - данные дашборда
- `GET /api/v1/analytics/statistics` - детальная статистика

## 🛠️ Разработка

### Backend
```
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```
cd frontend
npm install
npm start
```

### База данных
```
# PostgreSQL локально
createdb bank_classification
# или используйте Docker
docker run -d -p 5432:5432 -e POSTGRES_DB=bank_classification postgres:15
```

## 📈 Метрики и мониторинг

Система автоматически собирает метрики:
- Время обработки документов
- Точность классификации
- Статистика использования
- Распределение токсичности и рейтингов

## 🔒 Безопасность

- JWT токены для аутентификации
- CORS настройки
- Валидация входных данных
- Ограничения размера файлов

## 📝 Лицензия

MIT License

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch
3. Commit изменения
4. Push в branch
5. Создайте Pull Request

## 📞 Поддержка

Для вопросов и предложений создавайте Issues в репозитории.
```