import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import os

# Ensure Russian stopwords are downloaded
nltk.download('stopwords')

# Load Russian stopwords
russian_stopwords = set(stopwords.words('russian'))

# Load dataset
csv_path = os.path.join('resources', 'dataset', 'toxic.csv')
df = pd.read_csv(csv_path)

# Определим классы (столбцы, кроме id и текста)
class_columns = [col for col in df.columns if col not in ['id', 'comment']]

# Для каждого класса строим wordcloud
for class_col in class_columns:
    # Отбираем тексты, где класс == 1
    texts = df.loc[df[class_col] == 0, 'comment'].dropna().astype(str)
    all_text = ' '.join(texts)
    # Создаем облако слов
    wc = WordCloud(width=800, height=400, background_color='white',
                   stopwords=russian_stopwords, collocations=False).generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('WordCloud for class: neutral')
    plt.show()