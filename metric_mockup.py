import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Эпохи (от 0 до 4, шаг 0.2)
epochs = np.arange(0, 4.01, 0.2)

# Моки для loss и метрик
loss = np.exp(-epochs) * 0.7 + 0.2 + np.random.normal(0, 0.01, len(epochs))

# Метрики: стартуют с ~0.4, выходят к финальным значениям
def sigmoid_curve(x, start, end, k=2.5, x0=2):
    """Сигмоида для плавного роста метрик."""
    return start + (end - start) / (1 + np.exp(-k * (x - x0) / x0))

accuracy = sigmoid_curve(epochs, 0.42, 0.91) + np.random.normal(0, 0.005, len(epochs))
precision = sigmoid_curve(epochs, 0.40, 0.90) + np.random.normal(0, 0.005, len(epochs))
recall = sigmoid_curve(epochs, 0.38, 0.88) + np.random.normal(0, 0.005, len(epochs))
f1 = sigmoid_curve(epochs, 0.39, 0.89) + np.random.normal(0, 0.005, len(epochs))

# График: loss и метрики на одном графике
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label='Loss', color='black', linewidth=2)
plt.plot(epochs, accuracy, label='Accuracy')
plt.plot(epochs, precision, label='Precision')
plt.plot(epochs, recall, label='Recall')
plt.plot(epochs, f1, label='F1-score')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Loss and Metrics')
plt.ylim(0.0, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Тепловая карта confusion matrix
true_classes = ['neutral', 'toxic']
pred_classes = ['neutral', 'toxic']
conf_matrix = np.array([
    [480,  20],
    [ 18, 390]
])

plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=pred_classes, yticklabels=true_classes)
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
