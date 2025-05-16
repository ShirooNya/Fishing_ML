import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Конфигурация
MAX_URL_LEN = 100
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.01
DATA_PATH = "urls.csv"
MODEL_PATH = "phishing_model.npy"


# Активационные функции
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Загрузка данных
def load_data():
    try:
        # Чтение CSV с явным указанием заголовка и обработкой mixed types
        data = pd.read_csv(DATA_PATH,
                           sep=',',
                           header=0,  # Используем первую строку как заголовок
                           names=['url', 'is_phishing'],  # Явные имена столбцов
                           dtype={'url': str, 'is_phishing': int},  # Явные типы данных
                           on_bad_lines='warn',  # Пропускать проблемные строки с предупреждением
                           low_memory=False)

        # Удаление строк с пропущенными значениями
        data = data.dropna()

        # Преобразование меток в числовой формат (на случай, если некоторые значения строковые)
        data['is_phishing'] = pd.to_numeric(data['is_phishing'], errors='coerce').fillna(0).astype(int)

        # Проверка распределения классов
        class_counts = data['is_phishing'].value_counts()
        print(f"Распределение классов:\n{class_counts}")

        return data["url"].values, data["is_phishing"].values

    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        # Возвращаем тестовые данные, если файл не найден или есть другие ошибки
        test_urls = ["https://google.com", "http://phishing-site.com",
                     "https://example.com", "http://steal-info.xyz",
                     "https://github.com", "http://fake-login-page.com"]
        test_labels = [0, 1, 0, 1, 0, 1]
        return test_urls, test_labels


# Создание словаря символов
def create_char_mapping(urls):
    chars = set("".join(urls))
    return {char: i + 1 for i, char in enumerate(chars)}


# Преобразование URL в последовательность
def url_to_seq(url, char_to_idx, max_len=MAX_URL_LEN):
    url = url.lower().strip()
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    domain = url.split('//')[-1].split('/')[0]
    seq = [char_to_idx.get(c, 0) for c in domain[:max_len]]
    seq += [0] * (max_len - len(seq))
    return seq


# Инициализация весов
def init_weights(input_size, hidden_size, output_size):
    np.random.seed(1)
    return {
        'w1': np.random.randn(input_size, hidden_size) * 0.01,
        'b1': np.zeros((1, hidden_size)),
        'w2': np.random.randn(hidden_size, output_size) * 0.01,
        'b2': np.zeros((1, output_size))
    }


# Прямое распространение
def forward(X, weights):
    z1 = np.dot(X, weights['w1']) + weights['b1']
    a1 = sigmoid(z1)
    z2 = np.dot(a1, weights['w2']) + weights['b2']
    a2 = sigmoid(z2)
    return a1, a2


# Обратное распространение
def backward(X, y, a1, a2, weights):
    m = X.shape[0]

    # Ошибка на выходном слое
    dz2 = a2 - y
    dw2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # Ошибка на скрытом слое
    dz1 = np.dot(dz2, weights['w2'].T) * sigmoid_derivative(a1)
    dw1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}


# Обновление весов
def update_weights(weights, grads, lr):
    weights['w1'] -= lr * grads['dw1']
    weights['b1'] -= lr * grads['db1']
    weights['w2'] -= lr * grads['dw2']
    weights['b2'] -= lr * grads['db2']
    return weights


# Обучение модели
def train_model():
    urls, labels = load_data()
    char_to_idx = create_char_mapping(urls)
    vocab_size = len(char_to_idx)

    # Подготовка данных
    X = np.array([url_to_seq(url, char_to_idx) for url in urls])
    y = np.array(labels).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Инициализация модели
    input_size = MAX_URL_LEN
    hidden_size = 64
    output_size = 1
    weights = init_weights(input_size, hidden_size, output_size)

    # Обучение
    losses = []
    accuracies = []

    for epoch in range(EPOCHS):
        epoch_loss = 0
        correct = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            batch_X = X_train[i:i + BATCH_SIZE]
            batch_y = y_train[i:i + BATCH_SIZE]

            # Прямое распространение
            a1, a2 = forward(batch_X, weights)

            # Обратное распространение
            grads = backward(batch_X, batch_y, a1, a2, weights)

            # Обновление весов
            weights = update_weights(weights, grads, LEARNING_RATE)

            # Расчет потерь и точности
            loss = np.mean((a2 - batch_y) ** 2)
            epoch_loss += loss
            correct += np.sum((a2 > 0.5) == batch_y)

        # Валидация
        _, val_output = forward(X_test, weights)
        val_acc = np.mean((val_output > 0.5) == y_test)

        losses.append(epoch_loss / len(X_train))
        accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}, Loss: {losses[-1]:.4f}, Accuracy: {val_acc:.4f}")

    # Сохранение модели
    np.save(MODEL_PATH, weights)
    print(f"Model saved to {MODEL_PATH}")

    # Графики обучения
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig('training_plot.png')
    plt.close()


# Предсказание
def predict(url):
    try:
        # Загружаем данные один раз при старте
        if not hasattr(predict, 'char_to_idx'):
            urls, _ = load_data()
            predict.char_to_idx = create_char_mapping(urls)
            predict.weights = np.load(MODEL_PATH, allow_pickle=True).item()

        seq = url_to_seq(url, predict.char_to_idx)
        _, output = forward(np.array([seq]), predict.weights)
        prob = output[0][0]

        return "Фишинг" if prob > 0.5 else "Безопасный", prob
    except Exception as e:
        print(f"Ошибка предсказания: {e}")
        return "Ошибка", 0.0


if __name__ == "__main__":
    # train_model()
    # print("Модель обучена------------------------")


    test_urls = [
        "https://google.com",
        "http://phishing-site.com",
        "https://example.com/login",
        "http://steal-your-data.xyz",
        "https://www.speedtest.net/",
        "https://www.speedtestt.net/",
        "https://anytask.org/",
        "https://www.youtube.com/watch?v=7inR7uMd-Ng",
        "https://www.youtube.com/"
    ]

    for url in test_urls:
        result, prob = predict(url)
        print(f"{url}: {result} (вероятность: {prob:.4f})")