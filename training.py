import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Конфигурация
MAX_URL_LEN = 100
BATCH_SIZE = 32
EPOCHS = 20
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
        # Чтение CSV с явным указанием заголовка
        data = pd.read_csv(DATA_PATH,
                           sep=',',
                           header=0,
                           names=['url', 'is_phishing'],
                           dtype={'url': str, 'is_phishing': str},  # Сначала читаем как строку
                           on_bad_lines='skip')  # Пропускаем проблемные строки

        # Преобразование меток в числовой формат с обработкой ошибок
        data['is_phishing'] = pd.to_numeric(data['is_phishing'], errors='coerce')

        # Удаление строк с пропущенными значениями
        data = data.dropna()

        # Преобразование в целые числа
        data['is_phishing'] = data['is_phishing'].astype(int)

        # Проверка распределения классов
        class_counts = data['is_phishing'].value_counts()
        print(f"Распределение классов:\n{class_counts}")

        return data["url"].values, data["is_phishing"].values

    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")


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



# Предсказание
def predict(url):
    try:
        if not hasattr(predict, 'char_to_idx'):
            urls, _ = load_data()
            predict.char_to_idx = create_char_mapping(urls)
            predict.weights = np.load(MODEL_PATH, allow_pickle=True).item()

        seq = url_to_seq(url, predict.char_to_idx)
        _, output = forward(np.array([seq]), predict.weights)
        prob = output[0][0]

        return "Фишинг" if prob > 0.1 else "Безопасный", prob
    except Exception as e:
        print(f"Ошибка предсказания: {e}")
        return "Ошибка", 0.0


if __name__ == "__main__":
    # train_model()
    # print("Модель обучена------------------------")

    test_urls = [
        # Легитимные
        "https://www.google.com",
        "https://www.youtube.com",
        "https://github.com",
        "https://www.wikipedia.org",
        "https://www.amazon.com",
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.linkedin.com",
        "https://twitter.com",
        "https://www.instagram.com",
        "https://www.reddit.com",
        "https://www.netflix.com",
        "https://www.spotify.com",
        "https://www.paypal.com",
        "https://www.ebay.com",
        "https://www.dropbox.com",
        "https://www.twitch.tv",
        "https://www.aliexpress.com",
        "https://www.adobe.com",
        "https://www.cnn.com",
        "https://www.bbc.com",
        "https://www.nytimes.com",
        "https://www.speedtest.net/",
        "https://translate.google.ru",
        "https://www.ozon.ru",

        # Фишинговые
        "http://secure-login-facebook.com",
        "https://g00gle-account.com",
        "http://amaz0n-payments.com",
        "https://apple-id-verify.com",
        "http://microsoft-security-update.com",
        "https://linkedin-profile-confirm.com",
        "http://paypal-account-limit.com",
        "https://www.speedtestt.net/",
        "http://steal-your-data.xyz",
        "http://phishing-site.com",
        "http://steamcommunity-support.com",
        "https://twitter-password-reset.com",
        "http://instagram-login-safe.com",
        "https://whatsapp-verification-code.com",
        "http://bankofamerica-securelogin.com",
        "https://chase-online-banking.com",
        "http://wellsfargo-account-alert.com",
        "https://ebay-item-confirmation.com",
        "http://dropbox-file-share-alert.com",
        "https://spotify-premium-renew.com",
        "http://twitch-account-recovery.com",
        "https://aliexpress-order-confirm.com",
        "http://adobe-id-verification.com",
        "https://cnn-breaking-news-alert.com",
        "http://bbc-account-update.com",
        "https://nytimes-subscription-renew.com",
        "http://walmart-order-confirmation.com",
        "https://target-special-offers.com",
        "http://bestbuy-electronic-deals.com",
        "https://stackoverflow-account-help.com",
        "http://quora-email-verification.com",
        "https://medium-membership-upgrade.com",
        "http://tumblr-blog-secure.com",
        "https://pinterest-pin-alert.com",
        "http://flickr-photo-update.com",
        "https://slack-workspace-invite.com",
        "http://trello-board-security.com",
        "https://notion-account-recovery.com",
        "http://zoom-meeting-invite.com",
        "https://skype-chat-update.com",
        "http://discord-server-alert.com",
        "https://telegram-account-verify.com",
        "http://signal-message-update.com",
        "https://mozilla-firefox-update.com",
        "http://duckduckgo-search-alert.com",
        "https://cloudflare-security-check.com",
        "http://digitalocean-server-alert.com",
        "https://nginx-config-update.com",
        "http://python-package-alert.com",
        "https://java-install-required.com",
        "http://nodejs-update-required.com",
        "https://react-security-alert.com",
        "http://netflix-renew-subscription.com",
        "https://facebook-login-secure.ru",
        "http://youtube-premium-offer.com",
        "https://github-account-verify.com",
        "http://wikipedia-donation-scam.com",
        "https://amazon-payment-confirm.com",
        "http://microsoft-office-update.com",
        "https://apple-support-center.com",
        "http://linkedin-job-offer.com",
        "https://twitter-account-lock.com",
        "http://instagram-verify-profile.com",
        "https://reddit-gold-scam.com",
        "http://netflix-payment-error.com",
        "https://spotify-family-scam.com",
        "http://paypal-limited-account.com",
        "https://ebay-refund-scam.com",
        "http://dropbox-hacked-alert.com",
        "https://twitch-subscriber-scam.com",
        "http://aliexpress-refund.com",
        "https://adobe-license-expired.com"
    ]

    for url in test_urls:
        result, prob = predict(url)
        print(f"{url}: {result} (вероятность: {prob:.4f})")