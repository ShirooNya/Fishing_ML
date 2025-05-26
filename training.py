import numpy as np
import pandas as pd
import whois
import urllib.parse
from datetime import datetime
import tldextract
import requests
from bs4 import BeautifulSoup
import socket
from urllib3.exceptions import InsecureRequestWarning

# Отключаем предупреждения
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class PhishingDetectorNN:
    def __init__(self, input_size, hidden_size=10, output_size=1):
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.hidden = self.relu(np.dot(X, self.weights1) + self.bias1)
        self.output = self.sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output

    def backward(self, X, y, output, learning_rate):
        output_error = y - output
        output_delta = output_error * self.sigmoid(self.output) * (1 - self.sigmoid(self.output))

        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * self.relu_derivative(self.hidden)

        self.weights2 += self.hidden.T.dot(output_delta) * learning_rate
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights1 += X.T.dot(hidden_delta) * learning_rate
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs=3000, learning_rate=0.01, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output, learning_rate)

            if epoch % 500 == 0:
                output = self.forward(X)
                loss = np.mean(np.square(y - output))
                accuracy = np.mean((output > 0.5) == y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def predict(self, X, threshold=0.5):
        output = self.forward(X)
        return (output > threshold).astype(int), output


class PhishingDetector:
    def __init__(self):
        self.model = None

    def train_model(self, csv_path):
        """Обучение модели на данных из CSV"""
        # 1. Загрузка данных
        df = pd.read_csv(csv_path)
        X = []
        y = []

        # 2. Извлечение признаков
        for _, row in df.iterrows():
            features = self.extract_features(row)
            X.append(features[:16])  # Берем первые 16 признаков
            y.append(features[16])  # 17-й элемент - метка

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        # 4. Создание и обучение модели
        input_size = X.shape[1]
        self.model = PhishingDetectorNN(input_size=16)  # 16 входных признаков
        self.model.train(X, y)

    def predict(self, url):
        """Предсказание для нового URL"""
        if self.model is None:
            raise Exception("Модель не обучена!")

        features = self.extract_live_features(url)
        prediction, confidence = self.model.predict(features.reshape(1, -1))
        return "ФИШИНГ" if prediction[0][0] > 0.5 else "БЕЗОПАСНЫЙ", confidence[0][0]

    def safe_float(self, value, default=0.0):
        """Безопасное преобразование в float с обработкой пустых строк"""
        try:
            if pd.isna(value) or str(value).strip() == '':
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    def extract_features(self, row):
        """Извлечение признаков из строки CSV с защитой от пустых значений"""
        features = [
            self.safe_float(row.get('nb_hyperlinks', 0)),
            self.safe_float(row.get('nb_extCSS', 0)),
            self.safe_float(row.get('login_form', 0)),
            self.safe_float(row.get('external_favicon', 0)),
            self.safe_float(row.get('submit_email', 0)),
            self.safe_float(row.get('iframe', 0)),
            self.safe_float(row.get('popup_window', 0)),
            self.safe_float(row.get('onmouseover', 0)),
            self.safe_float(row.get('empty_title', 0)),
            self.safe_float(row.get('domain_in_title', 0)),
            self.safe_float(row.get('domain_registration_length', 0)),
            self.safe_float(row.get('domain_age', 0)),
            self.safe_float(row.get('web_traffic', 0)),
            self.safe_float(row.get('dns_record', 0)),
            self.safe_float(row.get('google_index', 0)),
            self.safe_float(row.get('page_rank', 0)),
            self.safe_float(row.get('status', 0))
        ]
        return np.array(features)

    def extract_live_features(self, url):
        """Извлечение признаков в реальном времени"""
        try:
            # 1. Базовые признаки URL
            ext = tldextract.extract(url)
            parsed_url = urllib.parse.urlparse(url)

            # 2. Получаем данные в реальном времени
            page_data = self._get_page_content(url)
            domain_info = self._get_domain_info(url)

            # 3. Формируем признаки
            features = [
                self._count_hyperlinks(page_data),  # nb_hyperlinks
                self._count_external_css(page_data),  # nb_extCSS
                self._has_login_form(page_data),  # login_form
                self._has_external_favicon(page_data),  # external_favicon
                self._has_submit_email(page_data),  # submit_email
                self._has_iframe(page_data),  # iframe
                0.0,  # popup_window (нужен JS)
                0.0,  # onmouseover (нужен JS)
                self._has_empty_title(page_data),  # empty_title
                self._domain_in_title(page_data, ext),  # domain_in_title
                self._get_registration_length(domain_info),  # domain_registration_length
                self._get_domain_age(domain_info),  # domain_age
                0.0,  # web_traffic (нужен API)
                float(self._check_dns(ext)),  # dns_record
                float(self._check_google_index(url)),  # google_index
                self._get_page_rank(url),  # page_rank
            ]

            return np.array(features)
        except Exception as e:
            print(f"Ошибка анализа URL: {str(e)}")
            return np.zeros(16)  # Возвращаем нули при ошибке

    def _get_page_content(self, url):
        """Получение содержимого страницы"""
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            return BeautifulSoup(response.text, 'html.parser')
        except:
            return BeautifulSoup("", 'html.parser')

    def _get_domain_info(self, url):
        """Получение WHOIS информации"""
        ext = tldextract.extract(url)
        try:
            return whois.whois(f"{ext.domain}.{ext.suffix}")
        except:
            return None

    def _count_hyperlinks(self, soup):
        """Подсчет гиперссылок на странице"""
        return len(soup.find_all('a', href=True))

    def _count_external_css(self, soup):
        """Подсчет внешних CSS"""
        return len([link for link in soup.find_all('link')
                    if 'stylesheet' in link.get('rel', [])
                    and 'http' in link.get('href', '')])

    def _has_login_form(self, soup):
        """Проверка наличия формы логина"""
        return 1.0 if soup.find('input', {'type': 'password'}) else 0.0

    def _has_external_favicon(self, soup):
        """Проверка внешнего фавикона"""
        favicon = soup.find('link', rel='icon')
        return 1.0 if favicon and 'http' in favicon.get('href', '') else 0.0

    def _has_submit_email(self, soup):
        """Проверка поля для email"""
        return 1.0 if soup.find('input', {'type': 'email'}) else 0.0

    def _has_iframe(self, soup):
        """Проверка iframe"""
        return 1.0 if soup.find('iframe') else 0.0

    def _has_empty_title(self, soup):
        """Проверка пустого заголовка"""
        title = soup.title.string if soup.title else ''
        return 1.0 if not title.strip() else 0.0

    def _domain_in_title(self, soup, ext):
        """Проверка домена в заголовке"""
        if not soup.title:
            return 0.0
        title = soup.title.string.lower()
        return 1.0 if ext.domain.lower() in title else 0.0

    def _get_registration_length(self, domain_info):
        """Срок регистрации домена в днях"""
        if not domain_info or not domain_info.expiration_date:
            return 0.0

        if isinstance(domain_info.expiration_date, list):
            exp_date = domain_info.expiration_date[0]
        else:
            exp_date = domain_info.expiration_date

        return (exp_date - datetime.now()).days

    def _get_domain_age(self, domain_info):
        """Возраст домена в днях"""
        if not domain_info or not domain_info.creation_date:
            return 0.0

        if isinstance(domain_info.creation_date, list):
            creation_date = domain_info.creation_date[0]
        else:
            creation_date = domain_info.creation_date

        return (datetime.now() - creation_date).days

    def _check_dns(self, ext):
        """Проверка DNS записи"""
        try:
            socket.gethostbyname(f"{ext.domain}.{ext.suffix}")
            return 1
        except:
            return 0

    def _check_google_index(self, url):
        """Проверка индексации в Google (заглушка)"""
        return 0

    def _get_page_rank(self, url):
        """PageRank (заглушка)"""
        return 0.0


# Пример использования
if __name__ == "__main__":
    detector = PhishingDetector()

    # 1. Обучение модели
    print("Обучение модели...")
    detector.train_model("urls.csv")

    # 2. Проверка URL
    while True:
        print("\n" + "=" * 50)
        url = input("Введите URL для проверки (или 'exit'): ").strip()
        if url.lower() == 'exit':
            break

        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        result, confidence = detector.predict(url)
        print(f"\nРезультат для {url}:")
        print(f"Вероятность фишинга: {confidence * 100:.2f}%")
        print(f"Заключение: {result}")