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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Отключаем предупреждения
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class PhishingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PhishingMLP(nn.Module):
    def __init__(self, input_size=16):
        super(PhishingMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Первый скрытый слой
        self.fc2 = nn.Linear(64, 32)          # Второй скрытый слой
        self.fc3 = nn.Linear(32, 1)           # Выходной слой
        self.dropout = nn.Dropout(0.4)        # Dropout для регуляризации

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


class PhishingDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, csv_path, epochs=50, batch_size=32, learning_rate=0.001):
        """Обучение модели на данных из CSV"""
        # 1. Загрузка данных
        df = pd.read_csv(csv_path)
        X = []
        y = []

        # 2. Извлечение признаков
        for _, row in df.iterrows():
            features = self.extract_features(row)
            X.append(features[:16])  # Берем первые 16 признаков
            y.append(features[16])   # 17-й элемент - метка

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        # 3. Нормализация данных
        X = self.scaler.fit_transform(X)

        # 4. Разделение на train/val
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5. Создание DataLoader
        train_dataset = PhishingDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = PhishingDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 6. Инициализация модели
        self.model = PhishingMLP(input_size=16).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 7. Обучение модели
        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            # Валидация
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Вывод статистики
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total

            print(
                f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')

            # Сохранение лучшей модели
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

        # Загрузка лучшей модели
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()

    def predict(self, url):
        """Предсказание для нового URL"""
        if self.model is None:
            raise Exception("Модель не обучена!")

        features = self.extract_live_features(url)
        features = self.scaler.transform(features.reshape(1, -1))
        features_tensor = torch.from_numpy(features).float().to(self.device)

        with torch.no_grad():
            output = self.model(features_tensor)
            confidence = output.item()
            prediction = "ФИШИНГ" if confidence > 0.5 else "БЕЗОПАСНЫЙ"

        return prediction, confidence

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
                self._has_popup_window(page_data),
                self._has_onmouseover(page_data),
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

    def _has_popup_window(self, soup):
        """Проверяет HTML на JS-код, создающий popup"""
        if not soup:
            return 1.0  # Если страница не загрузилась — считаем подозрительным

        scripts = soup.find_all("script")
        for script in scripts:
            if script.string and ("window.open" in script.string or "alert(" in script.string):
                return 1.0  # Нашли подозрительный JS
        return 0.

    def _has_onmouseover(self, soup):
        """Проверяет, использует ли страница onmouseover-события"""
        if not soup:
            return 1.0  # Страница не загрузилась — риск ↑

        # Ищем любые элементы с onmouseover
        return 1.0 if soup.find(attrs={"onmouseover": True}) else 0.0

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