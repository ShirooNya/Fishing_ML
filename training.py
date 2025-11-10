import random
import time

import joblib
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
    def __init__(self, input_size=83):  # 83 входных признака
        super(PhishingMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)   # Первый скрытый слой
        self.fc2 = nn.Linear(64, 32)    # Второй скрытый слой
        self.fc3 = nn.Linear(32, 1)     # Выходной слой
        self.dropout = nn.Dropout(0.4)   # Регуляризация

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))


class PhishingDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_scaler(self, path='scaler.save'):
        joblib.dump(self.scaler, path)

    def load_scaler(self, path='scaler.save'):
        self.scaler = joblib.load(path)

    def train_model(self, csv_path, epochs=10, batch_size=32, learning_rate=0.001):
        """Обучение модели на данных из CSV"""
        try:
            # Загрузка данных
            df = pd.read_csv(csv_path)

            # Очистка данных - удаление строк с nan в status
            df = df.dropna(subset=['status'])

            # Преобразование status в целые числа (на случай, если они записаны как float)
            df['status'] = df['status'].astype(int)

            if not set(df['status'].unique()).issubset({0, 1}):
                raise ValueError("Метки должны быть 0 или 1")

            X = df.iloc[:, 1:-1].values.astype(np.float32)
            y = df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

            # Нормализация признаков
            X = self.scaler.fit_transform(X)

            # Разделение данных
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Создание DataLoader
            train_dataset = PhishingDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            val_dataset = PhishingDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Инициализация модели
            self.model = PhishingMLP(input_size=X.shape[1]).to(self.device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

            # Обучение с проверкой выходов
            best_val_loss = float('inf')
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)

                    # Проверка выходов перед loss
                    if torch.any(outputs < 0) or torch.any(outputs > 1):
                        print("Ошибка: выходы за пределы [0, 1]")
                        print("Min output:", torch.min(outputs).item())
                        print("Max output:", torch.max(outputs).item())
                        raise ValueError("Выходы модели должны быть в [0, 1]")

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * inputs.size(0)

                # Валидация
                self.model.eval()
                val_loss, correct, total = 0.0, 0, 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                        predicted = (outputs > 0.5).float()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                # Логирование
                train_loss /= len(train_loader.dataset)
                val_loss /= len(val_loader.dataset)
                val_acc = correct / total
                print(
                    f'Epoch {epoch + 1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')

                # Сохранение лучшей модели
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'best_model.pth')

            # Загрузка лучшей модели
            self.model.load_state_dict(torch.load('best_model.pth'))
            return self.model

        except Exception as e:
            print(f"Ошибка при обучении: {str(e)}")
            raise

    def predict(self, url):
        """Предсказание для нового URL"""
        if not hasattr(self.scaler, 'mean_'):
            raise RuntimeError("Сначала обучите модель или загрузите scaler!")

        features = self.extract_live_features(url)
        features = self.scaler.transform(features.reshape(1, -1))
        features_tensor = torch.from_numpy(features).float().to(self.device)

        with torch.no_grad():
            output = self.model(features_tensor)
            return "ФИШИНГ" if output.item() > 0.5 else "БЕЗОПАСНЫЙ", output.item()

    def safe_float(self, value, default=0.0):
        """Безопасное преобразование в float с обработкой пустых строк"""
        try:
            if pd.isna(value) or str(value).strip() == '':
                return default
            return float(value)
        except (ValueError, TypeError):
            return default


    def extract_live_features(self, url):
        """Извлечение признаков в реальном времени"""
        try:
            # Базовые признаки URL
            ext = tldextract.extract(url)
            parsed_url = urllib.parse.urlparse(url)
            domain = f"{ext.domain}.{ext.suffix}"

            page_data = self._get_page_content(url)
            domain_info = self._get_domain_info(url)
            html_text = str(page_data) if page_data else ""

            # Формируем признаки
            features = [
                len(url),  # length_url
                len(parsed_url.netloc),  # length_hostname
                1.0 if self._is_ip_address(parsed_url.netloc) else 0.0,  # ip
                url.count('.'),  # nb_dots
                url.count('-'),  # nb_hyphens
                url.count('@'),  # nb_at
                url.count('?'),  # nb_qm
                url.count('&'),  # nb_and
                url.count('|'),  # nb_or
                url.count('='),  # nb_eq
                url.count('_'),  # nb_underscore
                url.count('~'),  # nb_tilde
                url.count('%'),  # nb_percent
                url.count('/'),  # nb_slash
                url.count('*'),  # nb_star
                url.count(':'),  # nb_colon
                url.count(','),  # nb_comma
                url.count(';'),  # nb_semicolumn
                url.count('$'),  # nb_dollar
                url.count(' '),  # nb_space
                1.0 if 'www' in parsed_url.netloc.lower() else 0.0,  # nb_www
                parsed_url.netloc.lower().count('.com'),  # nb_com
                url.count('//') - 1 if '//' in url[8:] else 0.0,  # nb_dslash (исключая начальные //)
                1.0 if 'http:' in parsed_url.path else 0.0,  # http_in_path
                1.0 if 'https' in parsed_url.path else 0.0,  # https_token
                self._ratio_digits(url),  # ratio_digits_url
                self._ratio_digits(parsed_url.netloc),  # ratio_digits_host
                1.0 if parsed_url.netloc.startswith('xn--') else 0.0,  # punycode
                1.0 if parsed_url.port is not None else 0.0,  # port
                1.0 if ext.suffix in parsed_url.path else 0.0,  # tld_in_path
                1.0 if ext.suffix in ext.subdomain else 0.0,  # tld_in_subdomain
                self._is_abnormal_subdomain(ext.subdomain),  # abnormal_subdomain
                len(ext.subdomain.split('.')),  # nb_subdomains
                self._has_prefix_suffix(ext.domain),  # prefix_suffix
                self._is_random_domain(ext.domain),  # random_domain
                self._is_shortening_service(url),  # shortening_service
                self._has_path_extension(parsed_url.path),  # path_extension
                self._count_redirections(url),  # nb_redirection
                self._count_external_redirections(url),  # nb_external_redirection
                len(html_text.split()),  # length_words_raw
                self._char_repeat_score(url),  # char_repeat
                self._shortest_word(html_text),  # shortest_words_raw
                self._shortest_word(ext.domain),  # shortest_word_host
                self._shortest_word(parsed_url.path),  # shortest_word_path
                self._longest_word(html_text),  # longest_words_raw
                self._longest_word(ext.domain),  # longest_word_host
                self._longest_word(parsed_url.path),  # longest_word_path
                self._avg_word_length(html_text),  # avg_words_raw
                self._avg_word_length(ext.domain),  # avg_word_host
                self._avg_word_length(parsed_url.path),  # avg_word_path
                self._phish_hints(url),  # phish_hints
                self._domain_in_brand(ext.domain),  # domain_in_brand
                self._brand_in_subdomain(ext.subdomain),  # brand_in_subdomain
                self._brand_in_path(parsed_url.path),  # brand_in_path
                self._is_suspicious_tld(ext.suffix),  # suspecious_tld
                0.0,  # statistical_report (нужен внешний сервис)
                self._count_hyperlinks(page_data),  # nb_hyperlinks
                self._ratio_int_hyperlinks(page_data, domain),  # ratio_intHyperlinks
                self._ratio_ext_hyperlinks(page_data, domain),  # ratio_extHyperlinks
                self._ratio_null_hyperlinks(page_data),  # ratio_nullHyperlinks
                self._count_external_css(page_data),  # nb_extCSS
                self._ratio_int_redirections(url),  # ratio_intRedirection
                self._ratio_ext_redirections(url),  # ratio_extRedirection
                self._ratio_int_errors(page_data),  # ratio_intErrors
                self._ratio_ext_errors(page_data),  # ratio_extErrors
                self._has_login_form(page_data),  # login_form
                self._has_external_favicon(page_data),  # external_favicon
                self._links_in_tags(page_data),  # links_in_tags
                self._has_submit_email(page_data),  # submit_email
                self._ratio_int_media(page_data, domain),  # ratio_intMedia
                self._ratio_ext_media(page_data, domain),  # ratio_extMedia
                self._check_sfh(page_data),  # sfh
                self._has_iframe(page_data),  # iframe
                self._has_popup_window(page_data),  # popup_window
                self._safe_anchor(page_data),  # safe_anchor
                self._has_onmouseover(page_data),  # onmouseover
                self._has_right_click(page_data),  # right_clic
                self._has_empty_title(page_data),  # empty_title
                self._domain_in_title(page_data, ext),  # domain_in_title
                self._domain_with_copyright(page_data, domain),  # domain_with_copyright
                1.0 if domain_info and domain_info.domain_name else 0.0,  # whois_registered_domain
                self._get_registration_length(domain_info),  # domain_registration_length
                self._get_domain_age(domain_info),  # domain_age
                0.0,  # web_traffic (нужен API)
                float(self._check_dns(ext)),  # dns_record
                float(self._check_google_index(url)),  # google_index
                self._get_page_rank(url),  # page_rank
            ]
            print(features)

            return np.array(features)
        except Exception as e:
            print(f"Ошибка анализа URL: {str(e)}")
            return np.zeros(len(features)) if 'features' in locals() else np.zeros(70)


    def _get_page_content(self, url):
        """Загрузка страницы с обходом защиты"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }

            # Добавляем задержку между запросами
            time.sleep(random.uniform(1, 3))

            response = requests.get(
                url,
                headers=headers,
                cookies={'session': 'fake_cookie_for_testing'},  # Может потребоваться реальная сессия
                timeout=10,
                verify=False,
                allow_redirects=True
            )

            if 'captcha' in response.text.lower() or response.status_code in [403, 429]:
                raise Exception("Обнаружена защита от ботов (CAPTCHA/403)")

            return BeautifulSoup(response.text, 'html.parser')

        except Exception as e:
            print(f"Ошибка загрузки {url}: {str(e)}")
            return None

    def _is_ip_address(self, hostname):
        """
        Проверяет, является ли переданный хост IP-адресом
        """
        try:
            socket.inet_aton(hostname)
            return True
        except socket.error:
            return False

    def _ratio_digits(self, text):
        """
        Вычисляет отношение цифр к общему количеству символов в тексте
        """
        if not text:
            return 0.0
        return sum(c.isdigit() for c in text) / len(text)

    def _is_abnormal_subdomain(self, subdomain):
        """
        Определяет аномальную длину поддомена
        """
        return 1.0 if len(subdomain) > 8 else 0.0

    def _has_prefix_suffix(self, domain):
        """
        Проверяет наличие дефиса в доменном имени
        """
        return 1.0 if '-' in domain else 0.0

    def _is_random_domain(self, domain):
        """
        Анализирует домен на "случайность" по соотношению гласных букв.
        Возвращает 1.0 если гласных меньше 30%, иначе 0.0.
        """
        if not domain:  # Проверка на пустую строку
            return 0.0

        vowels = sum(1 for c in domain.lower() if c in 'aeiou')
        ratio = vowels / len(domain)
        return 1.0 if ratio < 0.3 else 0.0

    def _is_shortening_service(self, url):
        """
        Проверяет, является ли URL сервисом сокращения ссылок
        Сравнивает с известными сервисами (bit.ly, goo.gl и др.)
        """
        services = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly']
        return 1.0 if any(s in url for s in services) else 0.0

    def _has_path_extension(self, path):
        """
        Проверяет наличие подозрительных расширений файлов в пути URL
        Ищет .exe, .js, .zip, .rar в конце пути
        """
        extensions = ['.exe', '.js', '.zip', '.rar']
        return 1.0 if any(path.endswith(ext) for ext in extensions) else 0.0

    def _count_redirections(self, url):
        """
        Подсчитывает общее количество редиректов при запросе URL
        """
        try:
            response = requests.get(url, allow_redirects=True, timeout=5, verify=False)
            return len(response.history)
        except:
            return 0.0

    def _count_external_redirections(self, url):
        """
        Считает количество внешних редиректов (на другие домены)
        """
        try:
            response = requests.get(url, allow_redirects=True, timeout=5, verify=False)
            domain = urllib.parse.urlparse(url).netloc
            return sum(1 for r in response.history if domain not in r.url)
        except:
            return 0.0

    def _char_repeat_score(self, text):
        """
        Вычисляет показатель повторяемости символов в тексте.
        Возвращает отношение максимального числа повторений символа к длине текста.
        """
        if not text:
            return 0.0
        counts = {}
        for c in text:
            counts[c] = counts.get(c, 0) + 1
        return max(counts.values()) / len(text)

    def _shortest_word(self, text):
        """
        Находит длину самого короткого слова в тексте
        """
        words = text.split()
        return min(len(word) for word in words) if words else 0.0

    def _longest_word(self, text):
        """
        Находит длину самого длинного слова в тексте
        """
        words = text.split()
        return max(len(word) for word in words) if words else 0.0

    def _avg_word_length(self, text):
        """
        Вычисляет среднюю длину слов в тексте
        """
        words = text.split()
        return sum(len(word) for word in words) / len(words) if words else 0.0

    def _phish_hints(self, url):
        """
        Ищет фишинговые ключевые слова в URL.
        Проверяет наличие слов: login, verify, account, secure, update
        """
        hints = ['login', 'verify', 'account', 'secure', 'update']
        return 1.0 if any(hint in url.lower() for hint in hints) else 0.0

    def _domain_in_brand(self, domain):
        """
        Проверяет наличие брендов в доменном имени.
        Ищет: paypal, ebay, amazon, bank
        """
        brands = ['paypal', 'ebay', 'amazon', 'bank']
        return 1.0 if any(brand in domain.lower() for brand in brands) else 0.0

    def _brand_in_subdomain(self, subdomain):
        """
        Проверяет наличие брендов в поддомене.
        Ищет те же бренды, что и в _domain_in_brand
        """
        brands = ['paypal', 'ebay', 'amazon', 'bank']
        return 1.0 if any(brand in subdomain.lower() for brand in brands) else 0.0

    def _brand_in_path(self, path):
        """
        Проверяет наличие брендов в пути URL.
        Ищет те же бренды, что и в _domain_in_brand.
        """
        brands = ['paypal', 'ebay', 'amazon', 'bank']
        return 1.0 if any(brand in path.lower() for brand in brands) else 0.0

    def _is_suspicious_tld(self, tld):
        """
        Проверяет TLD (домен верхнего уровня) на подозрительность.
        Сравнивает с списком: .xyz, .top, .gq, .tk.
        """
        suspicious = ['.xyz', '.top', '.gq', '.tk']
        return 1.0 if tld in suspicious else 0.0

    def _ratio_int_hyperlinks(self, soup, domain):
        """
        Вычисляет отношение внутренних гиперссылок к общему количеству.
        Внутренние ссылки содержат указанный домен.
        """
        if not soup:
            return 0.0
        links = soup.find_all('a', href=True)
        if not links:
            return 0.0
        internal = sum(1 for link in links if domain in link['href'])
        return internal / len(links)

    def _ratio_ext_hyperlinks(self, soup, domain):
        """
        Вычисляет отношение внешних гиперссылок к общему количеству.
        Внешние ссылки не содержат указанный домен и начинаются с http
        """
        if not soup:
            return 0.0
        links = soup.find_all('a', href=True)
        if not links:
            return 0.0
        external = sum(1 for link in links if domain not in link['href'] and link['href'].startswith('http'))
        return external / len(links)

    def _ratio_null_hyperlinks(self, soup):
        """
        Вычисляет отношение "нулевых" гиперссылок (не начинающихся с http).
        """
        if not soup:
            return 0.0
        links = soup.find_all('a', href=True)
        if not links:
            return 0.0
        null = sum(1 for link in links if not link['href'].startswith('http'))
        return null / len(links)

    def _links_in_tags(self, soup):
        """
        Подсчитывает количество ссылок в определенных HTML-тегах
        Анализирует теги: img, script, iframe, form
        Возвращает среднее количество на тег
        """
        if not soup:
            return 0.0
        tags = ['img', 'script', 'iframe', 'form']
        return sum(len(soup.find_all(tag)) for tag in tags) / 4.0

    def _ratio_int_redirections(self, url):
        """
        Вычисляет отношение внутренних редиректов к общему количеству.
        Внутренние редиректы содержат исходный домен.
        """
        try:
            response = requests.get(url, allow_redirects=True, timeout=5, verify=False)
            domain = urllib.parse.urlparse(url).netloc
            internal = sum(1 for r in response.history if domain in r.url)
            return internal / len(response.history) if response.history else 0.0
        except:
            return 0.0

    def _ratio_ext_redirections(self, url):
        """
        Вычисляет отношение внешних редиректов к общему количеству.
        Внешние редиректы не содержат исходный домен.
        """
        try:
            response = requests.get(url, allow_redirects=True, timeout=5, verify=False)
            domain = urllib.parse.urlparse(url).netloc
            external = sum(1 for r in response.history if domain not in r.url)
            return external / len(response.history) if response.history else 0.0
        except:
            return 0.0

    def _ratio_int_errors(self, soup):
        """
        Заглушка для анализа отношения внутренних ошибок.
        """
        return 0.0

    def _ratio_ext_errors(self, soup):
        """
        Заглушка для анализа отношения внешних ошибок.
        """
        return 0.0

    def _ratio_int_media(self, soup, domain):
        """
        Вычисляет отношение внутренних медиа-ресурсов к общему количеству.
        Анализирует теги img, video, audio с src, содержащим домен.
        """
        if not soup:
            return 0.0
        media = soup.find_all(['img', 'video', 'audio'])
        if not media:
            return 0.0
        internal = sum(1 for m in media if domain in m.get('src', ''))
        return internal / len(media)

    def _ratio_ext_media(self, soup, domain):
        """
        Вычисляет отношение внешних медиа-ресурсов к общему количеству.
        Анализирует теги img, video, audio с внешними src.
        """
        if not soup:
            return 0.0
        media = soup.find_all(['img', 'video', 'audio'])
        if not media:
            return 0.0
        external = sum(1 for m in media if domain not in m.get('src', '') and m.get('src', '').startswith('http'))
        return external / len(media)

    def _check_sfh(self, soup):
        """
        Проверяет формы на наличие пустого action (Server Form Handler).
        Возвращает отношение форм без action к общему количеству форм.
        """
        if not soup:
            return 0.0
        forms = soup.find_all('form')
        if not forms:
            return 0.0
        empty = sum(1 for form in forms if not form.get('action'))
        return empty / len(forms)

    def _safe_anchor(self, soup):
        """
        Проверяет ссылки на безопасность
        Возвращает отношение безопасных ссылок к общему количеству.
        """
        if not soup:
            return 0.0
        links = soup.find_all('a', href=True)
        if not links:
            return 0.0
        safe = sum(1 for link in links if 'javascript:' not in link['href'].lower())
        return safe / len(links)

    def _has_right_click(self, soup):
        """
        Проверяет наличие скриптов, блокирующих правую кнопку мыши.
        """
        if not soup:
            return 1.0
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'event.button==2' in script.string:
                return 1.0
        return 0.0

    def _domain_with_copyright(self, soup, domain):
        """
        Проверяет наличие упоминания домена в текстах copyright.
        """
        if not soup:
            return 0.0
        copyrights = soup.find_all(string=lambda text: 'copyright' in text.lower())
        if not copyrights:
            return 0.0
        return 1.0 if any(domain.lower() in c.lower() for c in copyrights) else 0.0

    def _count_hyperlinks(self, soup):
        """
        Подсчитывает общее количество гиперссылок на странице.
        """
        if not soup:
            return 0.0
        return float(len(soup.find_all('a', href=True)))

    def _count_external_css(self, soup):
        """Подсчитывает количество внешних CSS-файлов"""
        if not soup:
            return 0.0
        css_links = soup.find_all('link', {'rel': 'stylesheet'})
        external = sum(1 for link in css_links if link.get('href', '').startswith('http'))
        return float(external)

    def _has_login_form(self, soup):
        """Проверяет наличие форм входа"""
        if not soup:
            return 0.0
        forms = soup.find_all('form')
        for form in forms:
            if any(input_tag.get('type') in ['password', 'email', 'text']
                   for input_tag in form.find_all('input')):
                return 1.0
        return 0.0

    def _has_external_favicon(self, soup):
        """Проверяет использование внешнего favicon"""
        if not soup:
            return 0.0
        favicons = soup.find_all('link', rel=lambda x: x and 'icon' in x.lower())
        external = any(link.get('href', '').startswith('http') for link in favicons)
        return 1.0 if external else 0.0

    def _has_submit_email(self, soup):
        """Проверяет наличие формы с отправкой email"""
        if not soup:
            return 0.0
        forms = soup.find_all('form')
        for form in forms:
            if 'mailto:' in str(form).lower():
                return 1.0
            if any('email' in input_tag.get('name', '').lower()
                   for input_tag in form.find_all('input')):
                return 1.0
        return 0.0

    def _has_iframe(self, soup):
        """Проверяет наличие iframe"""
        if not soup:
            return 0.0
        return 1.0 if soup.find('iframe') else 0.0

    def _has_popup_window(self, soup):
        """Проверяет наличие JavaScript для popup-окон"""
        if not soup:
            return 0.0
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and ('window.open' in script.string or 'alert(' in script.string):
                return 1.0
        return 0.0

    def _has_onmouseover(self, soup):
        """Проверяет использование onmouseover"""
        if not soup:
            return 0.0
        return 1.0 if soup.find(attrs={"onmouseover": True}) else 0.0

    def _has_empty_title(self, soup):
        """Проверяет пустой ли title страницы"""
        if not soup:
            return 1.0  # Если страница не загрузилась, считаем title пустым
        title = soup.title
        return 1.0 if not title or not title.string or not title.string.strip() else 0.0

    def _domain_in_title(self, soup, ext):
        """Проверяет содержится ли домен в title страницы"""
        if not soup or not soup.title or not soup.title.string:
            return 0.0
        domain = f"{ext.domain}.{ext.suffix}"
        return 1.0 if domain.lower() in soup.title.string.lower() else 0.0

    def _get_registration_length(self, domain_info):
        """Вычисляет длительность регистрации домена (в днях)"""
        if not domain_info or not domain_info.creation_date:
            return 0.0

        if isinstance(domain_info.creation_date, list):
            creation_date = domain_info.creation_date[0]
        else:
            creation_date = domain_info.creation_date

        try:
            if isinstance(creation_date, str):
                creation_date = datetime.strptime(creation_date, '%Y-%m-%d %H:%M:%S')

            expiry_date = domain_info.expiration_date
            if expiry_date and isinstance(expiry_date, str):
                expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d %H:%M:%S')

            if expiry_date and isinstance(expiry_date, datetime):
                return float((expiry_date - creation_date).days)
            return float((datetime.now() - creation_date).days)
        except:
            return 0.0

    def _get_domain_age(self, domain_info):
        """Вычисляет возраст домена (в днях)"""
        if not domain_info or not domain_info.creation_date:
            return 0.0

        if isinstance(domain_info.creation_date, list):
            creation_date = domain_info.creation_date[0]
        else:
            creation_date = domain_info.creation_date

        try:
            if isinstance(creation_date, str):
                creation_date = datetime.strptime(creation_date, '%Y-%m-%d %H:%M:%S')
            return float((datetime.now() - creation_date).days)
        except:
            return 0.0

    def _check_dns(self, ext):
        """Проверяет наличие DNS-записей"""
        domain = f"{ext.domain}.{ext.suffix}"
        try:
            socket.gethostbyname(domain)
            return True
        except socket.gaierror:
            return False

    def _check_google_index(self, url):
        """Проверяет индексируется ли URL в Google"""
        try:
            query = f"site:{url}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(f"https://www.google.com/search?q={query}", headers=headers)
            return "did not match any documents" not in response.text
        except:
            return False

    def _get_page_rank(self, url):
        """Заглушка для PageRank (реализация требует API)"""
        return 0.0

    def _has_ssl(self, url):
        """Проверяет наличие SSL-сертификата"""
        try:
            if url.startswith('https://'):
                return 1.0
            response = requests.head(url, timeout=5, verify=False)
            return 1.0 if response.url.startswith('https://') else 0.0
        except:
            return 0.0

    def _get_domain_info(self, url):
        """Получает WHOIS-информацию о домене"""
        try:
            domain = tldextract.extract(url).fqdn
            if not domain:
                return None
            return whois.whois(domain)
        except:
            return None


if __name__ == "__main__":
    detector = PhishingDetector()

    # Обучение модели
    print("Обучение модели...")
    detector.train_model("urls.csv")
    detector.save_scaler()

    # detector.load_scaler()
    # Проверка URL
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