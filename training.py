import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Конфигурация
MAX_URL_LEN = 100
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DATA_PATH = "urls.csv"
MODEL_PATH = "phishing_model.pth"


# Загрузка и предобработка данных
def load_data():
    try:
        # Явно указываем разделитель и обработку пустых значений
        data = pd.read_csv(DATA_PATH, sep=',', header=0, names=['url', 'is_phishing'], engine='python')
        # Удаление строк с NaN
        data = data.dropna()
        # Преобразование меток в числовой формат
        data['is_phishing'] = data['is_phishing'].astype(int)
        return data["url"].values, data["is_phishing"].values
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        # Возвращаем тестовые данные, если файл не найден
        test_urls = ["google.com", "phishing-site.com"]
        test_labels = [0, 1]
        return test_urls, test_labels

def create_char_mapping(urls):
    all_chars = set("".join(urls))
    return {char: idx + 1 for idx, char in enumerate(all_chars)}


def url_to_seq(url, char_to_idx, max_len=MAX_URL_LEN):
    # Добавляем http:// если URL не содержит схему
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    seq = [char_to_idx.get(char, 0) for char in url[:max_len]]
    seq += [0] * (max_len - len(seq))
    return seq


# Датасет и DataLoader
class URLDataset(Dataset):
    def __init__(self, urls, labels, char_to_idx):
        self.urls = urls
        self.labels = labels
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        seq = url_to_seq(self.urls[idx], self.char_to_idx)
        label = self.labels[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.float).unsqueeze(0)


# Модель (CNN)
class URLClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(64 * 25, 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))


# Обучение
def train_model():
    urls, labels = load_data()
    char_to_idx = create_char_mapping(urls)

    X_train, X_test, y_train, y_test = train_test_split(urls, labels, test_size=0.2, random_state=42)

    train_dataset = URLDataset(X_train, y_train, char_to_idx)
    test_dataset = URLDataset(X_test, y_test, char_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = URLClassifier(len(char_to_idx)).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                predicted = (outputs > 0.5).float()
                total += y.size(0)
                correct += (predicted == y).sum().item()

        print(f"Epoch {epoch + 1}, Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


# Предсказание
def predict(url):
    try:
        urls, _ = load_data()
        char_to_idx = create_char_mapping(urls)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = URLClassifier(len(char_to_idx)).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        seq = url_to_seq(url, char_to_idx)
        with torch.no_grad():
            output = model(torch.tensor([seq], device=device)).item()

        return "Фишинг" if output > 0.5 else "Безопасный"
    except Exception as e:
        print(f"Ошибка предсказания: {e}")
        return "Не удалось выполнить предсказание"


# Главная функция
if __name__ == "__main__":
    # train_model()
    # print("Модель обучена------------------------")

    test_urls = [
        "https://google.com",
        "http://phishing-site.com",
        "https://example.com/login",
        "http://steal-your-data.xyz"
    ]

    for url in test_urls:
        result, prob = predict(url)
        print(f"{url}: {result} (вероятность: {prob:.4f})")