import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from doctr.models import crnn_vgg16_bn
from datasets import load_dataset
from doctr.datasets import VOCABS

# Параметры обучения
num_epochs = 10
learning_rate = 0.001
batch_size = 16

# Подключаем GPU, если доступно
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Загружаем модель
own_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=VOCABS["ukrainian"]).to(device)
# Загружаем датасет
dataset = load_dataset("DonkeySmall/OCR-Cyrillic-Printed-6", split='train')


# Создаем класс CustomDataset для оборачивания загруженного датасета
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example['image']
        target = example['text']

        # Преобразования: конвертация изображения и текста в нужный формат
        return image, target


def training_model_process():
    # Инициализируем экземпляр класса на основе CustomDataset
    train_dataset = CustomDataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Определяем функцию потерь и оптимизатор
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(own_model.parameters(), lr=learning_rate)

    # Цикл обучения
    for epoch in range(num_epochs):
        own_model.train()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Обнуляем градиенты
            optimizer.zero_grad()

            # Прогоняем изображение через модель
            outputs = own_model(images)

            # Вычисляем потери
            loss = criterion(outputs, targets)

            # Обратное распространение
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Сохранение дообученной модели
    torch.save(own_model.state_dict(), 'crnn_vgg16_bn_finetuned.pt')


def run_script():
    print(f'INFO: Запуск обучения модели')
    start = time.time()
    training_model_process()
    print(f'INFO: Время дообучения модели - {time.time() - start}')


run_script()
