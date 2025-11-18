import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.init as init
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

def get_data():
    from ucimlrepo import fetch_ucirepo 
    
    # fetch dataset 
    mushroom = fetch_ucirepo(id=73) 
    
    # data (as pandas dataframes) 
    X = mushroom.data.features 
    y = mushroom.data.targets 
    
    return X, y


X, y = get_data()

X['stalk-root'] = X['stalk-root'].fillna('-')

X = X.drop(['veil-type', 'gill-attachment', 'veil-color', 'stalk-color-below-ring'], axis=1)

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(X)

X = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(X.columns), index=X.index)


y['poisonous'] = y['poisonous'].map({'p': 0, 'e': 1})

y['edible'] = 1 - y['poisonous']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        init.xavier_uniform_(m.weight)  # Xavier uniform initialization for weights
        if m.bias is not None:
            m.bias.data.fill_(0.0)      # Initialize biases to zero

# Предполагаем, что X, y — pandas DataFrame/Series с числовыми признаками и метками классов 0 или 1

# Преобразуем в тензоры
X_torch = torch.tensor(X.values, dtype=torch.float32)
y_torch = torch.tensor(y.values, dtype=torch.float32)  # метки как целые числа

# Опционально one-hot кодируем y для вывода сети с 2 нейронами, можно и без этого с CrossEntropyLoss

# Модель нейросети
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

input_dim = X_torch.shape[1]
hidden1 = 16
hidden2 = 16
output_dim = 2

model = SimpleNN(input_dim, hidden1, hidden2, output_dim)

model.apply(init_weights)

# Функция потерь
criterion = nn.BCELoss()  # для бинарной классификации с выходом из 2 нейронов в сигмоиде нужен особый формат, ниже альтернативный вариант

# Градиентный спуск вручную без Optimizer
lr = 0.01
epochs = 10000

loss_values = []

for epoch in range(epochs):
    # Прямой проход
    outputs = model(X_torch)
    # Потеря
    loss = criterion(outputs, y_torch)
    
    # Обратный проход
    loss.backward()
    
    # Обновление весов
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad
    
    # Обнуляем градиенты
    model.zero_grad()
    loss_values.append(loss.item())

    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


plt.figure(figsize=(8, 6))
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.show()