import numpy as np
import matplotlib.pyplot as plt
from itertools import product

n = 10
print(f"Генерация таблицы истинности для n={n}...")
X_full = np.array(list(product([0, 1], repeat=n)))
# Функция AND: 1 только если все элементы равны 1
y_full = np.all(X_full == 1, axis=1).astype(float)
indices = np.random.permutation(len(X_full))
train_size = int(0.8 * len(X_full))

pos_idx = np.where(y_full == 1)[0][0]
train_indices = indices[:train_size]
if pos_idx not in train_indices:
    train_indices[0] = pos_idx

test_indices = np.array([i for i in range(len(X_full)) if i not in train_indices])

X_train, y_train = X_full[train_indices], y_full[train_indices]
X_test, y_test = X_full[test_indices], y_full[test_indices]

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train_perceptron(X_tr, y_tr, X_te, y_te, mode='fixed', lr=0.1, epochs=500):
    np.random.seed(42)
    W = np.random.randn(n) * 0.01
    b = np.random.randn() * 0.01

    train_hist, test_hist = [], []

    for epoch in range(epochs):
        for i in range(len(X_tr)):
            xi, yi = X_tr[i], y_tr[i]

            z = np.dot(W, xi) + b
            y_hat = sigmoid(z)

            alpha = lr if mode == 'fixed' else 1 / (1 + np.sum(xi ** 2) + 1)

            error = y_hat - yi
            W -= alpha * error * xi
            b -= alpha * error

        pred_tr = sigmoid(X_tr @ W + b)
        pred_te = sigmoid(X_te @ W + b)
        train_hist.append(compute_loss(y_tr, pred_tr))
        test_hist.append(compute_loss(y_te, pred_te))

        if train_hist[-1] < 0.001: break

    return W, b, train_hist, test_hist

print("Запуск обучения (фиксированный шаг)...")
W_f, b_f, tr_f, te_f = train_perceptron(X_train, y_train, X_test, y_test, mode='fixed', lr=0.1)

print("Запуск обучения (адаптивный шаг)...")
W_a, b_a, tr_a, te_a = train_perceptron(X_train, y_train, X_test, y_test, mode='adaptive')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(tr_f, label='Train (Fixed)')
plt.plot(te_f, label='Test (Fixed)', linestyle='--')
plt.title('Фиксированный шаг (lr=0.1)')
plt.xlabel('Эпоха');
plt.ylabel('BCE Loss');
plt.legend();
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(tr_a, label='Train (Adaptive)')
plt.plot(te_a, label='Test (Adaptive)', linestyle='--')
plt.title('Адаптивный шаг (Формула 2.36)')
plt.xlabel('Эпоха');
plt.ylabel('BCE Loss');
plt.legend();
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 5. Режим функционирования ---
def run_interactive(W, b):
    print("\n--- Проверка работы сети ---")
    user_input = input(f"Введите {n} бит через пробел (например, 1 1 ... 1): ")
    try:
        x_user = np.array([int(i) for i in user_input.split()])
        z = np.dot(W, x_user) + b
        prob = sigmoid(z)
        print(f"Вероятность '1': {prob:.6f}")
        print(f"Класс: {1 if prob > 0.5 else 0}")
    except:
        print("Ошибка ввода.")


# Вывод параметров
print(f"\nФинальные веса (адаптивный шаг): {np.round(W_a, 2)}")
print(f"Порог (bias): {b_a:.2f}")

run_interactive(W_a, b_a)