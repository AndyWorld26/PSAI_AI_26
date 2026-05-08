import numpy as np
import matplotlib.pyplot as plt
from itertools import product

n = 10
print(f"Генерация таблицы истинности для функции AND (n={n})...")
X_full = np.array(list(product([0, 1], repeat=n)))
y_full = np.all(X_full == 1, axis=1).astype(float)

print("\nФрагмент таблицы истинности (первые 5 и последняя строки):")
for i in list(range(5)) + [len(y_full) - 1]:
    print(f"Вход: {X_full[i]} -> Выход: {int(y_full[i])}")

np.random.seed(42)
indices = np.random.permutation(len(X_full))
train_size = int(0.8 * len(X_full))

pos_idx = np.where(y_full == 1)[0][0]
train_indices = indices[:train_size]
if pos_idx not in train_indices:
    train_indices[0] = pos_idx

X_train, y_train = X_full[train_indices], y_full[train_indices]
test_indices = np.array([i for i in range(len(X_full)) if i not in train_indices])
X_test, y_test = X_full[test_indices], y_full[test_indices]

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def calc_sum_error(y_true, y_pred, loss_type='bce'):
    if loss_type == 'mse':
        return np.sum((y_true - y_pred) ** 2)
    y_pred_safe = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.sum(y_true * np.log(y_pred_safe) + (1 - y_true) * np.log(1 - y_pred_safe))


def calculate_accuracy(X, y, W, b):
    preds = sigmoid(np.dot(X, W) + b)
    return np.mean((preds >= 0.5).astype(float) == y) * 100

def train_perceptron(X_tr, y_tr, loss_type='bce', mode='fixed', lr=0.1, target_err=0.01, epochs=200):
    np.random.seed(42)
    W = np.random.randn(n) * 0.01
    b = np.random.randn() * 0.01
    history = []

    print(f"\n--- Старт: {loss_type.upper()} | {mode} ---")

    for epoch in range(1, epochs + 1):
        for i in range(len(X_tr)):
            xi, yi = X_tr[i], y_tr[i]
            y_hat = sigmoid(np.dot(W, xi) + b)

            alpha = lr if mode == 'fixed' else 1 / (1 + np.sum(xi ** 2) + 1)

            if loss_type == 'mse':
                error = (y_hat - yi) * (y_hat * (1 - y_hat))
            else:
                error = y_hat - yi

            W -= alpha * error * xi
            b -= alpha * error

        all_preds = sigmoid(np.dot(X_tr, W) + b)
        loss = calc_sum_error(y_tr, all_preds, loss_type)
        history.append(loss)

        if epoch % 50 == 0 or epoch == 1:
            print(f"Эпоха {epoch:3}: Ошибка = {loss:.6f}")

        if loss <= target_err:
            print(f"Сошлось на {epoch} эпохе!")
            break

    return W, b, history

TARGET_ERROR = 0.05
MAX_EPOCHS = 300

res_a = train_perceptron(X_train, y_train, 'mse', 'fixed', lr=0.5, target_err=TARGET_ERROR, epochs=MAX_EPOCHS)
res_b = train_perceptron(X_train, y_train, 'mse', 'adaptive', target_err=TARGET_ERROR, epochs=MAX_EPOCHS)
res_v = train_perceptron(X_train, y_train, 'bce', 'fixed', lr=0.5, target_err=TARGET_ERROR, epochs=MAX_EPOCHS)
res_g = train_perceptron(X_train, y_train, 'bce', 'adaptive', target_err=TARGET_ERROR, epochs=MAX_EPOCHS)

results = {"А. MSE Фикс": res_a, "Б. MSE Адапт": res_b, "В. BCE Фикс": res_v, "Г. BCE Адапт": res_g}

print("\n" + "=" * 85)
print(f"{'Конфигурация':<18} | {'Эпох':<5} | {'Acc Train':<10} | {'Acc Test':<10} | {'Acc Full'}")
print("=" * 85)
for name, (W, b, hist) in results.items():
    tr = calculate_accuracy(X_train, y_train, W, b)
    te = calculate_accuracy(X_test, y_test, W, b)
    fu = calculate_accuracy(X_full, y_full, W, b)
    print(f"{name:<18} | {len(hist):<5} | {tr:<9.2f}% | {te:<9.2f}% | {fu:.2f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for name, (_, _, hist) in results.items():
    ax1.plot(hist, label=name)
ax1.set_yscale('log')
ax1.set_title('Общий вид сходимости (Логарифмическая шкала)')
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Суммарная ошибка (Es)')
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

ax2.plot(results["А. MSE Фикс"][2], label="MSE Фикс", color='blue')
ax2.plot(results["Б. MSE Адапт"][2], label="MSE Адапт", color='cyan')
ax2.set_title('Детализация MSE (Линейная шкала)')
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Es')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

best_W, best_b, _ = results["Г. BCE Адапт"]

print("\n" + "=" * 30)
print("РЕЖИМ ФУНКЦИОНИРОВАНИЯ")
print("=" * 30)
print(f"Введите {n} значений (0 или 1) через пробел.")

while True:
    user_input = input("\nВаш вектор (или 'exit' для выхода): ").strip()
    if user_input.lower() == 'exit':
        break

    try:
        x_user = np.array([int(i) for i in user_input.split()])
        if len(x_user) != n:
            print(f"Ошибка: нужно ввести ровно {n} чисел.")
            continue

        prob = sigmoid(np.dot(best_W, x_user) + best_b)
        pred_class = 1 if prob >= 0.5 else 0

        true_class = 1 if np.all(x_user == 1) else 0
        status = "Совпадает с таблицей истинности" if pred_class == true_class else "Расхождение"

        print(f"Вероятность ŷ: \t{prob:.4f}")
        print(f"Класс: \t\t{pred_class}")
        print(f"Статус: \t{status}")

    except ValueError:
        print("Ошибка: вводите только числа 0 и 1 через пробел.")