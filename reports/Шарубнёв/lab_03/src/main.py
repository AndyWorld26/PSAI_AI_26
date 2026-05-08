import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [6, 1],
    [-6, 1],
    [6, -1],
    [-6, -1]
], dtype=float)

y = np.array([0, 0, 0, 1], dtype=float)

TARGET_ERROR = 0.01
MAX_EPOCHS = 100
LR_0 = 0.1



def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def predict_class(x1, x2, W, b):
    z = W[0] * x1 + W[1] * x2 + b
    prob = sigmoid(z)
    return 1 if prob >= 0.5 else 0, prob


# Конфигурация 1: MSE + фиксированный шаг
def train_mse_fixed(X, y, lr=LR_0, target_error=TARGET_ERROR, epochs=MAX_EPOCHS):
    np.random.seed(42)
    W = np.random.randn(2)
    b = np.random.randn()
    history = []

    for epoch in range(epochs):
        pred = X @ W + b
        error = pred - y
        mse = np.mean(error ** 2)
        history.append(mse)

        if mse <= target_error: break

        W -= lr * (2 / len(X)) * X.T @ error
        b -= lr * (2 / len(X)) * np.sum(error)
    return W, b, history


# Конфигурация 2: MSE + адаптивный шаг
def train_mse_adaptive(X, y, lr0=LR_0, target_error=TARGET_ERROR, epochs=MAX_EPOCHS):
    np.random.seed(42)
    W = np.random.randn(2)
    b = np.random.randn()
    history = []
    t = 0

    for epoch in range(epochs):
        epoch_mse = 0
        for i in range(len(X)):
            t += 1
            alpha = lr0 / (1 + 0.1 * t)

            xi = X[i]
            pred = np.dot(W, xi) + b
            error = pred - y[i]
            epoch_mse += error ** 2

            W -= alpha * 2 * error * xi
            b -= alpha * 2 * error

        mse = epoch_mse / len(X)
        history.append(mse)
        if mse <= target_error: break
    return W, b, history


# Конфигурация 3: BCE + фиксированный шаг
def train_bce_fixed(X, y, lr=LR_0, target_error=TARGET_ERROR, epochs=MAX_EPOCHS):
    np.random.seed(42)
    W = np.random.randn(2)
    b = np.random.randn()
    history = []

    for epoch in range(epochs):
        pred = sigmoid(X @ W + b)
        pred_safe = np.clip(pred, 1e-9, 1 - 1e-9)

        loss = -np.mean(y * np.log(pred_safe) + (1 - y) * np.log(1 - pred_safe))
        history.append(loss)

        if loss <= target_error: break

        error = pred - y  # Градиент BCE + Sigmoid = y_pred - y
        W -= lr * (1 / len(X)) * X.T @ error
        b -= lr * (1 / len(X)) * np.sum(error)
    return W, b, history


# Конфигурация 4: BCE + адаптивный шаг
def train_bce_adaptive(X, y, target_error=0.01, epochs=100):
    np.random.seed(42)
    W = np.random.randn(2)
    b = np.random.randn()
    history = []

    for epoch in range(epochs):
        epoch_loss = 0
        indices = np.random.permutation(len(X))

        for i in indices:
            xi = X[i]
            alpha_t = 1 / (1 + np.sum(xi ** 2) + 1)
            z = np.dot(W, xi) + b
            pred = sigmoid(z)
            error_val = pred - y[i]
            W -= alpha_t * error_val * xi
            b -= alpha_t * error_val
        all_preds = sigmoid(X @ W + b)
        all_preds_safe = np.clip(all_preds, 1e-9, 1 - 1e-9)
        loss = -np.mean(y * np.log(all_preds_safe) + (1 - y) * np.log(1 - all_preds_safe))
        history.append(loss)

        if loss <= target_error: break
    return W, b, history


W_mse_f, b_mse_f, hist_mse_f = train_mse_fixed(X, y)
W_mse_a, b_mse_a, hist_mse_a = train_mse_adaptive(X, y)
W_bce_f, b_bce_f, hist_bce_f = train_bce_fixed(X, y)
W_bce_a, b_bce_a, hist_bce_a = train_bce_adaptive(X, y)


plt.figure(figsize=(10, 6))
plt.plot(hist_mse_f, label=f"MSE Фиксированный шаг (Эпох: {len(hist_mse_f)})", linestyle='--')
plt.plot(hist_mse_a, label=f"MSE Адаптивный шаг (Эпох: {len(hist_mse_a)})", linestyle='-.')
plt.plot(hist_bce_f, label=f"BCE Фиксированный шаг (Эпох: {len(hist_bce_f)})", linewidth=2)
plt.plot(hist_bce_a, label=f"BCE Адаптивный шаг (Эпох: {len(hist_bce_a)})", linewidth=2)
plt.xlabel("Эпоха")
plt.ylabel("Функция ошибки (Loss / E)")
plt.title("Сравнение сходимости: MSE vs BCE (Порог ошибки $\leq$ 0.01)")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 8))

for i in range(len(X)):
    color = 'red' if y[i] == 1 else 'blue'
    plt.scatter(X[i, 0], X[i, 1], color=color, s=100, edgecolors='k')

x_vals = np.linspace(-8, 8, 200)

if abs(W_mse_f[1]) > 1e-6:
    y_vals_mse = -(W_mse_f[0] * x_vals + b_mse_f) / W_mse_f[1]
    plt.plot(x_vals, y_vals_mse, 'g--', alpha=0.6, label='Граница MSE (ЛР 1)')

if abs(W_bce_a[1]) > 1e-6:
    y_vals_bce = -(W_bce_a[0] * x_vals + b_bce_a) / W_bce_a[1]
    plt.plot(x_vals, y_vals_bce, 'k-', linewidth=2, label='Граница BCE + Адапт.')

plt.xlim(-8, 8)
plt.ylim(-4, 4)
plt.grid(True)
plt.title("Сравнение разделяющих линий")
plt.legend()
plt.show()

print("\n--- Режим классификации пользовательской точки ---")
x1_user = float(input("Введите координату x1: "))
x2_user = float(input("Введите координату x2: "))

cls_pred, prob_pred = predict_class(x1_user, x2_user, W_bce_a, b_bce_a)
print(f"Вероятность класса 1: {prob_pred:.4f}")
print(f"Предсказанный класс: {cls_pred}")

plt.figure(figsize=(8, 8))
for i in range(len(X)):
    color = 'red' if y[i] == 1 else 'blue'
    plt.scatter(X[i, 0], X[i, 1], color=color, s=100, alpha=0.5)

if abs(W_bce_a[1]) > 1e-6:
    y_vals_bce = -(W_bce_a[0] * x_vals + b_bce_a) / W_bce_a