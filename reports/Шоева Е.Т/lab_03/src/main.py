import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

X = np.array([
    [6, 2],
    [-6, 2],
    [6, -2],
    [-6, -2]
], dtype=float)

y = np.array([0, 0, 1, 0], dtype=float)

X = X / 6.0
X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

Ee_mse = 0.01
Ee_bce = 0.1

max_epochs = 2000
lr = 0.1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def mse(y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred) ** 2)

def bce(y_true, y_pred):
    eps = 1e-8
    return -np.sum(y_true*np.log(y_pred+eps) +
                   (1-y_true)*np.log(1-y_pred+eps))

w_mse_fixed = np.random.randn(3) * 0.1
Es_mse_fixed = []
epochs_mse_fixed = 0

while True:
    y_pred = X_bias @ w_mse_fixed
    error = y - y_pred
    Es = mse(y, y_pred)
    Es_mse_fixed.append(Es)

    if Es <= Ee_mse or epochs_mse_fixed >= max_epochs:
        break

    w_mse_fixed += lr * (X_bias.T @ error)
    epochs_mse_fixed += 1

print("Эпохи MSE фиксированный =", epochs_mse_fixed)

w_mse_adapt = np.random.randn(3) * 0.1
Es_mse_adapt = []
epochs_mse_adapt = 0

while True:
    for i in range(len(X_bias)):
        xi = X_bias[i]
        yi = y[i]

        y_pred = np.dot(xi, w_mse_adapt)
        error = yi - y_pred

        alpha = 1 / (1 + np.sum(xi[:-1]**2))
        w_mse_adapt += alpha * error * xi

    y_all = X_bias @ w_mse_adapt
    Es = mse(y, y_all)
    Es_mse_adapt.append(Es)

    if Es <= Ee_mse or epochs_mse_adapt >= max_epochs:
        break

    epochs_mse_adapt += 1

print("Эпохи MSE адаптивный =", epochs_mse_adapt)

w_bce_fixed = np.random.randn(3) * 0.1
Es_bce_fixed = []
epochs_bce_fixed = 0

while True:
    y_pred = sigmoid(X_bias @ w_bce_fixed)
    error = y_pred - y
    Es = bce(y, y_pred)
    Es_bce_fixed.append(Es)

    if Es <= Ee_bce or epochs_bce_fixed >= max_epochs:
        break

    w_bce_fixed -= lr * (X_bias.T @ error)
    epochs_bce_fixed += 1

print("Эпохи BCE фиксированный =", epochs_bce_fixed)

w_bce_adapt = np.random.randn(3) * 0.1
Es_bce_adapt = []
epochs_bce_adapt = 0

while True:
    for i in range(len(X_bias)):
        xi = X_bias[i]
        yi = y[i]

        y_pred = sigmoid(np.dot(xi, w_bce_adapt))
        error = y_pred - yi

        alpha = 1 / (1 + np.sum(xi[:-1]**2))
        w_bce_adapt -= alpha * error * xi

    y_all = sigmoid(X_bias @ w_bce_adapt)
    Es = bce(y, y_all)
    Es_bce_adapt.append(Es)

    if Es <= Ee_bce or epochs_bce_adapt >= max_epochs:
        break

    epochs_bce_adapt += 1

print("Эпохи BCE адаптивный =", epochs_bce_adapt)

plt.figure(figsize=(10,6))
plt.plot(Es_mse_fixed, label="MSE фиксированный")
plt.plot(Es_mse_adapt, label="MSE адаптивный")
plt.plot(Es_bce_fixed, label="BCE фиксированный")
plt.plot(Es_bce_adapt, label="BCE адаптивный")
plt.xlabel("Эпоха")
plt.ylabel("Ошибка")
plt.title("Сравнение сходимости методов")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(7,7))

X_plot = X * 6

for i in range(len(X_plot)):
    if y[i] == 0:
        plt.scatter(X_plot[i][0], X_plot[i][1],
                    color='blue', label='Класс 0' if i == 0 else "")
    else:
        plt.scatter(X_plot[i][0], X_plot[i][1],
                    color='red', label='Класс 1')

x_vals = np.linspace(-8, 8, 200)
x_vals_norm = x_vals / 6

y_line1 = -(w_mse_fixed[0]*x_vals_norm + w_mse_fixed[2]) / w_mse_fixed[1]
y_line1 = y_line1 * 6
plt.plot(x_vals, y_line1, '--', label="Линия MSE")

y_line2 = -(w_bce_adapt[0]*x_vals_norm + w_bce_adapt[2]) / w_bce_adapt[1]
y_line2 = y_line2 * 6
plt.plot(x_vals, y_line2, label="Линия BCE адаптивный")

x1 = float(input("Введите x1: "))
x2 = float(input("Введите x2: "))

user = np.array([x1/6, x2/6, 1])
prob = sigmoid(np.dot(user, w_bce_adapt))
cls = 1 if prob >= 0.5 else 0

print("Вероятность =", prob)
print("Класс =", cls)

plt.scatter(x1, x2, color='green', marker='x',
            s=120, label="Ваша точка")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Классификация (логистический нейрон)")
plt.legend()
plt.grid()
plt.show()