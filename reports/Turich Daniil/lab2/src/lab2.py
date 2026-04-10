import numpy as np
import matplotlib.pyplot as plt
X = np.array([
    [4, 6],
    [-4, 6],
    [4, -6],
    [-4, -6]
])
y = np.array([0, 1, 1, 1]).reshape(-1, 1)
def net_output(w, x, b):
    return x @ w - b
def train_constant(X, y, lr=0.01, error_limit=0.5, max_epochs=10000):
    n, m = X.shape
    w = np.zeros((m, 1))
    b = 0.0
    errors = []
    for epoch in range(max_epochs):
        for i in range(n):
            x = X[i:i+1]
            target = y[i]
            out = net_output(w, x, b)
            delta = out - target
            w -= lr * x.T * delta
            b += lr * delta
        pred = net_output(w, X, b)
        E = np.sum((pred - y) ** 2)
        errors.append(E)
        if E <= error_limit:
            break
    return w, b, errors, epoch + 1
def train_adaptive(X, y, error_limit=0.5, max_epochs=10000):
    n, m = X.shape
    w = np.zeros((m, 1))
    b = 0.0
    errors = []
    for epoch in range(max_epochs):
        for i in range(n):
            x = X[i:i+1]
            target = y[i]
            norm = np.sum(x ** 2)
            alpha = 1 / norm
            out = net_output(w, x, b)
            delta = out - target
            w -= alpha * x.T * delta
            b += alpha * delta
        pred = net_output(w, X, b)
        E = np.sum((pred - y) ** 2)
        errors.append(E)
        if E <= error_limit:
            break
    return w, b, errors, epoch + 1
w_const, b_const, err_const, ep_const = train_constant(X, y)
w_adapt, b_adapt, err_adapt, ep_adapt = train_adaptive(X, y)
print("Эпох (фиксированный шаг):", ep_const)
print("Эпох (адаптивный шаг):", ep_adapt)
print("Ускорение:", ep_const / ep_adapt)
plt.figure(figsize=(7,5))
plt.plot(err_const, label=f"Фиксированный шаг ({ep_const} эпох)")
plt.plot(err_adapt, label=f"Адаптивный шаг ({ep_adapt} эпох)")
plt.yscale("log")
plt.xlabel("Эпоха")
plt.ylabel("Суммарная ошибка")
plt.title("График обучения")
plt.legend()
plt.grid()
plt.show()
mask0 = y[:,0] == 0
mask1 = y[:,0] == 1
plt.figure(figsize=(6,6))
plt.scatter(X[mask0][:,0], X[mask0][:,1], s=150, label="Class 0")
plt.scatter(X[mask1][:,0], X[mask1][:,1], s=150, label="Class 1")
w1, w2 = w_adapt.flatten()
x_line = np.linspace(-8, 8, 200)
if abs(w2) > 1e-9:
    y_line = ((b_adapt + 0.5 - w1 * x_line) / w2).flatten()
    plt.plot(x_line, y_line, label="Разделяющая линия")
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Разделяющая линия (адаптивный метод)")
plt.legend()
plt.grid()
plt.show()
print("\nРежим работы сети (адаптивная модель)")
print("Введите координаты: x1 x2   или 'exit'")
while True:
    s = input("Ввод: ")
    if s == "exit":
        break
    try:
        x1, x2 = map(float, s.split())
        x = np.array([[x1, x2]])
        score = net_output(w_adapt, x, b_adapt)[0][0]
        cls = 1 if score >= 0.5 else 0
        print("Выход сети:", score)
        print("Класс:", cls)
    except:
        print("Ошибка ввода")