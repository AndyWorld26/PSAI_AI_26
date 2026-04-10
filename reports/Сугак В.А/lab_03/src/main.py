import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [4.0, 1.0],
    [-4.0, 1.0],
    [1.0, 4.0],
    [-1.0, -4.0]
])

Y = np.array([1, 0, 1, 0])

Xn = (X - X.mean(axis=0)) / X.std(axis=0)

def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def bce(y, y_pred):
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_mse(X, Y, lr0, adaptive=False):
    w = np.random.randn(2)
    b = 0
    errors = []
    k = 0.01

    for t in range(1, 2001):
        if adaptive:
            lr = lr0 / (1 + k * t)
        else:
            lr = lr0

        y_pred = X.dot(w) + b
        err = mse(Y, y_pred)
        errors.append(err)

        if err < 0.01:
            break

        dw = -2 * X.T.dot(Y - y_pred) / len(Y)
        db = -2 * np.mean(Y - y_pred)

        w -= lr * dw
        b -= lr * db

    return w, b, errors

def train_bce(X, Y, lr0, adaptive=False):
    w = np.random.randn(2)
    b = 0
    errors = []
    k = 0.01

    for t in range(1, 2001):
        if adaptive:
            lr = lr0 / (1 + k * t)
        else:
            lr = lr0

        z = X.dot(w) + b
        y_pred = sigmoid(z)
        err = bce(Y, y_pred)
        errors.append(err)

        if err < 0.01:
            break

        dw = X.T.dot(y_pred - Y) / len(Y)
        db = np.mean(y_pred - Y)

        w -= lr * dw
        b -= lr * db

    return w, b, errors

w1, b1, h1 = train_mse(Xn, Y, 0.01, False)
w2, b2, h2 = train_mse(Xn, Y, 0.01, True)
w3, b3, h3 = train_bce(Xn, Y, 0.1, False)
w4, b4, h4 = train_bce(Xn, Y, 0.1, True)

plt.figure()
plt.plot(h1, label="MSE фикс")
plt.plot(h2, label="MSE адапт")
plt.plot(h3, label="BCE фикс")
plt.plot(h4, label="BCE адапт")
plt.xlabel("Эпоха")
plt.ylabel("Ошибка")
plt.title("Сходимость моделей")
plt.legend()
plt.grid()
plt.show()

def plot_boundary(w, b, title):
    x_vals = np.linspace(-2, 2, 100)
    y_vals = -(w[0] * x_vals + b) / w[1]

    plt.plot(x_vals, y_vals, label=title)

plt.figure()
plt.scatter(Xn[:,0], Xn[:,1], c=Y)

plot_boundary(w3, b3, "BCE фикс")
plot_boundary(w4, b4, "BCE адапт")

plt.title("Разделяющие границы")
plt.legend()
plt.grid()
plt.show()

print("MSE фикс эпох:", len(h1))
print("MSE адапт эпох:", len(h2))
print("BCE фикс эпох:", len(h3))
print("BCE адапт эпох:", len(h4))