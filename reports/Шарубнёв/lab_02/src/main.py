import numpy as np
import matplotlib.pyplot as plt

class DenseLayer:
    def __init__(self):
        self.w = None
        self.b = None
        self.z = None

    def forward(self, x):
        if self.w is None:
            self.w = np.random.randn(x.shape[1], 1)
            self.b = np.zeros((1,))
        self.z = x @ self.w + self.b
        return (self.z >= 0).astype(float)

    def linear(self, x):
        if self.w is None:
            self.w = np.random.randn(x.shape[1], 1)
            self.b = np.zeros((1,))
        self.z = x @ self.w + self.b
        return self.z


def train(X, y, epochs=100, alpha=0.01, adaptive=False, Ee=1e-3):
    n = X.shape[0]
    layer = DenseLayer()
    history = []

    for epoch in range(epochs):
        Es = 0.0

        for i in range(n):
            x_i = X[i:i+1]
            y_i = y[i:i+1]

            y_lin = layer.linear(x_i)
            y_pred = (y_lin >= 0).astype(float)

            e = (y_pred - y_i).item()

            if adaptive:
                alpha_t = 1.0 / (1 + np.sum(x_i**2))
            else:
                alpha_t = alpha

            layer.w -= alpha_t * e * x_i.T
            layer.b -= alpha_t * e

            Es += 0.5 * (y_lin - y_i).item()**2

        history.append(Es)

        if Es <= Ee:
            break

    return layer, history

X = np.array([[6, 1], [-6, 1], [6, -1], [-6, -1]], dtype=float)
y = np.array([[0], [0], [0], [1]], dtype=float)

layer_fixed, hist_fixed = train(X, y, alpha=0.01, adaptive=False)
layer_adapt, hist_adapt = train(X, y, adaptive=True)

print("Эпох фиксированный шаг:", len(hist_fixed))
print("Эпох адаптивный шаг:", len(hist_adapt))


plt.plot(hist_fixed, label="Фиксированный шаг")
plt.plot(hist_adapt, label="Адаптивный шаг")
plt.xlabel("Эпоха")
plt.ylabel("Es(p)")
plt.grid()
plt.legend()
plt.show()

def plot_boundary(layer):
    plt.figure(figsize=(6,6))

    for i in range(len(X)):
        color = "red" if y[i] == 1 else "blue"
        plt.scatter(X[i,0], X[i,1], color=color, s=80)

    xs = np.linspace(-6, 6, 200)
    if abs(layer.w[1]) < 1e-6:
        x0 = -layer.b / layer.w[0]
        plt.axvline(x0, color="k")
    else:
        ys = -(layer.w[0]*xs + layer.b) / layer.w[1]
        plt.plot(xs, ys, "k--")

    plt.grid()
    plt.title("Разделяющая линия (адаптивный шаг)")
    plt.show()

plot_boundary(layer_adapt)

def classify_point(x1, x2, layer):
    x = np.array([[x1, x2]])
    y_lin = layer.linear(x)
    return int(y_lin.item() >= 0)

x1 = float(input("Введите x1: "))
x2 = float(input("Введите x2: "))

cls = classify_point(x1, x2, layer_adapt)
print("Класс =", cls)
