import numpy as np
import matplotlib.pyplot as plt


class BinaryClassifier:

    def __init__(self, features_count):
        self.w = np.random.uniform(-0.5, 0.5, features_count + 1)

    def _add_bias(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        bias = np.ones((X.shape[0], 1))
        return np.concatenate((bias, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        Xb = self._add_bias(X)
        return self.sigmoid(Xb @ self.w)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


    def train_mse(self, X, y, alpha=0.1, adaptive=False, Ee=0.01, max_epochs=500):
        Xb = self._add_bias(X)
        y = np.array(y, dtype=float)
        history = []

        for epoch in range(max_epochs):

            for xi, target in zip(Xb, y):
                output = np.dot(xi, self.w)
                error = target - output

                if adaptive:
                    alpha = 1.0 / np.dot(xi, xi)

                self.w += alpha * error * xi

            Es = 0.5 * np.mean((y - Xb @ self.w) ** 2)
            history.append(Es)

            if Es <= Ee:
                break

        return history

    # ===================== BCE =====================

    def train_bce(self, X, y, alpha=0.1, adaptive=False, Ee=0.01, max_epochs=500):
        Xb = self._add_bias(X)
        y = np.array(y, dtype=float)
        history = []

        for epoch in range(max_epochs):

            for xi, target in zip(Xb, y):
                y_hat = self.sigmoid(np.dot(xi, self.w))
                gradient = (y_hat - target) * xi

                if adaptive:
                    alpha = 1.0 / np.dot(xi, xi)

                self.w -= alpha * gradient

            # BCE ошибка
            y_hat_all = self.sigmoid(Xb @ self.w)
            eps = 1e-12
            Es = -np.mean(
                y * np.log(y_hat_all + eps) +
                (1 - y) * np.log(1 - y_hat_all + eps)
            )
            history.append(Es)

            if Es <= Ee:
                break

        return history


X_data = np.array([
    [3, 4],
    [-3, 4],
    [3, -4],
    [-3, -4]
])

y_data = np.array([1, 1, 0, 0])

Ee = 0.01


# ====================== ОБУЧЕНИЕ ======================

model_mse_fixed = BinaryClassifier(2)
h_mse_fixed = model_mse_fixed.train_mse(X_data, y_data, alpha=0.1)

model_mse_adapt = BinaryClassifier(2)
h_mse_adapt = model_mse_adapt.train_mse(X_data, y_data, adaptive=True)

model_bce_fixed = BinaryClassifier(2)
h_bce_fixed = model_bce_fixed.train_bce(X_data, y_data, alpha=0.1)

model_bce_adapt = BinaryClassifier(2)
h_bce_adapt = model_bce_adapt.train_bce(X_data, y_data, adaptive=True)


plt.figure(figsize=(8, 5))
plt.plot(h_mse_fixed, label="MSE + фиксированный")
plt.plot(h_mse_adapt, label="MSE + адаптивный")
plt.plot(h_bce_fixed, label="BCE + фиксированный")
plt.plot(h_bce_adapt, label="BCE + адаптивный")
plt.xlabel("Эпоха")
plt.ylabel("Ошибка Es(p)")
plt.legend()
plt.grid()
plt.show()


xmin, xmax = -6, 6
ymin, ymax = -6, 6

grid_x, grid_y = np.meshgrid(
    np.linspace(xmin, xmax, 250),
    np.linspace(ymin, ymax, 250)
)

grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
grid_pred = model_bce_adapt.predict(grid_points).reshape(grid_x.shape)

plt.figure(figsize=(6, 6))
plt.contourf(grid_x, grid_y, grid_pred, levels=[-0.1, 0.5, 1.1], alpha=0.3)
plt.contour(grid_x, grid_y, grid_pred, levels=[0.5])

colors = ['red' if label == 0 else 'blue' for label in y_data]
plt.scatter(X_data[:, 0], X_data[:, 1], c=colors, edgecolors='black')

w = model_bce_adapt.w
x_line = np.linspace(xmin, xmax, 200)
y_line = -(w[0] + w[1] * x_line) / w[2]
plt.plot(x_line, y_line, 'k')

plt.title("Разделяющая линия (BCE + адаптивный)")
plt.grid()
plt.show()


print("\nЭпохи:")
print("MSE фикс:", len(h_mse_fixed))
print("MSE адапт:", len(h_mse_adapt))
print("BCE фикс:", len(h_bce_fixed))
print("BCE адапт:", len(h_bce_adapt))