import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, n_inputs):

        self.w = np.random.uniform(-0.1, 0.1, n_inputs + 1)

    def _add_bias(self, X):
        X = np.atleast_2d(X)
        bias = np.ones((X.shape[0], 1))
        return np.concatenate((bias, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -15, 15)))

    def predict_proba(self, X):
        Xb = self._add_bias(X)
        return self.sigmoid(Xb @ self.w)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def compute_loss(self, y_true, y_pred, mode='BCE'):
        if mode == 'BCE':

            eps = 1e-9
            return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
        else:
            return 0.5 * np.mean((y_true - y_pred) ** 2)

    def train(self, X_train, y_train, alpha=0.1, adaptive=False, loss_mode='BCE', max_epochs=2000, Ee=0.01):
        Xb_train = self._add_bias(X_train)
        history = []

        for epoch in range(max_epochs):

            for xi, target in zip(Xb_train, y_train):
                y_hat = self.sigmoid(np.dot(xi, self.w))

                if loss_mode == 'BCE':

                    gradient = (y_hat - target) * xi
                else:

                    gradient = (y_hat - target) * (y_hat * (1 - y_hat)) * xi

                if adaptive:
                    curr_alpha = 1.0 / (np.dot(xi, xi) + 1e-6)
                else:
                    curr_alpha = alpha

                self.w -= curr_alpha * gradient

            all_preds = self.sigmoid(Xb_train @ self.w)
            current_error = self.compute_loss(y_train, all_preds, mode=loss_mode)
            history.append(current_error)

            if current_error <= Ee:
                break

        return history, epoch + 1


n = 5
X_all = np.array([[int(b) for b in format(i, f'0{n}b')] for i in range(2 ** n)])
# Логическая функция (например, NOR: истина только если все нули)
y_all = (np.sum(X_all, axis=1) == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

configs = [
    {'name': 'MSE + Fix', 'loss': 'MSE', 'adapt': False, 'alpha': 0.5, 'color': 'r'},
    {'name': 'MSE + Adapt', 'loss': 'MSE', 'adapt': True, 'alpha': None, 'color': 'orange'},
    {'name': 'BCE + Fix', 'loss': 'BCE', 'adapt': False, 'alpha': 0.5, 'color': 'g'},
    {'name': 'BCE + Adapt', 'loss': 'BCE', 'adapt': True, 'alpha': None, 'color': 'b'},
]

plt.figure(figsize=(10, 6))
results_table = []

for cfg in configs:
    model = Perceptron(n)
    history, epochs = model.train(X_train, y_train, alpha=cfg['alpha'],
                                  adaptive=cfg['adapt'], loss_mode=cfg['loss'], Ee=0.01)

    train_acc = np.mean(model.predict(X_train) == y_train)
    test_acc = np.mean(model.predict(X_test) == y_test)
    full_table_acc = np.mean(model.predict(X_all) == y_all)

    results_table.append([cfg['name'], epochs, f"{train_acc:.2%}", f"{test_acc:.2%}", f"{full_table_acc:.2%}"])
    plt.plot(history, label=cfg['name'], color=cfg['color'])

plt.title('Сравнение сходимости: MSE vs BCE')
plt.xlabel('Эпохи')
plt.ylabel('Значение функции потерь (Es)')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

print(f"{'Конфигурация':<15} | {'Эпохи':<6} | {'Acc Train':<10} | {'Acc Test':<10} | {'Full Table'}")
print("-" * 65)
for row in results_table:
    print(f"{row[0]:<15} | {row[1]:<6} | {row[2]:<10} | {row[3]:<10} | {row[4]}")

print("\n--- Режим функционирования (BCE + Adapt) ---")
test_vec = np.zeros(n)
prob = model.predict_proba(test_vec)[0]
pred_class = 1 if prob > 0.5 else 0
true_val = y_all[0]

print(f"Вход: {test_vec}")
print(f"Вероятность: {prob:.4f}")
print(f"Класс: {pred_class}")
print("Результат:", "Совпадает с таблицей" if pred_class == true_val else "Расхождение")