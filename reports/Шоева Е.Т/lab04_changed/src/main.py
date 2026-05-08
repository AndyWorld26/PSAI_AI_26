import numpy as np
import matplotlib.pyplot as plt
import itertools

N = 11
np.random.seed(42)

X_all = np.array(list(itertools.product([0, 1], repeat=N)), dtype=float)
y_all = np.any(X_all == 1, axis=1).astype(float)

print(f"Всего примеров: {len(X_all)}")
print(f"Класс 1: {int(y_all.sum())} | Класс 0: {int((1-y_all).sum())}\n")

idx_zero = np.where(y_all == 0)[0]
idx_one = np.where(y_all == 1)[0]


train_idx = np.concatenate([
    idx_zero,
    np.random.choice(idx_one, size=499, replace=False)])

test_idx = np.array([i for i in range(len(X_all)) if i not in train_idx])

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

print(f"Обучающая выборка: {len(X_train)} (класс 1: {int(y_train.sum())})")
print(f"Тестовая выборка: {len(X_test)} (класс 1: {int(y_test.sum())})\n")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

def total_error(X, y, w, b):
    yp = sigmoid(X @ w + b)
    return 0.5 * np.sum((y - yp) ** 2)

def accuracy(X, y, w, b):
    yp = sigmoid(X @ w + b)
    return np.mean((yp >= 0.5).astype(float) == y)

def train(X_tr, y_tr, X_te, y_te, mode='фиксированный', alpha=0.05, max_epochs=2000, tol=1e-4):
    rng = np.random.default_rng(1)
    w = rng.uniform(0.0, 0.5, size=N)
    b = -5.0

    train_err, test_err = [], []
    lr = alpha
    prev_err = float('inf')

    for epoch in range(max_epochs):
        for i in rng.permutation(len(X_tr)):
            xi, yi = X_tr[i], y_tr[i]
            out = sigmoid(xi @ w + b)
            delta = (yi - out) * out * (1 - out)
            w += lr * delta * xi
            b += lr * delta

        err_tr = total_error(X_tr, y_tr, w, b)
        err_te = total_error(X_te, y_te, w, b)
        train_err.append(err_tr)
        test_err.append(err_te)

        if mode == 'адаптивный':
            if err_tr < prev_err:
                lr = min(lr * 1.05, 1.0)
            else:
                lr = max(lr * 0.5, 1e-5)
        prev_err = err_tr

        if err_tr < tol:
            print(f"{mode.capitalize()} обучение завершено на эпохе {epoch+1}")
            break

    return w, b, train_err, test_err, epoch+1

print("\nОбучение: фиксированный шаг")
w_fix, b_fix, tr_fix, te_fix, ep_fix = train(
    X_train, y_train, X_test, y_test, mode='фиксированный', alpha=0.05
)

print("\nОбучение: адаптивный шаг")
w_adapt, b_adapt, tr_adapt, te_adapt, ep_adapt = train(
    X_train, y_train, X_test, y_test, mode='адаптивный', alpha=0.05
)

print(f"\nКоличество эпох:")
print(f"  (фиксированный) {ep_fix}")
print(f"  (адаптивный) {ep_adapt}\n")

print(f"Точность на обучении:")
print(f"  (фиксированный) {accuracy(X_train, y_train, w_fix, b_fix)*100:.2f}%")
print(f"  (адаптивный) {accuracy(X_train, y_train, w_adapt, b_adapt)*100:.2f}%\n")

print(f"Точность на тесте:")
print(f"  (фиксированный) {accuracy(X_test, y_test, w_fix, b_fix)*100:.2f}%")
print(f"  (адаптивный) {accuracy(X_test, y_test, w_adapt, b_adapt)*100:.2f}%\n")

print("Веса (фиксированный шаг):")
for i, wi in enumerate(w_fix):
    print(f"w{i+1} = {wi:.4f}")
print(f"b = {b_fix:.4f}\n")

print("Веса (адаптивный шаг):")
for i, wi in enumerate(w_adapt):
    print(f"w{i+1} = {wi:.4f}")
print(f"b = {b_adapt:.4f}\n")

plt.figure(figsize=(10, 6))

plt.plot(tr_fix, label="train(фиксированный)")
plt.plot(te_fix, label="test(фиксированный)")
plt.plot(tr_adapt, label="train(адаптивный)")
plt.plot(te_adapt, label="test(адаптивный)")

plt.xlabel("Эпоха")
plt.ylabel("Ошибка")
plt.title("Сходимость сети OR, 11")
plt.legend()
plt.grid(True)
plt.show()

def predict(vec, w, b):
    x = np.array(vec, dtype=float)
    prob = float(sigmoid(x @ w + b))
    return prob, int(prob >= 0.5)

print("\nПроверка")
print(f"Введите {N} чисел (0 или 1) через пробел, либо 'q' для выхода")

while True:
    s = input("Вход: ")
    if s.lower() == 'q':
        break

    parts = s.split()
    if len(parts) != N:
        print("Ошибка: требуется 11 значений\n")
        continue

    vec = list(map(int, parts))
    prob, cls = predict(vec, w_fix, b_fix)
    true_val = int(np.any(vec))

    print(f"Вероятность принадлежности к классу 1: {prob:.4f}")
    print(f"Предсказанный класс: {cls} | Истинный класс: {true_val}\n")