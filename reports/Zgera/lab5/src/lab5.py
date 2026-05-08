import numpy as np
import matplotlib.pyplot as plt
import itertools

n_bits = 6
err_limit = 0.01
base_lr = 0.1

np.random.seed(42)

all_inputs = np.array(list(itertools.product([0, 1], repeat=n_bits)))
all_targets = np.array([0 if np.all(v == 0) else 1 for v in all_inputs])

zero_idx = np.where(all_targets == 0)[0]
one_idx = np.where(all_targets == 1)[0]

np.random.shuffle(zero_idx)
np.random.shuffle(one_idx)

train_zero = 1
train_one = int(0.8 * len(all_inputs)) - 1

train_idx = np.concatenate((zero_idx[:train_zero], one_idx[:train_one]))
test_idx = one_idx[train_one:]

np.random.shuffle(train_idx)
np.random.shuffle(test_idx)

X_train = all_inputs[train_idx]
y_train = all_targets[train_idx]

X_test = all_inputs[test_idx]
y_test = all_targets[test_idx]

X_train_b = np.c_[np.ones(len(X_train)), X_train]
X_test_b = np.c_[np.ones(len(X_test)), X_test]
X_full_b = np.c_[np.ones(len(all_inputs)), all_inputs]


def act_sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -100, 100)))


def run_training(data_x, data_y, loss_type, lr_type):
    np.random.seed(42)
    w = np.random.uniform(-0.1, 0.1, n_bits + 1)
    epoch_errors = []

    for ep in range(10000):
        total_err = 0

        for i in range(len(data_x)):
            x = data_x[i]
            t = data_y[i]
            y = act_sigmoid(np.dot(w, x))

            lr = base_lr if lr_type == "fixed" else 1.0 / (1.0 + np.sum(x * x))

            if loss_type == "MSE":
                total_err += 0.5 * (t - y) ** 2
                grad = (t - y) * y * (1 - y)
            else:
                yp = np.clip(y, 1e-15, 1 - 1e-15)
                total_err += -(t * np.log(yp) + (1 - t) * np.log(1 - yp))
                grad = (t - y)

            w += lr * grad * x

        epoch_errors.append(total_err)
        if total_err <= err_limit:
            break

    return w, epoch_errors, ep + 1


modes = {
    "MSE-Fixed": ("MSE", "fixed"),
    "MSE-Adapt": ("MSE", "adaptive"),
    "BCE-Fixed": ("BCE", "fixed"),
    "BCE-Adapt": ("BCE", "adaptive")
}

results = {}

for name, (lf, lr) in modes.items():
    results[name] = run_training(X_train_b, y_train, lf, lr)
    print(f"{name} done. Epochs: {results[name][2]}")


plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red', 'purple']
styles = ['-', '--', '-.', ':']

for (name, (w, hist, ep)), c, s in zip(results.items(), colors, styles):
    plt.plot(hist, label=f"{name} ({ep} ep)", color=c, linestyle=s)

plt.axhline(err_limit, color='orange', linestyle='--', label=f"Ee = {err_limit}")
plt.yscale('log')
plt.title("Convergence curves")
plt.xlabel("Epoch")
plt.ylabel("Total error (log)")
plt.grid(True)
plt.legend()
plt.show()


def accuracy(weights, X, y):
    preds = (act_sigmoid(np.dot(X, weights)) >= 0.5).astype(int)
    return np.mean(preds == y) * 100


print("\nAccuracy results:")
for name, (w, hist, ep) in results.items():
    acc_tr = accuracy(w, X_train_b, y_train)
    acc_te = accuracy(w, X_test_b, y_test)
    acc_full = accuracy(w, X_full_b, all_targets)
    print(f"{name:<12} | Ep: {ep:<5} | Train: {acc_tr:>5.1f}% | Test: {acc_te:>5.1f}% | Full: {acc_full:>5.1f}%")


print("\nInteractive mode:")
final_w = results["BCE-Adapt"][0]

try:
    raw = input("Enter 6 bits separated by spaces: ")
    vec = np.array([int(v) for v in raw.split()])
    vec_b = np.insert(vec, 0, 1)

    prob = act_sigmoid(np.dot(final_w, vec_b))
    cls = 1 if prob >= 0.5 else 0
    true_cls = 0 if np.all(vec == 0) else 1

    print(f"ŷ = {prob:.4f}")
    print(f"Predicted: {cls}")
    print("Match" if cls == true_cls else "Mismatch")

except Exception:
    pass
