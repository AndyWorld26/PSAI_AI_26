import numpy as np
import matplotlib.pyplot as plt

x = np.array([
    [ 5,  6],
    [-5,  6],
    [ 5, -6],
    [-5, -6]
])

etalon_value = np.array([0, 1, 1, 1]).reshape(-1, 1).astype(float)

def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

def linear_output(W, X, T):
    return X @ W - T

def bce_loss(y_pred, y_true):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def mse_loss(y_pred, y_true):
    return np.sum((y_pred - y_true) ** 2)

def bce_fixed_fit(X, y, alpha=0.01, Ee=0.01, max_epochs=20000):
    N, n = X.shape
    W = np.zeros((n, 1))
    T = 0.0
    Es_hist = []
    stopped_epoch = max_epochs

    print(f"(BCE + фиксированный) alpha={alpha}, Ee={Ee}")

    for ep in range(max_epochs):
        order = np.random.permutation(N)
        for idx in order:
            xi = X[idx:idx+1]
            ti = float(y[idx, 0])

            s   = float(linear_output(W, xi, T)[0, 0])
            yi  = sigmoid(s)          
            d   = yi - ti             

            W  -= alpha * xi.T * d
            T  += alpha * d

        preds = sigmoid(linear_output(W, X, T))
        Es = bce_loss(preds, y)
        Es_hist.append(Es)

        if ep % 200 == 0 and ep > 0:
            print(f"  эпоха {ep:5d} | Es = {Es:.6f}")

        if Es <= Ee:
            stopped_epoch = ep + 1
            print(f"  Остановка на эпохе {stopped_epoch}, Es={Es:.6f}\n")
            break
    else:
        print(f"  Лимит {max_epochs} эпох, Es={Es_hist[-1]:.6f}\n")

    return W, T, Es_hist, stopped_epoch

def bce_adaptive_fit(X, y, Ee=0.01, max_epochs=20000):
    N, n = X.shape
    W = np.zeros((n, 1))
    T = 0.0
    Es_hist = []
    stopped_epoch = max_epochs

    print(f"(BCE + адаптивный) α(t), Ee={Ee}")

    for ep in range(max_epochs):
        order = np.random.permutation(N)
        for idx in order:
            xi      = X[idx:idx+1]
            ti      = float(y[idx, 0])
            norm_sq = float(np.sum(xi ** 2))
            alpha_t = 1.0 / norm_sq if norm_sq > 1e-12 else 0.0

            s   = float(linear_output(W, xi, T)[0, 0])
            yi  = sigmoid(s)
            d   = yi - ti             

            W  -= alpha_t * xi.T * d
            T  += alpha_t * d

        preds = sigmoid(linear_output(W, X, T))
        Es = bce_loss(preds, y)
        Es_hist.append(Es)

        if ep % 50 == 0 and ep > 0:
            print(f"  эпоха {ep:5d} | Es = {Es:.6f}")

        if Es <= Ee:
            stopped_epoch = ep + 1
            print(f"  Остановка на эпохе {stopped_epoch}, Es={Es:.6f}\n")
            break
    else:
        print(f"  Лимит {max_epochs} эпох, Es={Es_hist[-1]:.6f}\n")

    return W, T, Es_hist, stopped_epoch

def mse_fixed_fit(X, y, alpha=0.01, Ee=0.5, max_epochs=20000):
    N, n = X.shape
    W = np.zeros((n, 1))
    T = 0.0
    Es_hist = []
    stopped_epoch = max_epochs
    for ep in range(max_epochs):
        order = np.random.permutation(N)
        for idx in order:
            xi = X[idx:idx+1]; ti = float(y[idx, 0])
            s  = float(linear_output(W, xi, T)[0, 0])
            d  = s - ti
            W -= alpha * xi.T * d; T += alpha * d
        Es = mse_loss(linear_output(W, X, T), y)
        Es_hist.append(Es)
        if Es <= Ee:
            stopped_epoch = ep + 1; break
    return W, T, Es_hist, stopped_epoch

def mse_adaptive_fit(X, y, Ee=0.5, max_epochs=20000):
    N, n = X.shape
    W = np.zeros((n, 1))
    T = 0.0
    Es_hist = []
    stopped_epoch = max_epochs
    for ep in range(max_epochs):
        order = np.random.permutation(N)
        for idx in order:
            xi = X[idx:idx+1]; ti = float(y[idx, 0])
            norm_sq = float(np.sum(xi ** 2))
            alpha_t = 1.0 / norm_sq if norm_sq > 1e-12 else 0.0
            s  = float(linear_output(W, xi, T)[0, 0])
            d  = s - ti
            W -= alpha_t * xi.T * d; T += alpha_t * d
        Es = mse_loss(linear_output(W, X, T), y)
        Es_hist.append(Es)
        if Es <= Ee:
            stopped_epoch = ep + 1; break
    return W, T, Es_hist, stopped_epoch

np.random.seed(42)
Ee_MSE = 0.5
Ee_BCE = 0.01

W_mf,  T_mf,  es_mf,  ep_mf  = mse_fixed_fit (x, etalon_value, alpha=0.01, Ee=Ee_MSE)
np.random.seed(42)
W_ma,  T_ma,  es_ma,  ep_ma  = mse_adaptive_fit (x, etalon_value, Ee=Ee_MSE)
np.random.seed(42)
W_bf,  T_bf,  es_bf,  ep_bf  = bce_fixed_fit (x, etalon_value, alpha=0.01, Ee=Ee_BCE)
np.random.seed(42)
W_ba,  T_ba,  es_ba,  ep_ba  = bce_adaptive_fit (x, etalon_value, Ee=Ee_BCE)

print("\n Итог: число эпох до Es ≤ Ee ")
print(f"  MSE + фиксированный : {ep_mf}")
print(f"  MSE + адаптивный    : {ep_ma}")
print(f"  BCE + фиксированный : {ep_bf}")
print(f"  BCE + адаптивный    : {ep_ba}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax1 = axes[0]

ax1.plot(es_mf, lw=2, color="#2563EB",
         label=f"MSE + фиксированный α=0.01  (эпох: {ep_mf})")
ax1.plot(es_ma, lw=2, color="#60A5FA",
         label=f"MSE + адаптивный α(t)        (эпох: {ep_ma})")
ax1.plot(es_bf, lw=2, color="#DC2626",
         label=f"BCE + фиксированный α=0.01  (эпох: {ep_bf})")
ax1.plot(es_ba, lw=2, color="#F97316",
         label=f"BCE + адаптивный α(t)        (эпох: {ep_ba})")
ax1.axhline(Ee_BCE, color="gray", lw=1.4, ls="--", label=f"Eₑ(BCE) = {Ee_BCE}")
ax1.axhline(Ee_MSE,  color="#3B82F6", lw=1.2, ls=":", label=f"Eₑ(MSE) = {Ee_MSE}")
ax1.set_yscale("log")
ax1.set_xlabel("Эпоха (p)", fontsize=12)
ax1.set_ylabel("Es(p)", fontsize=12)
ax1.set_title("Кривые сходимости: все 4 конфигурации", fontsize=13, fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(alpha=0.35)

ax2 = axes[1]

cls0 = etalon_value[:, 0] == 0
cls1 = etalon_value[:, 0] == 1

ax2.scatter(x[cls0][:, 0], x[cls0][:, 1], s=200, color="#3B82F6",
            edgecolors="black", lw=1.5, label="Класс 0", zorder=5)
ax2.scatter(x[cls1][:, 0], x[cls1][:, 1], s=200, color="#EF4444",
            edgecolors="black", lw=1.5, label="Класс 1", zorder=5)

for pt, lbl in zip(x, ["(5,6)", "(-5,6)", "(5,-6)", "(-5,-6)"]):
    ax2.annotate(lbl, xy=(pt[0], pt[1]), xytext=(pt[0]+0.3, pt[1]+0.6), fontsize=9)

xs_line = np.linspace(-9, 9, 500)

def draw_boundary(ax, W, T, color, label, threshold=0.0):
    w1, w2 = W.flatten()
    if abs(w2) > 1e-9:
        ys = (T + threshold - w1 * xs_line) / w2
        ax.plot(xs_line, ys, color=color, lw=2.5, label=label)

draw_boundary(ax2, W_mf, T_mf, "#2563EB", f"MSE+фикс  (Эпох: {ep_mf})", threshold=0.5)
draw_boundary(ax2, W_ma, T_ma, "#60A5FA", f"MSE+адап  (Эпох: {ep_ma})", threshold=0.5)
draw_boundary(ax2, W_bf, T_bf, "#DC2626", f"BCE+фикс  (Эпох: {ep_bf})", threshold=0.0)
draw_boundary(ax2, W_ba, T_ba, "#F97316", f"BCE+адап  (Эпох: {ep_ba})", threshold=0.0)

ax2.set_xlim(-9, 9); ax2.set_ylim(-9, 9)
ax2.set_xlabel("x_1", fontsize=12); ax2.set_ylabel("x_2", fontsize=12)
ax2.set_title("Разделяющие линии: все 4 конфигурации", fontsize=13, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
ax2.axhline(0, color='black', lw=0.5); ax2.axvline(0, color='black', lw=0.5)

plt.tight_layout()
plt.show()

added_pts, added_cls = [], []

print("─" * 50)
print(" Режим функционирования (BCE + адаптивный)")
print(" Введите координаты x_1 x_2, или 'exit'\n")

while True:
    user_input = input(" -> ").strip()
    if user_input in ("exit", ""):
        break
    try:
        x1_val, x2_val = map(float, user_input.split())
        point = np.array([[x1_val, x2_val]])

        s = float(linear_output(W_ba, point, T_ba)[0, 0])
        prob = sigmoid(s)
        cls  = 1 if prob >= 0.5 else 0   

        print(f"  S = {s:+.6f}")
        print(f"  ŷ = P(класс=1) = {prob:.6f}")
        print(f"  Класс = {cls} ({'красный' if cls else 'синий'})\n")

        added_pts.append([x1_val, x2_val])
        added_cls.append(cls)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x[cls0][:, 0], x[cls0][:, 1], s=200, color="#3B82F6",
                   edgecolors="black", label="Класс 0")
        ax.scatter(x[cls1][:, 0], x[cls1][:, 1], s=200, color="#EF4444",
                   edgecolors="black", label="Класс 1")

        draw_boundary(ax, W_ba, T_ba, "#F97316", "BCE+адап (граница)", threshold=0.0)
        draw_boundary(ax, W_ma, T_ma, "#60A5FA", "MSE+адап (для сравнения)", threshold=0.5)

        for p, c in zip(added_pts, added_cls):
            ax.scatter(p[0], p[1], marker="X", s=280,
                       color="#EF4444" if c else "#3B82F6",
                       edgecolors="black", zorder=6)

        ax.set_xlim(-9, 9); ax.set_ylim(-9, 9)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.set_title(f"Точка ({x1_val}, {x2_val}) -> ŷ={prob:.3f} -> класс {cls}")
        ax.legend(); ax.grid(alpha=0.3)
        ax.axhline(0, color='black', lw=0.5); ax.axvline(0, color='black', lw=0.5)
        plt.tight_layout(); plt.show()

    except Exception:
        print("  Ошибка ввода. Формат: x_1 x_2\n")

print("Работа завершена.")
