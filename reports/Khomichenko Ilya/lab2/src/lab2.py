import numpy as np
import matplotlib.pyplot as plt

points_data = np.array([
    [5, 6],
    [-5, 6],
    [5, -6],
    [-5, -6]
])

target_data = np.array([0, 1, 1, 1]).reshape(-1, 1)


def model_output(weights, inputs, bias):
    return inputs @ weights - bias


def train_constant(dataset, labels, lr=0.01, threshold=0.5, epochs_limit=20000):
    sample_count, feature_count = dataset.shape

    weights_vec = np.zeros((feature_count, 1))
    bias_term = 0.0

    error_track = []
    final_epoch = epochs_limit

    print(f"[Constant LR] Started | lr={lr} | target error={threshold}")

    for epoch in range(epochs_limit):

        shuffled_idx = np.random.permutation(sample_count)

        for i in shuffled_idx:

            sample = dataset[i:i+1]
            label = float(labels[i, 0])

            raw = model_output(weights_vec, sample, bias_term)
            delta = float(raw[0, 0]) - label

            weights_vec -= lr * sample.T * delta
            bias_term += lr * delta

        predictions = model_output(weights_vec, dataset, bias_term)

        total_error = float(np.sum((predictions - labels) ** 2))
        error_track.append(total_error)

        if epoch % 40 == 0 and epoch > 0:
            print(f"Epoch {epoch:5d} | Error = {total_error:.6f}")

        if total_error <= threshold:
            final_epoch = epoch + 1
            print(f"Training stopped at epoch {final_epoch} | Error={total_error:.6f}\n")
            break
    else:
        print(f"Reached max epochs ({epochs_limit}) | Error={error_track[-1]:.6f}\n")

    return weights_vec, bias_term, error_track, final_epoch


def train_adaptive(dataset, labels, threshold=0.5, epochs_limit=20000):

    sample_count, feature_count = dataset.shape

    weights_vec = np.zeros((feature_count, 1))
    bias_term = 0.0

    error_track = []
    final_epoch = epochs_limit

    print(f"[Adaptive LR] Started | target error={threshold}")

    for epoch in range(epochs_limit):

        shuffled_idx = np.random.permutation(sample_count)

        for i in shuffled_idx:

            sample = dataset[i:i+1]
            label = float(labels[i, 0])

            norm_value = float(np.sum(sample ** 2))
            dynamic_lr = 1.0 / norm_value if norm_value > 1e-12 else 0.0

            raw = model_output(weights_vec, sample, bias_term)
            delta = float(raw[0, 0]) - label

            weights_vec -= dynamic_lr * sample.T * delta
            bias_term += dynamic_lr * delta

        predictions = model_output(weights_vec, dataset, bias_term)

        total_error = float(np.sum((predictions - labels) ** 2))
        error_track.append(total_error)

        if epoch % 8 == 0 and epoch > 0:
            print(f"Epoch {epoch:5d} | Error = {total_error:.6f}")

        if total_error <= threshold:
            final_epoch = epoch + 1
            print(f"Training stopped at epoch {final_epoch} | Error={total_error:.6f}\n")
            break
    else:
        print(f"Reached max epochs ({epochs_limit}) | Error={error_track[-1]:.6f}\n")

    return weights_vec, bias_term, error_track, final_epoch


np.random.seed(42)

error_target = 0.5

w_const, b_const, err_const, ep_const = train_constant(
    points_data, target_data, lr=0.01, threshold=error_target
)

np.random.seed(42)

w_adapt, b_adapt, err_adapt, ep_adapt = train_adaptive(
    points_data, target_data, threshold=error_target
)

print(f"Speed ratio: {ep_const} / {ep_adapt} = {ep_const / ep_adapt:.2f}")


figure, (plot_loss, plot_space) = plt.subplots(1, 2, figsize=(14, 6))

plot_loss.plot(
    np.arange(1, len(err_const) + 1),
    err_const,
    linewidth=2,
    label=f"Constant LR (epochs: {ep_const})"
)

plot_loss.plot(
    np.arange(1, len(err_adapt) + 1),
    err_adapt,
    linewidth=2,
    label=f"Adaptive LR (epochs: {ep_adapt})"
)

plot_loss.axhline(error_target, linestyle="--", linewidth=1.3)

plot_loss.set_xlabel("Epoch")
plot_loss.set_ylabel("Squared Error")
plot_loss.set_yscale("log")
plot_loss.set_title("Learning Curve")
plot_loss.legend()
plot_loss.grid(alpha=0.3)


mask_zero = target_data[:, 0] == 0
mask_one = target_data[:, 0] == 1

plot_space.scatter(
    points_data[mask_zero][:, 0],
    points_data[mask_zero][:, 1],
    s=200,
    label="Class 0"
)

plot_space.scatter(
    points_data[mask_one][:, 0],
    points_data[mask_one][:, 1],
    s=200,
    label="Class 1"
)


weight1, weight2 = w_adapt.flatten()
x_line = np.linspace(-9, 9, 500)

if abs(weight2) > 1e-9:
    y_line = (b_adapt - weight1 * x_line) / weight2
    plot_space.plot(
        x_line,
        y_line,
        linewidth=2.5,
        label="Decision Boundary"
    )

plot_space.set_xlim(-9, 9)
plot_space.set_ylim(-9, 9)
plot_space.set_xlabel("Feature 1")
plot_space.set_ylabel("Feature 2")
plot_space.set_title("Classification Result (Adaptive)")
plot_space.legend()
plot_space.grid(alpha=0.3)

plt.tight_layout()
plt.show()


history_points = []
history_classes = []

print("-" * 40)
print("Prediction mode (Adaptive model)")
print("Enter coordinates: x1 x2   or type 'exit'\n")

while True:

    user_entry = input("Input: ").strip()

    if user_entry in ("exit", ""):
        break

    try:

        val1, val2 = map(float, user_entry.split())

        sample = np.array([[val1, val2]])

        result_score = model_output(w_adapt, sample, b_adapt).item()

        predicted_class = 1 if result_score >= 0 else 0

        print(f"Score: {result_score:+.6f}")
        print(f"Predicted class: {predicted_class}\n")

        history_points.append([val1, val2])
        history_classes.append(predicted_class)

        fig2, axis2 = plt.subplots(figsize=(8, 8))

        axis2.scatter(
            points_data[mask_zero][:, 0],
            points_data[mask_zero][:, 1],
            s=200,
            label="Class 0"
        )

        axis2.scatter(
            points_data[mask_one][:, 0],
            points_data[mask_one][:, 1],
            s=200,
            label="Class 1"
        )

        if abs(weight2) > 1e-9:
            y_dynamic = (b_adapt - weight1 * x_line) / weight2
            axis2.plot(
                x_line,
                y_dynamic,
                linewidth=2.5,
                label="Boundary"
            )

        for pt, cls in zip(history_points, history_classes):

            axis2.scatter(
                pt[0],
                pt[1],
                marker="X",
                s=260,
                zorder=6
            )

        axis2.set_xlim(-9, 9)
        axis2.set_ylim(-9, 9)

        axis2.set_xlabel("Feature 1")
        axis2.set_ylabel("Feature 2")

        axis2.set_title(
            f"Last input: ({val1}, {val2}) → class {predicted_class}"
        )

        axis2.legend()
        axis2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    except Exception:
        print("Invalid input format. Use: number number\n")

print("Program finished.")