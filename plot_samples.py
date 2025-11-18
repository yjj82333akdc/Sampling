# python
import matplotlib.pyplot as plt

def plot_3d_samples(X_train, samples, title="3D Samples with all pairwise projections"):
    """
    Plot 3D scatter of (x0,x1,x2) and all three 2D projections:
    (x0,x1), (x0,x2), (x1,x2).
    """
    if X_train.shape[1] != 3 or samples.shape[1] != 3:
        raise ValueError("Both X_train and samples must have shape (N, 3) for dim=3.")

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(title, y=0.98, fontsize=12)

    # 3D scatter (top-left)
    ax3d = fig.add_subplot(221, projection="3d")
    ax3d.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2],
                 s=8, c="steelblue", alpha=0.35, label="X_train")
    ax3d.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
                 s=20, c="crimson", alpha=0.7, label="samples")
    ax3d.set_xlabel("x0")
    ax3d.set_ylabel("x1")
    ax3d.set_zlabel("x2")
    ax3d.set_title("3D Scatter")
    ax3d.legend(loc="upper left")

    # Pairwise projections
    pairs = [
        (0, 1, 222, "(x0, x1)"),
        (0, 2, 223, "(x0, x2)"),
        (1, 2, 224, "(x1, x2)"),
    ]
    for i, j, code, subtitle in pairs:
        ax = fig.add_subplot(code)
        ax.scatter(X_train[:, i], X_train[:, j], s=10, c="steelblue", alpha=0.35, label="X_train")
        ax.scatter(samples[:, i], samples[:, j], s=20, c="crimson", alpha=0.7, label="samples")
        ax.set_xlabel(f"x{i}")
        ax.set_ylabel(f"x{j}")
        ax.set_title(f"Projection {subtitle}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_2d_samples(X_train, samples, title: str = "2D Samples vs X_train"):
    """
    Overlay scatter plots for X_train and generated samples when dim == 2.
    """
    if X_train.shape[1] != 2 or samples.shape[1] != 2:
        raise ValueError("Both X_train and samples must have shape (N, 2) for dim=2.")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_train[:, 0], X_train[:, 1], s=10, c="steelblue", alpha=0.35, label="X_train")
    ax.scatter(samples[:, 0], samples[:, 1], s=20, c="crimson", alpha=0.7, label="samples")

    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.show()




