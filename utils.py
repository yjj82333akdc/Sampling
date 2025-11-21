# python
import matplotlib.pyplot as plt
import numpy as np

#for comparison of two samples
def energy_distance(X, Y):
    """
    X: array (n, d)
    Y: array (m, d)
    returns: energy distance (scalar)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    n, m = X.shape[0], Y.shape[0]

    # pairwise distances
    d_xy = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)  # (n, m)
    d_xx = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)  # (n, n)
    d_yy = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)  # (m, m)

    # unbiased estimates (ignore diagonal for xx, yy)
    term_xy = 2.0 * d_xy.mean()
    if n > 1:
        term_xx = d_xx[np.triu_indices(n, k=1)].mean()
    else:
        term_xx = 0.0
    if m > 1:
        term_yy = d_yy[np.triu_indices(m, k=1)].mean()
    else:
        term_yy = 0.0

    ed2 = term_xy - term_xx - term_yy  # squared energy distance
    return abs(ed2)



def compare_mean_var(X_train, samples, name_train="X_train", name_samples="samples"):
    """
    Compare per-dimension mean and variance between X_train and samples.

    Returns
    -------
    mean_train, mean_samples, var_train, var_samples : np.ndarray
        Per-dimension means and variances.
    ss_mean_diff : float
        Sum of squares of per-dimension mean differences:
            sum_j (mean_samples[j] - mean_train[j])^2
    ss_var_diff : float
        Sum of squares of per-dimension variance differences:
            sum_j (var_samples[j] - var_train[j])^2
    ss_total : float
        ss_mean_diff + ss_var_diff
    """
    X_train = np.asarray(X_train)
    samples  = np.asarray(samples)

    if X_train.ndim != 2 or samples.ndim != 2:
        raise ValueError("X_train and samples must be 2D arrays.")
    if X_train.shape[1] != samples.shape[1]:
        raise ValueError(
            f"Dim mismatch: X_train has dim={X_train.shape[1]}, "
            f"samples has dim={samples.shape[1]}."
        )

    d = X_train.shape[1]

    mean_train    = X_train.mean(axis=0)
    mean_samples  = samples.mean(axis=0)
    var_train     = X_train.var(axis=0)
    var_samples   = samples.var(axis=0)

    diff_mean = mean_samples - mean_train
    diff_var  = var_samples - var_train

    ss_mean_diff = float(np.sum(diff_mean ** 2))
    ss_var_diff  = float(np.sum(diff_var ** 2))
    ss_total     = ss_mean_diff + ss_var_diff

    print("\nPer-dimension mean / variance comparison")
    header = (
        f"{'dim':>3} | {name_train}_mean | {name_samples}_mean | diff_mean "
        f"| {name_train}_var | {name_samples}_var | diff_var"
    )
    print(header)

    for j in range(d):
        print(
            f"{j:3d} | "
            f"{mean_train[j]: .6f} | {mean_samples[j]: .6f} | {diff_mean[j]: .6f} | "
            f"{var_train[j]: .6f} | {var_samples[j]: .6f} | {diff_var[j]: .6f}"
        )

    print("\nSum of squares:")
    print(f"  mean diff SS = {ss_mean_diff:.6e}")
    print(f"  var  diff SS = {ss_var_diff:.6e}")
    print(f"  total SS     = {ss_total:.6e}")

    return mean_train, mean_samples, var_train, var_samples, ss_mean_diff, ss_var_diff, ss_total


def metric(X_train, samples, dim, N_samples, elapsed):
    """
    Compute diagnostics and plots for a batch of samples.

    Parameters
    ----------
    X_train : ndarray, shape (N_train, dim)
    samples : ndarray, shape (N_samples, dim)
    dim : int
        Dimension.
    N_samples : int
        Number of generated samples.
    elapsed : float
        Wall-clock time (seconds) used to generate `samples`.

    Returns
    -------
    metrics : dict
        Contains ED2, mean/var stats, SS differences, and elapsed time.
    """
    # accuracy evaluation
    ED2 = energy_distance(X_train, samples)
    print("Energy distance^2:", ED2)

    (mean_train, mean_samples,
     var_train, var_samples,
     ss_mean_diff, ss_var_diff, ss_total) = compare_mean_var(X_train, samples)

    print("SS(mean diff) / dim =", ss_mean_diff / dim)
    print("SS(var  diff) / dim =", ss_var_diff / dim)
    print("SS(total) / dim     =", ss_total / dim)

    # timing
    print(f"Sampling {N_samples} points in dim={dim} took {elapsed:.4f} seconds")
    print(f"sampling time per point ≈ {elapsed / N_samples:.6e} seconds")

    # plots
    if dim == 2:
        plot_2d_samples(X_train, samples)
    if dim == 3:
        plot_3d_samples(X_train, samples)

    return {
        "ED2": ED2,
        "mean_train": mean_train,
        "mean_samples": mean_samples,
        "var_train": var_train,
        "var_samples": var_samples,
        "ss_mean_diff": ss_mean_diff,
        "ss_var_diff": ss_var_diff,
        "ss_total": ss_total,
        "elapsed": elapsed,
    }

#for plotting samples when d=2 or 3
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




