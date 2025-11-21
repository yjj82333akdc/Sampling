import numpy as np
from scipy import stats


class kernel_density:
    def __init__(self, dim=None, data=None, bw_method=None):
        """
        Optional: directly fit a KDE if dim and data are provided.
        """
        self.dim = None
        self.kernel = None
        if dim is not None and data is not None:
            self.fit(dim, data, bw_method=bw_method)

    def fit(self, dim, data, bw_method=None):
        """
        Fit a Gaussian KDE to the given data.

        Parameters
        ----------
        dim : int
            Dimension of the data.
        data : array-like, shape (N, dim)
            Training data.
        bw_method : str, scalar or callable, optional
            Bandwidth method for scipy.stats.gaussian_kde.
        """
        self.dim = dim
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != dim:
            raise ValueError(f"data should have shape (N, {dim}), got {data.shape}.")

        # gaussian_kde expects shape (dim, N)
        dataset = data.T
        self.kernel = stats.gaussian_kde(dataset=dataset, bw_method=bw_method)

    def evaluate(self, X_new):
        """
        Evaluate fitted KDE at new points.

        Parameters
        ----------
        X_new : array-like, shape (M, dim) or (dim,) for a single point.

        Returns
        -------
        vals : ndarray, shape (M,)
        """
        if self.kernel is None:
            raise ValueError("KDE not fitted. Call fit() first or use compute().")

        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        if X_new.shape[1] != self.dim:
            raise ValueError(
                f"X_new should have second dimension {self.dim}, got {X_new.shape[1]}."
            )

        # gaussian_kde expects shape (dim, M)
        points = X_new.T
        return self.kernel(points)

    def compute(self, dim, data, X_new, bw_method=None):
        """
        Backwards-compatible one-shot API: fit on (dim, data), then evaluate at X_new.
        """
        self.fit(dim, data, bw_method=bw_method)
        return self.evaluate(X_new)
