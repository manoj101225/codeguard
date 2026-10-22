"""
Transfer Component Analysis Plus (TCA+) implementation.

Based on:
  Nam, J., Pan, S. J., & Kim, S. (2013). Transfer Defect Learning.
  ICSE 2013.  https://dl.acm.org/doi/10.1109/ICSE.2013.6606584

TCA+ steps:
  1. Normalize source and target feature matrices (z-score, per-feature on
     combined data so the scales match).
  2. Compute an RBF-kernel matrix K over the combined (source + target) data.
  3. Construct the MMD (Maximum Mean Discrepancy) penalty matrix L.
  4. Solve the generalized eigenvalue problem to find the projection W that
     minimizes MMD while preserving variance.
  5. Project both source and target into the shared latent subspace.
  6. The projected source data (with labels) is used to train; the projected
     target data is used for inference / fine-tuning.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh


class TCAPlus:
    """
    Transfer Component Analysis+ for cross-project defect prediction.

    Parameters
    ----------
    n_components : int
        Dimensionality of the shared latent subspace (default 10).
    kernel_type : str
        Kernel to use; currently only 'rbf' is supported.
    mu : float
        Regularisation weight balancing MMD and variance (default 1.0).
    gamma : float or None
        RBF bandwidth.  If None, uses the median-heuristic on pairwise distances.
    """

    def __init__(
        self,
        n_components: int = 10,
        kernel_type: str = "rbf",
        mu: float = 1.0,
        gamma: float | None = None,
    ):
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.mu = mu
        self.gamma = gamma
        self._scaler = StandardScaler()
        self._W = None          # projection matrix
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal kernel helpers
    # ------------------------------------------------------------------

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix K(X, Y)."""
        from sklearn.metrics.pairwise import rbf_kernel
        gamma = self.gamma
        if gamma is None:
            # Median heuristic
            from sklearn.metrics.pairwise import euclidean_distances
            D = euclidean_distances(X, Y)
            median_dist = np.median(D[D > 0]) if D[D > 0].size else 1.0
            gamma = 1.0 / (2 * median_dist ** 2 + 1e-10)
        return rbf_kernel(X, Y, gamma=gamma)

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self.kernel_type == "rbf":
            return self._rbf_kernel(X, Y)
        raise ValueError(f"Unsupported kernel: {self.kernel_type}")

    # ------------------------------------------------------------------
    # Fit: compute the projection matrix W
    # ------------------------------------------------------------------

    def fit(self, X_src: np.ndarray, X_tgt: np.ndarray):
        """
        Fit TCA+ using source and target feature matrices (no labels needed).

        Parameters
        ----------
        X_src : (n_s, d) source feature matrix
        X_tgt : (n_t, d) target feature matrix
        """
        n_s = X_src.shape[0]
        n_t = X_tgt.shape[0]
        n = n_s + n_t

        # 1. Normalize
        X_combined = np.vstack([X_src, X_tgt])
        X_combined = self._scaler.fit_transform(X_combined)
        X_src_n = X_combined[:n_s]
        X_tgt_n = X_combined[n_s:]

        # 2. Kernel matrix K ∈ ℝ^{n×n}
        K = self._compute_kernel(X_combined, X_combined)
        # Regularise
        K = K + 1e-5 * np.eye(n)

        # 3. MMD matrix L
        #    L_ij = 1/n_s^2  if both i,j ∈ source
        #          -1/(n_s*n_t) if one is source, other is target
        #           1/n_t^2  if both i,j ∈ target
        L = np.zeros((n, n))
        L[:n_s, :n_s] = 1.0 / (n_s * n_s)
        L[n_s:, n_s:] = 1.0 / (n_t * n_t)
        L[:n_s, n_s:] = -1.0 / (n_s * n_t)
        L[n_s:, :n_s] = -1.0 / (n_s * n_t)

        # 4. Centering matrix H
        H = np.eye(n) - (1.0 / n) * np.ones((n, n))

        # 5. Generalised eigenvalue problem:
        #    (K L K + mu*I) W = K H K W lambda
        A = K @ L @ K + self.mu * np.eye(n)
        B = K @ H @ K

        # Solve for top-k eigenvectors of B^{-1} A (minimise A, maximise B)
        k = min(self.n_components, n - 1)
        try:
            eigenvalues, eigenvectors = eigh(B, A, subset_by_index=[n - k, n - 1])
        except Exception:
            # Fallback: regular eig on B @ pinv(A)
            M = np.linalg.pinv(A) @ B
            eigenvalues, eigenvectors = np.linalg.eigh(M)
            eigenvectors = eigenvectors[:, -k:]

        self._W = eigenvectors          # shape (n, k)
        self._X_combined_n = X_combined  # needed to project new data via kernel
        self._n_s = n_s
        self._n_t = n_t
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform: project source & target into latent subspace
    # ------------------------------------------------------------------

    def transform(self, X_src: np.ndarray, X_tgt: np.ndarray):
        """
        Project source and target into the shared latent space.

        Returns
        -------
        Z_src : (n_s, k) projected source
        Z_tgt : (n_t, k) projected target
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        X_combined_n = self._X_combined_n
        X_src_n = self._scaler.transform(X_src)
        X_tgt_n = self._scaler.transform(X_tgt)
        X_new_n = np.vstack([X_src_n, X_tgt_n])

        K_new = self._compute_kernel(X_new_n, X_combined_n)  # (n_new, n_orig)
        Z = K_new @ self._W                                    # (n_new, k)

        n_s = X_src.shape[0]
        return Z[:n_s], Z[n_s:]

    def fit_transform(self, X_src: np.ndarray, X_tgt: np.ndarray):
        """Convenience: fit then transform."""
        self.fit(X_src, X_tgt)
        return self.transform(X_src, X_tgt)

    def transform_single(self, x: np.ndarray) -> np.ndarray:
        """
        Project a single feature vector x (shape (d,) or (1, d)) for what-if inference.
        Uses the already-fitted kernel mapping against the training combined data.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform_single().")
        x2d = np.array(x).reshape(1, -1)
        x_n = self._scaler.transform(x2d)
        k = self._compute_kernel(x_n, self._X_combined_n)  # (1, n_orig)
        z = k @ self._W                                       # (1, k)
        return z
