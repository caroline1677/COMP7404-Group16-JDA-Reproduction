"""
JDA Comparison: A framework for comparing transfer learning methods
Implements NN, PCA, TCA, GFK, TSL, and JDA for domain adaptation experiments.

Reference papers:
- JDA: Long et al., "Transfer Feature Learning with Joint Distribution Adaptation", ICCV 2013
- TCA: Pan et al., "Domain Adaptation via Transfer Component Analysis", TNN 2011
- GFK: Gong et al., "Geodesic Flow Kernel for Unsupervised Domain Adaptation", CVPR 2012
- TSL: Si et al., "Bregman divergence-based regularization for transfer subspace learning", TKDE 2010
"""

import argparse
import os
import time
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

np.random.seed(42)

# Default method execution order
DEFAULT_METHOD_ORDER = ["NN", "PCA", "GFK", "TCA", "TSL", "JDA"]


def load_preset_data(dataset_type, src_name, tar_name, data_dir="data"):
    """Load source and target domain data based on dataset type (preset mode).

    Args:
        dataset_type: 'digit', 'coil', 'pie', or 'surf'
        src_name: Source domain name (e.g., 'USPS', 'COIL1', 'PIE05', 'webcam')
        tar_name: Target domain name (e.g., 'MNIST', 'COIL2', 'PIE27', 'dslr')
        data_dir: Path to data directory

    Returns:
        Xs, Ys, Xt, Yt: Source/target features and labels
    """
    if dataset_type == "digit":
        # MNIST vs USPS
        data = scipy.io.loadmat(f"{data_dir}/digit/MNIST_vs_USPS.mat")
        Xs = data["X_src"].T.astype(np.float64)
        Ys = data["Y_src"].ravel()
        Xt = data["X_tar"].T.astype(np.float64)
        Yt = data["Y_tar"].ravel()

    elif dataset_type == "coil":
        # COIL1 -> COIL2: Use X_src and X_tar from the SAME file
        # Note: COIL_1.X_src == COIL_2.X_tar, COIL_1.X_tar == COIL_2.X_src
        # So we use COIL_1 file with both X_src (source) and X_tar (target)
        data = scipy.io.loadmat(f"{data_dir}/coil/COIL_1.mat")
        Xs = data["X_src"].T.astype(np.float64)
        Ys = data["Y_src"].ravel()
        Xt = data["X_tar"].T.astype(np.float64)
        Yt = data["Y_tar"].ravel()

    elif dataset_type == "pie":
        # PIE datasets - src_name is like "PIE05", extract last 4 chars
        src_suffix = src_name.replace("PIE", "") if "PIE" in src_name else src_name[-4:]
        tar_suffix = tar_name.replace("PIE", "") if "PIE" in tar_name else tar_name[-4:]
        src_file = f"{data_dir}/pie/PIE{src_suffix}.mat"
        tar_file = f"{data_dir}/pie/PIE{tar_suffix}.mat"
        src_data = scipy.io.loadmat(src_file)
        tar_data = scipy.io.loadmat(tar_file)
        Xs = src_data["fea"].astype(np.float64)
        Ys = src_data["gnd"].ravel()
        Xt = tar_data["fea"].astype(np.float64)
        Yt = tar_data["gnd"].ravel()
        # Normalize to [0,1] if needed
        if Xs.max() > 1:
            Xs = Xs / 255.0
        if Xt.max() > 1:
            Xt = Xt / 255.0

    elif dataset_type == "surf":
        # Office SURF features (already z-score standardized)
        src_file = f"{data_dir}/surf/{src_name}_zscore_SURF_L10.mat"
        tar_file = f"{data_dir}/surf/{tar_name}_zscore_SURF_L10.mat"
        src_data = scipy.io.loadmat(src_file)
        tar_data = scipy.io.loadmat(tar_file)

        if "Xt" in src_data:
            Xs, Ys = src_data["Xt"], src_data["Yt"].ravel()
        else:
            Xs, Ys = src_data["Xs"], src_data["Ys"].ravel()

        if "Xt" in tar_data:
            Xt, Yt = tar_data["Xt"], tar_data["Yt"].ravel()
        else:
            Xt, Yt = tar_data["Xs"], tar_data["Ys"].ravel()

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return Xs, Ys, Xt, Yt


def load_custom_data(src_file, src_feat, src_label, tar_file, tar_feat, tar_label):
    """Load source and target data from custom .mat files.

    Args:
        src_file: Path to source .mat file
        src_feat: Variable name for source features
        src_label: Variable name for source labels
        tar_file: Path to target .mat file
        tar_feat: Variable name for target features
        tar_label: Variable name for target labels

    Returns:
        Xs, Ys, Xt, Yt: Source/target features and labels
    """
    src_data = scipy.io.loadmat(src_file)
    tar_data = scipy.io.loadmat(tar_file)

    # Get source data
    Xs = src_data[src_feat].astype(np.float64)
    Ys = src_data[src_label].ravel()

    # Get target data
    Xt = tar_data[tar_feat].astype(np.float64)
    Yt = tar_data[tar_label].ravel()

    return Xs, Ys, Xt, Yt


# ============ Transfer Learning Methods ============

def method_nn(Xs, Ys, Xt, Yt):
    """Nearest Neighbor classifier (baseline)"""
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs, Ys)
    return sklearn.metrics.accuracy_score(Yt, clf.predict(Xt)) * 100


def method_pca(Xs, Ys, Xt, Yt, dim=100):
    """PCA-based dimensionality reduction + NN"""
    pca = PCA(n_components=min(dim, Xs.shape[1]))
    Xs_pca = pca.fit_transform(Xs)
    Xt_pca = pca.transform(Xt)
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs_pca, Ys)
    return sklearn.metrics.accuracy_score(Yt, clf.predict(Xt_pca)) * 100


class TCA:
    """Transfer Component Analysis
    Uses MMD for marginal distribution adaptation only.
    Reference: Pan et al., TNN 2011
    """
    def __init__(self, dim=100, lamb=0.1):
        self.dim, self.lamb = dim, lamb

    def fit_predict(self, Xs, Ys, Xt, Yt):
        X = np.hstack((Xs.T, Xt.T))
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)

        # MMD matrix for marginal distribution
        e = np.vstack((1/ns * np.ones((ns, 1)), -1/nt * np.ones((nt, 1))))
        M = e * e.T

        # Centering matrix
        H = np.eye(n) - 1/n * np.ones((n, n))

        K = X
        a = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(m)
        b = np.linalg.multi_dot([K, H, K.T]) + 1e-6 * np.eye(m)

        # Ensure symmetry
        a = (a + a.T) / 2
        b = (b + b.T) / 2

        w, V = scipy.linalg.eig(a, b)
        w = np.real(w)
        V = np.real(V)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = A.T @ K
        Z = np.real(Z)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        return sklearn.metrics.accuracy_score(Yt, clf.predict(Xt_new)) * 100


class GFK:
    """Geodesic Flow Kernel (GFK) for unsupervised domain adaptation.
    Reference: Gong et al., CVPR 2012.

    This implementation uses proper kernel distance metric:
    D = diag(K_ss) + diag(K_tt) - 2*K_st
    """
    def __init__(self, dim=100):
        self.dim = dim
        self.eps = 1e-20

    def fit_predict(self, Xs, Ys, Xt, Yt):
        d = min(self.dim, Xs.shape[1], Xt.shape[1])

        # Step 1: PCA on source and target separately to get subspaces
        pca_s = PCA(n_components=d)
        pca_s.fit(Xs)
        Ps = pca_s.components_.T

        pca_t = PCA(n_components=d)
        pca_t.fit(Xt)
        Pt = pca_t.components_.T

        # Step 2: Compute principal angles between subspaces
        C = Ps.T @ Pt
        U, s, Vh = np.linalg.svd(C, full_matrices=False)
        cos_theta = s
        sin_theta = np.sqrt(np.maximum(1 - cos_theta**2, 0))

        sin2_theta = sin_theta**2
        sin_2theta = 2 * sin_theta * cos_theta

        # Step 3: Construct the R matrix
        I = np.eye(d)
        A = (I + np.diag(sin2_theta)) / 2
        B = np.diag(sin_2theta) / 2
        D = (I - np.diag(sin2_theta)) / 2
        R_blk = np.block([[A, B], [B.T, D]])

        # Transform basis
        PsU = Ps @ U
        PtV = Pt @ Vh.T

        # Project data onto combined subspace
        z_s = np.hstack([Xs @ PsU, Xs @ PtV])
        z_t = np.hstack([Xt @ PsU, Xt @ PtV])

        # Compute kernel matrices
        K_ss = z_s @ R_blk @ z_s.T
        K_tt = z_t @ R_blk @ z_t.T
        K_st = z_s @ R_blk @ z_t.T

        # Compute distance matrix using kernel trick
        diag_ss = np.diag(K_ss).reshape(-1, 1)
        diag_tt = np.diag(K_tt).reshape(1, -1)
        D = diag_ss + diag_tt - 2 * K_st
        D = np.maximum(D, 0)

        # 1-NN based on minimum distance
        pred = Ys[np.argmin(D, axis=0)]

        return np.mean(pred == Yt) * 100


class TSL:
    """Transfer Subspace Learning using Bregman divergence (LogDet)
    Reference: Si et al., TKDE 2010

    Implements the original TSL objective:
    - Minimize Bregman divergence (LogDet) between source and target
    - Maximize variance (similar to PCA)

    Uses iterative optimization with sample-level weighting.
    """
    def __init__(self, dim=100, lamb=1.0, max_iter=10):
        self.dim, self.lamb = dim, lamb
        self.max_iter = max_iter

    def _logdet(self, A):
        """Compute log determinant with numerical stability"""
        A = (A + A.T) / 2
        try:
            L = np.linalg.cholesky(A + 1e-8 * np.eye(A.shape[0]))
            return 2 * np.sum(np.log(np.diag(L) + 1e-10))
        except:
            w, _ = np.linalg.eigh(A)
            w = np.maximum(w, 1e-10)
            return np.sum(np.log(w))

    def fit_predict(self, Xs, Ys, Xt, Yt):
        m, n = Xs.shape[1], Xs.shape[0] + Xt.shape[0]
        ns, nt = len(Xs), len(Xt)
        d = self.dim

        X = np.hstack((Xs.T, Xt.T))
        H = np.eye(n) - 1/n * np.ones((n, n))

        Xc = X @ H
        Xs_c = Xc[:, :ns]
        Xt_c = Xc[:, ns:]

        # Initialize A using PCA
        pca = PCA(n_components=min(d, m))
        X_combined = np.vstack([Xs, Xt])
        pca.fit(X_combined)
        A = pca.components_.T

        # Iterative optimization
        for _ in range(self.max_iter):
            Zs = A.T @ Xs_c
            Zt = A.T @ Xt_c

            Cov_s = Zs @ Zs.T / ns
            Cov_t = Zt @ Zt.T / nt

            Cov_s_reg = Cov_s + 1e-6 * np.eye(d)
            Cov_t_reg = Cov_t + 1e-6 * np.eye(d)

            try:
                Cov_s_inv = np.linalg.inv(Cov_s_reg)
                Cov_t_inv = np.linalg.inv(Cov_t_reg)
            except:
                Cov_s_inv = np.eye(d)
                Cov_t_inv = np.eye(d)

            W_s = Cov_s_inv
            W_t = Cov_t_inv

            # Sample-level weighting based on Mahalanobis distance
            ws_diag = np.diag(Zs.T @ W_s @ Zs.T.T) / ns
            wt_diag = np.diag(Zt.T @ W_t @ Zt.T.T) / nt

            ws_diag = ws_diag / (np.sum(ws_diag) + 1e-10)
            wt_diag = wt_diag / (np.sum(wt_diag) + 1e-10)

            e_s = np.sqrt(ws_diag).reshape(-1, 1)
            e_t = np.sqrt(wt_diag).reshape(-1, 1)
            e = np.vstack((e_s, -e_t))
            M = e @ e.T

            # Solve generalized eigenvalue problem
            K = X
            a = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(m)
            b = np.linalg.multi_dot([K, H, K.T]) + 1e-6 * np.eye(m)

            a = (a + a.T) / 2
            b = (b + b.T) / 2

            w, V = scipy.linalg.eig(a, b)
            w = np.real(w)
            V = np.real(V)

            ind = np.argsort(w)
            A_new = V[:, ind[:d]]

            Q, _ = np.linalg.qr(A_new)
            A = Q

        Z = A.T @ X
        Z = np.real(Z)
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new = Z[:, :ns].T
        Xt_new = Z[:, ns:].T

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        return sklearn.metrics.accuracy_score(Yt, clf.predict(Xt_new)) * 100


class JDA:
    """Joint Distribution Adaptation
    Key differences from TCA:
    - Adapts both marginal P(x) and conditional Q(y|x)
    - Iteratively refines target pseudo-labels
    Reference: Long et al., ICCV 2013
    """
    def __init__(self, dim=100, lamb=0.1, T=10):
        self.dim, self.lamb, self.T = dim, lamb, T

    def fit_predict(self, Xs, Ys, Xt, Yt):
        X = np.hstack((Xs.T, Xt.T))
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)

        # MMD for marginal distribution
        e = np.vstack((1/ns * np.ones((ns, 1)), -1/nt * np.ones((nt, 1))))
        M0 = e * e.T

        C = len(np.unique(Ys))
        H = np.eye(n) - 1/n * np.ones((n, n))

        Y_tar_pseudo = None

        for t in range(self.T):
            N = np.zeros((n, n))

            # Conditional distribution adaptation
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e_c = np.zeros((n, 1))

                    idx_s = np.where(Ys == c)[0]
                    if len(idx_s) > 0:
                        e_c[idx_s] = 1.0 / len(idx_s)

                    idx_t = np.where(Y_tar_pseudo == c)[0]
                    if len(idx_t) > 0:
                        e_c[idx_t + ns] = -1.0 / len(idx_t)

                    if np.sum(np.abs(e_c)) > 0:
                        N = N + np.dot(e_c, e_c.T)

            M = M0 + N

            K = X
            a = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(m)
            b = np.linalg.multi_dot([K, H, K.T]) + 1e-6 * np.eye(m)

            a = (a + a.T) / 2
            b = (b + b.T) / 2

            w, V = scipy.linalg.eig(a, b)
            w = np.real(w)
            V = np.real(V)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = A.T @ K
            Z = np.real(Z)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)

        return sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo) * 100


def run_single_method(method_name, Xs, Ys, Xt, Yt, dim, lamb, jda_iter, tsl_iter):
    """Run a single method and return accuracy and runtime.

    Args:
        method_name: Name of the method (NN, PCA, TCA, GFK, TSL, JDA)
        Xs, Ys: Source features and labels
        Xt, Yt: Target features and labels
        dim: Subspace dimensionality
        lamb: Regularization parameter
        jda_iter: Number of iterations for JDA
        tsl_iter: Number of iterations for TSL

    Returns:
        accuracy, runtime (seconds)
    """
    start_time = time.time()

    if method_name == "NN":
        acc = method_nn(Xs, Ys, Xt, Yt)
    elif method_name == "PCA":
        acc = method_pca(Xs, Ys, Xt, Yt, dim)
    elif method_name == "TCA":
        acc = TCA(dim, lamb).fit_predict(Xs, Ys, Xt, Yt)
    elif method_name == "GFK":
        acc = GFK(dim).fit_predict(Xs, Ys, Xt, Yt)
    elif method_name == "TSL":
        acc = TSL(dim, lamb, tsl_iter).fit_predict(Xs, Ys, Xt, Yt)
    elif method_name == "JDA":
        acc = JDA(dim, lamb, jda_iter).fit_predict(Xs, Ys, Xt, Yt)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    runtime = time.time() - start_time
    return acc, runtime


def print_markdown_table(results):
    """Print results in markdown table format."""
    # Header
    print("\n| Method | Accuracy | Runtime (s) |")
    print("|--------|----------|-------------|")
    for method_name, (acc, runtime) in results.items():
        print(f"| {method_name:6} | {acc:7.2f}% | {runtime:10.3f} |")


# ============ Main Function ============

def run_comparison(args):
    """Run transfer learning methods comparison."""
    # Load data
    if args.dataset:
        # Preset dataset mode
        Xs, Ys, Xt, Yt = load_preset_data(args.dataset, args.src, args.tar, args.data_dir)
        task_name = f"{args.src} -> {args.tar}"
    elif args.src_file and args.tar_file:
        # Custom data mode
        Xs, Ys, Xt, Yt = load_custom_data(
            args.src_file, args.src_feat, args.src_label,
            args.tar_file, args.tar_feat, args.tar_label
        )
        task_name = f"{args.src_file} -> {args.tar_file}"
    else:
        raise ValueError("Please specify either --dataset (preset) or --src-file/--tar-file (custom)")

    dim = args.dim
    lamb = args.lamb

    # Determine iterations for each method
    jda_iter = args.jda_iter if args.jda_iter else args.iter
    tsl_iter = args.tsl_iter if args.tsl_iter else args.iter

    # Parse methods
    if args.methods == "all" or args.methods is None:
        method_list = DEFAULT_METHOD_ORDER
    else:
        # Support comma-separated or space-separated
        if isinstance(args.methods, str) and ',' in args.methods:
            method_list = [m.strip().upper() for m in args.methods.split(',')]
        else:
            method_list = [m.upper() for m in args.methods]

    # Run methods
    results = {}

    for method_name in method_list:
        acc, runtime = run_single_method(method_name, Xs, Ys, Xt, Yt, dim, lamb, jda_iter, tsl_iter)
        results[method_name] = (acc, runtime)

    # Print results
    print(f"\n{'='*60}")
    print(f"Transfer Learning Comparison: {task_name}")
    print(f"Dataset: {args.dataset or 'custom'}, Dim: {dim}, Lambda: {lamb}")
    print(f"JDA Iter: {jda_iter}, TSL Iter: {tsl_iter}")
    print(f"{'='*60}")

    # Print markdown table
    print_markdown_table(results)

    # Save to file if requested
    if args.output:
        # Check if file exists and has content
        file_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0

        with open(args.output, "a") as f:
            # Write header if file doesn't exist or is empty
            if not file_exists:
                f.write("Task,Method,Accuracy,Runtime\n")

            for method_name in method_list:
                acc, runtime = results[method_name]
                f.write(f"{task_name},{method_name},{acc:.2f},{runtime:.3f}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="JDA Comparison Framework for Transfer Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preset dataset mode (basic)
  python jda_comparison.py --dataset digit --src USPS --tar MNIST
  python jda_comparison.py --dataset surf --src webcam --tar dslr

  # Preset dataset with custom parameters
  python jda_comparison.py --dataset surf --src webcam --tar dslr --dim 50 --lamb 1.0 --jda-iter 15

  # Custom data mode (advanced)
  python jda_comparison.py \\
      --src-file data/source.mat --src-feat X --src-label y \\
      --tar-file data/target.mat --tar-feat X --tar-label y

  # Run only specific methods (comma-separated)
  python jda_comparison.py --dataset coil --src COIL1 --tar COIL2 --methods nn,pca,jda

  # Save results to CSV
  python jda_comparison.py --dataset pie --src PIE05 --tar PIE27 --output results.csv
        """
    )

    # Data input options (mutually exclusive groups handled in code)
    data_group = parser.add_argument_group('Data Input (choose one mode)')
    data_group.add_argument("--dataset", type=str, default=None,
                            choices=["digit", "coil", "pie", "surf"],
                            help="Dataset type (preset mode)")
    data_group.add_argument("--src", type=str, default=None,
                            help="Source domain name (used with --dataset)")
    data_group.add_argument("--tar", type=str, default=None,
                            help="Target domain name (used with --dataset)")
    data_group.add_argument("--data-dir", type=str, default="data",
                            help="Path to data directory (default: data)")

    # Custom data mode
    custom_group = parser.add_argument_group('Custom Data (advanced mode)')
    custom_group.add_argument("--src-file", type=str, default=None,
                            help="Path to source .mat file")
    custom_group.add_argument("--src-feat", type=str, default=None,
                            help="Variable name for source features in .mat file")
    custom_group.add_argument("--src-label", type=str, default=None,
                            help="Variable name for source labels in .mat file")
    custom_group.add_argument("--tar-file", type=str, default=None,
                            help="Path to target .mat file")
    custom_group.add_argument("--tar-feat", type=str, default=None,
                            help="Variable name for target features in .mat file")
    custom_group.add_argument("--tar-label", type=str, default=None,
                            help="Variable name for target labels in .mat file")

    # Method parameters
    param_group = parser.add_argument_group('Method Parameters')
    param_group.add_argument("--dim", type=int, default=100,
                            help="Dimensionality of subspace (default: 100)")
    param_group.add_argument("--lamb", type=float, default=0.1,
                            help="Regularization parameter (TCA/TSL/JDA) (default: 0.1)")

    # Method-specific iterations
    param_group.add_argument("--iter", type=int, default=10,
                            help="Default iterations for JDA and TSL (default: 10)")
    param_group.add_argument("--jda-iter", type=int, default=None,
                            help="Iterations for JDA (default: use --iter value)")
    param_group.add_argument("--tsl-iter", type=int, default=None,
                            help="Iterations for TSL internal optimization (default: use --iter value)")

    # Output options
    parser.add_argument("--methods", type=str, default="all",
                        help="Methods to run: 'all' or comma-separated list (nn,pca,tca,gfk,tsl,jda)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file to append results")

    args = parser.parse_args()

    # Validate arguments
    if args.dataset:
        if not args.src or not args.tar:
            parser.error("--src and --tar are required when using --dataset")
    elif args.src_file or args.tar_file:
        if not all([args.src_file, args.src_feat, args.src_label,
                    args.tar_file, args.tar_feat, args.tar_label]):
            parser.error("Custom mode requires: --src-file, --src-feat, --src-label, --tar-file, --tar-feat, --tar-label")
    else:
        parser.error("Please specify either --dataset (preset) or --src-file/--tar-file (custom)")

    run_comparison(args)


if __name__ == "__main__":
    main()
