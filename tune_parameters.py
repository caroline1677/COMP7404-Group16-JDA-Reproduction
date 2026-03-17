"""
Parameter Tuning Script for JDA Comparison Framework

This script performs grid search to find optimal hyperparameters for each method.
As per the original paper, target domain labels are used for parameter selection.

NOTE: Using target domain labels for parameter tuning is only acceptable for
research reproduction. In real-world scenarios, this is not feasible as
target domain labels are unknown.

Usage:
    python tune_parameters.py --dataset digit --src USPS --tar MNIST
    python tune_parameters.py --dataset digit --src USPS --tar MNIST --methods pca,gfk
    python tune_parameters.py --dataset digit --src USPS --tar MNIST --compare-paper
"""

# Original paper results (for comparison)
PAPER_RESULTS = {
    # Digit datasets
    ("digit", "USPS", "MNIST"): {"NN": 44.70, "PCA": 44.95, "GFK": 46.45, "TCA": 51.05, "TSL": 53.75, "JDA": 59.65},
    ("digit", "MNIST", "USPS"): {"NN": 65.94, "PCA": 66.22, "GFK": 67.22, "TCA": 56.28, "TSL": 66.06, "JDA": 67.28},
    # COIL datasets
    ("coil", "COIL1", "COIL2"): {"NN": 83.61, "PCA": 84.72, "GFK": 72.50, "TCA": 88.47, "TSL": 88.06, "JDA": 89.31},
    ("coil", "COIL2", "COIL1"): {"NN": 82.78, "PCA": 84.03, "GFK": 74.17, "TCA": 85.83, "TSL": 87.92, "JDA": 88.47},
    # PIE datasets (PIE1=PIE05, PIE2=PIE07, PIE3=PIE09, PIE4=PIE27, PIE5=PIE29)
    ("pie", "PIE1", "PIE2"): {"NN": 26.09, "PCA": 24.80, "GFK": 26.15, "TCA": 40.76, "TSL": 44.08, "JDA": 58.81},
    ("pie", "PIE1", "PIE3"): {"NN": 26.59, "PCA": 25.18, "GFK": 27.27, "TCA": 41.79, "TSL": 47.49, "JDA": 54.23},
    ("pie", "PIE1", "PIE4"): {"NN": 30.67, "PCA": 29.26, "GFK": 31.15, "TCA": 59.63, "TSL": 62.78, "JDA": 84.50},
    ("pie", "PIE1", "PIE5"): {"NN": 16.67, "PCA": 16.30, "GFK": 17.59, "TCA": 29.35, "TSL": 36.15, "JDA": 49.75},
    ("pie", "PIE2", "PIE1"): {"NN": 24.49, "PCA": 24.22, "GFK": 25.24, "TCA": 41.81, "TSL": 46.28, "JDA": 57.62},
    ("pie", "PIE2", "PIE3"): {"NN": 46.63, "PCA": 45.53, "GFK": 47.37, "TCA": 51.47, "TSL": 57.60, "JDA": 62.93},
    ("pie", "PIE2", "PIE4"): {"NN": 54.07, "PCA": 53.35, "GFK": 54.25, "TCA": 64.73, "TSL": 71.43, "JDA": 75.82},
    ("pie", "PIE2", "PIE5"): {"NN": 26.53, "PCA": 25.43, "GFK": 27.08, "TCA": 33.70, "TSL": 35.66, "JDA": 39.89},
    ("pie", "PIE3", "PIE1"): {"NN": 21.37, "PCA": 20.95, "GFK": 21.82, "TCA": 34.69, "TSL": 36.94, "JDA": 50.96},
    ("pie", "PIE3", "PIE2"): {"NN": 41.01, "PCA": 40.45, "GFK": 43.16, "TCA": 47.70, "TSL": 47.02, "JDA": 57.95},
    ("pie", "PIE3", "PIE4"): {"NN": 46.53, "PCA": 46.14, "GFK": 46.41, "TCA": 56.23, "TSL": 59.45, "JDA": 68.45},
    ("pie", "PIE3", "PIE5"): {"NN": 26.23, "PCA": 25.31, "GFK": 26.78, "TCA": 33.15, "TSL": 36.34, "JDA": 39.95},
    ("pie", "PIE4", "PIE1"): {"NN": 32.95, "PCA": 31.96, "GFK": 34.24, "TCA": 55.64, "TSL": 63.66, "JDA": 80.58},
    ("pie", "PIE4", "PIE2"): {"NN": 62.68, "PCA": 60.96, "GFK": 62.92, "TCA": 67.83, "TSL": 72.68, "JDA": 82.63},
    ("pie", "PIE4", "PIE3"): {"NN": 73.22, "PCA": 72.18, "GFK": 73.35, "TCA": 75.86, "TSL": 83.52, "JDA": 87.25},
    ("pie", "PIE4", "PIE5"): {"NN": 37.19, "PCA": 35.11, "GFK": 37.38, "TCA": 40.26, "TSL": 44.79, "JDA": 54.66},
    ("pie", "PIE5", "PIE1"): {"NN": 18.49, "PCA": 18.85, "GFK": 20.35, "TCA": 26.98, "TSL": 33.28, "JDA": 46.46},
    ("pie", "PIE5", "PIE2"): {"NN": 24.19, "PCA": 23.39, "GFK": 24.62, "TCA": 29.90, "TSL": 34.13, "JDA": 42.05},
    ("pie", "PIE5", "PIE3"): {"NN": 28.31, "PCA": 27.21, "GFK": 28.49, "TCA": 29.90, "TSL": 36.58, "JDA": 53.31},
    ("pie", "PIE5", "PIE4"): {"NN": 31.24, "PCA": 30.34, "GFK": 31.33, "TCA": 33.64, "TSL": 38.75, "JDA": 57.01},
    # SURF datasets (C=Caltech10, A=amazon, W=webcam, D=dslr)
    ("surf", "Caltech10", "amazon"): {"NN": 23.70, "PCA": 36.95, "GFK": 41.02, "TCA": 38.20, "TSL": 44.47, "JDA": 44.78},
    ("surf", "Caltech10", "webcam"): {"NN": 25.76, "PCA": 32.54, "GFK": 40.68, "TCA": 38.64, "TSL": 34.24, "JDA": 41.69},
    ("surf", "Caltech10", "dslr"): {"NN": 25.48, "PCA": 38.22, "GFK": 38.85, "TCA": 41.40, "TSL": 43.31, "JDA": 45.22},
    ("surf", "amazon", "Caltech10"): {"NN": 26.00, "PCA": 34.73, "GFK": 40.25, "TCA": 37.76, "TSL": 37.58, "JDA": 39.36},
    ("surf", "amazon", "webcam"): {"NN": 29.83, "PCA": 35.59, "GFK": 38.98, "TCA": 37.63, "TSL": 33.90, "JDA": 37.97},
    ("surf", "amazon", "dslr"): {"NN": 25.48, "PCA": 27.39, "GFK": 36.31, "TCA": 33.12, "TSL": 26.11, "JDA": 39.49},
    ("surf", "webcam", "Caltech10"): {"NN": 19.86, "PCA": 26.36, "GFK": 30.72, "TCA": 29.30, "TSL": 29.83, "JDA": 31.17},
    ("surf", "webcam", "amazon"): {"NN": 22.96, "PCA": 31.00, "GFK": 29.75, "TCA": 30.06, "TSL": 30.27, "JDA": 32.78},
    ("surf", "webcam", "dslr"): {"NN": 59.24, "PCA": 77.07, "GFK": 80.89, "TCA": 87.26, "TSL": 87.26, "JDA": 89.17},
    ("surf", "dslr", "Caltech10"): {"NN": 26.27, "PCA": 29.65, "GFK": 30.28, "TCA": 31.70, "TSL": 28.50, "JDA": 31.52},
    ("surf", "dslr", "amazon"): {"NN": 28.50, "PCA": 32.05, "GFK": 32.05, "TCA": 32.15, "TSL": 27.56, "JDA": 33.09},
    ("surf", "dslr", "webcam"): {"NN": 63.39, "PCA": 75.93, "GFK": 75.59, "TCA": 86.10, "TSL": 89.49, "JDA": 89.49},
}

import argparse
import os
import sys
import time
import numpy as np
import scipy.io
import scipy.linalg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import sklearn.metrics
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

np.random.seed(42)

# Parameter search space as per paper
# PCA, GFK, TCA use k up to 200
K_VALUES_LARGE = list(range(10, 201, 10))  # [10, 20, 30, ..., 200]
# TSL, JDA use k up to 150 (slower methods)
K_VALUES_SMALL = list(range(10, 151, 10))  # [10, 20, 30, ..., 150]
LAMBDA_VALUES = [0.01, 0.1, 1.0]  # Reduced set as per paper
JDA_ITERS = 10  # Fixed as per paper

# ============== Module-level parallel task functions ==============
# These are defined at module level for ProcessPoolExecutor compatibility

def _pca_task(args):
    """Task for PCA parallel execution."""
    k, Xs, Ys, Xt, Yt = args
    acc = run_pca(Xs, Ys, Xt, Yt, k)
    return (k, acc)

def _gfk_task(args):
    """Task for GFK parallel execution."""
    k, Xs, Ys, Xt, Yt = args
    acc = run_gfk(Xs, Ys, Xt, Yt, k)
    return (k, acc)

def _tca_task(args):
    """Task for TCA parallel execution."""
    task, Xs, Ys, Xt, Yt = args
    k, lamb = task
    acc = run_tca(Xs, Ys, Xt, Yt, k, lamb)
    return ({"k": k, "lamb": lamb}, acc)

def _tsl_task(args):
    """Task for TSL parallel execution."""
    task, Xs, Ys, Xt, Yt = args
    k, lamb = task
    try:
        acc = run_tsl(Xs, Ys, Xt, Yt, k, lamb)
        if np.isnan(acc):
            acc = 0.0
    except:
        acc = 0.0
    return ({"k": k, "lamb": lamb}, acc)

def _jda_task(args):
    """Task for JDA parallel execution."""
    task, Xs, Ys, Xt, Yt = args
    k, lamb = task
    try:
        acc = run_jda(Xs, Ys, Xt, Yt, k, lamb, T=JDA_ITERS)
        if np.isnan(acc):
            acc = 0.0
    except:
        acc = 0.0
    return ({"k": k, "lamb": lamb}, acc)


# ============== Data Loading ==============
def load_preset_data(dataset_type, src_name, tar_name, data_dir="data"):
    """Load source and target domain data."""
    if dataset_type == "digit":
        data = scipy.io.loadmat(f"{data_dir}/digit/MNIST_vs_USPS.mat")
        Xs = data["X_src"].T.astype(np.float64)
        Ys = data["Y_src"].ravel()
        Xt = data["X_tar"].T.astype(np.float64)
        Yt = data["Y_tar"].ravel()
    elif dataset_type == "coil":
        data = scipy.io.loadmat(f"{data_dir}/coil/COIL_1.mat")
        Xs = data["X_src"].T.astype(np.float64)
        Ys = data["Y_src"].ravel()
        Xt = data["X_tar"].T.astype(np.float64)
        Yt = data["Y_tar"].ravel()
    elif dataset_type == "pie":
        src_suffix = src_name.replace("PIE", "") if "PIE" in src_name else src_name[-1]
        tar_suffix = tar_name.replace("PIE", "") if "PIE" in tar_name else tar_name[-1]
        src_file = f"{data_dir}/pie/PIE{src_suffix}.mat"
        tar_file = f"{data_dir}/pie/PIE{tar_suffix}.mat"
        src_data = scipy.io.loadmat(src_file)
        tar_data = scipy.io.loadmat(tar_file)
        Xs = src_data["fea"].astype(np.float64)
        Ys = src_data["gnd"].ravel()
        Xt = tar_data["fea"].astype(np.float64)
        Yt = tar_data["gnd"].ravel()
        if Xs.max() > 1:
            Xs = Xs / 255.0
        if Xt.max() > 1:
            Xt = Xt / 255.0
    elif dataset_type == "surf":
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
        raise ValueError(f"Unknown dataset: {dataset_type}")
    return Xs, Ys, Xt, Yt


# ============== Method Implementations ==============
def run_nn(Xs, Ys, Xt, Yt):
    """NN: Nearest Neighbor baseline (1-NN)."""
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs, Ys.ravel())
    acc = sklearn.metrics.accuracy_score(Yt, clf.predict(Xt)) * 100
    return acc


def run_pca(Xs, Ys, Xt, Yt, dim):
    """PCA: subspace dimensionality search.
    Note: Only use source data to fit PCA (no target label leakage).
    """
    # Only use source data to fit PCA
    pca = PCA(n_components=min(dim, Xs.shape[1]))
    Xs_new = pca.fit_transform(Xs)
    Xt_new = pca.transform(Xt)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs_new, Ys)
    acc = sklearn.metrics.accuracy_score(Yt, clf.predict(Xt_new)) * 100
    return acc


def run_gfk(Xs, Ys, Xt, Yt, dim):
    """GFK: geodesic flow kernel with subspace dimensionality search."""
    # Simplified GFK for parameter tuning (faster)
    d = min(dim, Xs.shape[1], Xt.shape[1])

    pca_s = PCA(n_components=d)
    pca_s.fit(Xs)
    Ps = pca_s.components_.T

    pca_t = PCA(n_components=d)
    pca_t.fit(Xt)
    Pt = pca_t.components_.T

    C = Ps.T @ Pt
    U, s, Vh = np.linalg.svd(C, full_matrices=False)
    cos_theta = s
    sin_theta = np.sqrt(np.maximum(1 - cos_theta**2, 0))

    sin2_theta = sin_theta**2
    sin_2theta = 2 * sin_theta * cos_theta

    I = np.eye(d)
    A = (I + np.diag(sin2_theta)) / 2
    B = np.diag(sin_2theta) / 2
    D = (I - np.diag(sin2_theta)) / 2
    R_blk = np.block([[A, B], [B.T, D]])

    PsU = Ps @ U
    PtV = Pt @ Vh.T

    z_s = np.hstack([Xs @ PsU, Xs @ PtV])
    z_t = np.hstack([Xt @ PsU, Xt @ PtV])

    K_ss = z_s @ R_blk @ z_s.T
    K_tt = z_t @ R_blk @ z_t.T
    K_st = z_s @ R_blk @ z_t.T

    diag_ss = np.diag(K_ss).reshape(-1, 1)
    diag_tt = np.diag(K_tt).reshape(1, -1)
    dist = diag_ss + diag_tt - 2 * K_st
    dist = np.maximum(dist, 0)

    pred = Ys[np.argmin(dist, axis=0)]
    acc = np.mean(pred == Yt) * 100
    return acc


def run_tca(Xs, Ys, Xt, Yt, dim, lamb):
    """TCA: Transfer Component Analysis."""
    X = np.hstack((Xs.T, Xt.T))
    m, n = X.shape
    ns, nt = len(Xs), len(Xt)

    e = np.vstack((1/ns * np.ones((ns, 1)), -1/nt * np.ones((nt, 1))))
    M = e * e.T
    H = np.eye(n) - 1/n * np.ones((n, n))

    K = X
    a = np.linalg.multi_dot([K, M, K.T]) + lamb * np.eye(m)
    b = np.linalg.multi_dot([K, H, K.T]) + 1e-6 * np.eye(m)

    a = (a + a.T) / 2
    b = (b + b.T) / 2

    try:
        w, V = scipy.linalg.eig(np.linalg.pinv(b) @ a)
    except:
        w, V = scipy.linalg.eig(a, b)
    w = np.real(w)
    V = np.real(V)
    ind = np.argsort(w)
    A = V[:, ind[:dim]]
    Z = A.T @ K
    Z = np.real(Z)
    Z /= np.linalg.norm(Z, axis=0) + 1e-12
    Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs_new, Ys.ravel())
    return sklearn.metrics.accuracy_score(Yt, clf.predict(Xt_new)) * 100


def _logdet(A):
    """Log determinant with numerical stability."""
    A = (A + A.T) / 2
    try:
        return np.log(np.linalg.det(A + 1e-6 * np.eye(A.shape[0])))
    except:
        return -1000


def run_tsl(Xs, Ys, Xt, Yt, dim, lamb, max_iter=10):
    """TSL: Transfer Subspace Learning - use implementation from jda_comparison.py"""
    from jda_comparison import TSL
    tsl = TSL(dim=dim, lamb=lamb, max_iter=max_iter)
    return tsl.fit_predict(Xs, Ys, Xt, Yt)


def run_jda(Xs, Ys, Xt, Yt, dim, lamb, T=10):
    """JDA: Joint Distribution Adaptation (fixed implementation)."""
    X = np.hstack((Xs.T, Xt.T))
    m, n = X.shape
    ns, nt = len(Xs), len(Xt)

    # MMD for marginal distribution
    e = np.vstack((1/ns * np.ones((ns, 1)), -1/nt * np.ones((nt, 1))))
    M0 = e * e.T

    C = len(np.unique(Ys))
    H = np.eye(n) - 1/n * np.ones((n, n))

    Y_tar_pseudo = None
    A = None

    for t in range(T):
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
        if A is None:
            # Initialize with PCA on first iteration
            pca = PCA(n_components=dim)
            pca.fit(X.T)
            A = pca.components_.T

        a = np.linalg.multi_dot([K, M, K.T]) + lamb * np.eye(m)
        b = np.linalg.multi_dot([K, H, K.T]) + 1e-6 * np.eye(m)

        a = (a + a.T) / 2
        b = (b + b.T) / 2

        try:
            w, V = scipy.linalg.eig(np.linalg.pinv(b) @ a)
        except:
            w, V = scipy.linalg.eig(a, b)
        w = np.real(w)
        V = np.real(V)
        ind = np.argsort(w)
        A = V[:, ind[:dim]]
        Z = A.T @ K
        Z = np.real(Z)
        Z /= np.linalg.norm(Z, axis=0) + 1e-12
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        Y_tar_pseudo = clf.predict(Xt_new)

    # Final prediction with last A
    if A is not None:
        Z = A.T @ X
        Z = np.real(Z)
        Z /= np.linalg.norm(Z, axis=0) + 1e-12
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        return sklearn.metrics.accuracy_score(Yt, clf.predict(Xt_new)) * 100

    return 0.0


# ============== Grid Search with Parallel Execution ==============
def tune_pca_parallel(Xs, Ys, Xt, Yt, k_values, target_acc=None, workers=4, verbose=True):
    """Grid search for PCA with ThreadPoolExecutor (fast for simple methods)."""
    if verbose:
        print(f"  Tuning PCA: {len(k_values)} values with {workers} workers (ThreadPool)...")

    start_time = time.time()
    results = []

    def run_task(k):
        start = time.time()
        acc = run_pca(Xs, Ys, Xt, Yt, k)
        runtime = time.time() - start
        return (k, acc, runtime)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_task, k): k for k in k_values}
        with tqdm(total=len(k_values), desc="  PCA", leave=False) as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

    runtime = time.time() - start_time
    avg_time = runtime / len(k_values)

    if target_acc is not None:
        best_k, best_acc, best_time = min(results, key=lambda x: abs(x[1] - target_acc))
        diff = abs(best_acc - target_acc)
        if diff <= 1.5:
            if verbose:
                print(f"    Found within +/-1.5%: k={best_k}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%)")
        else:
            if verbose:
                print(f"    NOT within +/-1.5%, closest: k={best_k}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%, diff: {diff:.2f}%)")
    else:
        best_k, best_acc, best_time = max(results, key=lambda x: x[1])
        if verbose:
            print(f"    Best: k={best_k}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s")

    return {"k": best_k}, best_acc, avg_time


def tune_pca(Xs, Ys, Xt, Yt, k_values, target_acc=None, workers=1, verbose=True):
    """Grid search for PCA (sequential or parallel based on workers)."""
    if workers > 1:
        return tune_pca_parallel(Xs, Ys, Xt, Yt, k_values, target_acc, workers, verbose)

    results = []

    if verbose:
        print(f"  Tuning PCA: {len(k_values)} values...")

    for k in tqdm(k_values, desc="PCA", leave=False):
        start = time.time()
        acc = run_pca(Xs, Ys, Xt, Yt, k)
        runtime = time.time() - start
        results.append((k, acc, runtime))

    if target_acc is not None:
        best_k, best_acc, best_time = min(results, key=lambda x: abs(x[1] - target_acc))
        diff = abs(best_acc - target_acc)
        if diff <= 1.5:
            if verbose:
                print(f"    Found within +/-1.5%: k={best_k}, Acc={best_acc:.2f}%, AvgTime={best_time:.3f}s (target: {target_acc:.2f}%)")
        else:
            if verbose:
                print(f"    NOT within +/-1.5%, closest: k={best_k}, Acc={best_acc:.2f}%, AvgTime={best_time:.3f}s (target: {target_acc:.2f}%, diff: {diff:.2f}%)")
    else:
        best_k, best_acc, best_time = max(results, key=lambda x: x[1])
        if verbose:
            print(f"    Best: k={best_k}, Acc={best_acc:.2f}%, AvgTime={best_time:.3f}s")

    return {"k": best_k}, best_acc, best_time


def tune_gfk_parallel(Xs, Ys, Xt, Yt, k_values, target_acc=None, workers=4, verbose=True):
    """Grid search for GFK with ThreadPoolExecutor (fast for simple methods)."""
    if verbose:
        print(f"  Tuning GFK: {len(k_values)} values with {workers} workers (ThreadPool)...")

    start_time = time.time()
    results = []

    def run_task(k):
        start = time.time()
        acc = run_gfk(Xs, Ys, Xt, Yt, k)
        runtime = time.time() - start
        return (k, acc, runtime)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_task, k): k for k in k_values}
        with tqdm(total=len(k_values), desc="  GFK", leave=False) as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

    runtime = time.time() - start_time
    avg_time = runtime / len(k_values)

    if target_acc is not None:
        best_k, best_acc, _ = min(results, key=lambda x: abs(x[1] - target_acc))
        diff = abs(best_acc - target_acc)
        if diff <= 1.5:
            if verbose:
                print(f"    Found within +/-1.5%: k={best_k}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%)")
        else:
            if verbose:
                print(f"    NOT within +/-1.5%, closest: k={best_k}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%, diff: {diff:.2f}%)")
    else:
        best_k, best_acc, _ = max(results, key=lambda x: x[1])
        if verbose:
            print(f"    Best: k={best_k}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s")

    return {"k": best_k}, best_acc, avg_time


def tune_gfk(Xs, Ys, Xt, Yt, k_values, target_acc=None, workers=1, verbose=True):
    """Grid search for GFK (sequential or parallel based on workers)."""
    if workers > 1:
        return tune_gfk_parallel(Xs, Ys, Xt, Yt, k_values, target_acc, workers, verbose)

    results = []

    if verbose:
        print(f"  Tuning GFK: {len(k_values)} values...")

    for k in tqdm(k_values, desc="GFK", leave=False):
        start = time.time()
        acc = run_gfk(Xs, Ys, Xt, Yt, k)
        runtime = time.time() - start
        results.append((k, acc, runtime))

    if target_acc is not None:
        best_k, best_acc, best_time = min(results, key=lambda x: abs(x[1] - target_acc))
        diff = abs(best_acc - target_acc)
        if diff <= 1.5:
            if verbose:
                print(f"    Found within +/-1.5%: k={best_k}, Acc={best_acc:.2f}%, AvgTime={best_time:.3f}s (target: {target_acc:.2f}%)")
        else:
            if verbose:
                print(f"    NOT within +/-1.5%, closest: k={best_k}, Acc={best_acc:.2f}%, AvgTime={best_time:.3f}s (target: {target_acc:.2f}%, diff: {diff:.2f}%)")
    else:
        best_k, best_acc, best_time = max(results, key=lambda x: x[1])
        if verbose:
            print(f"    Best: k={best_k}, Acc={best_acc:.2f}%, AvgTime={best_time:.3f}s")

    return {"k": best_k}, best_acc, best_time


def tune_tca_parallel(Xs, Ys, Xt, Yt, k_values, lamb_values, target_acc=None, workers=4, verbose=True):
    """Grid search for TCA with ThreadPoolExecutor (fast for simple methods)."""
    tasks = [(k, lamb) for k in k_values for lamb in lamb_values]
    total = len(tasks)

    if verbose:
        print(f"  Tuning TCA: {total} combinations with {workers} workers (ThreadPool)...")

    start_time = time.time()
    results = []

    def run_task(task):
        k, lamb = task
        acc = run_tca(Xs, Ys, Xt, Yt, k, lamb)
        return ({"k": k, "lamb": lamb}, acc)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        with tqdm(total=total, desc="  TCA", leave=False) as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

    runtime = time.time() - start_time
    avg_time = runtime / total

    if target_acc is not None:
        best_params, best_acc = min(results, key=lambda x: abs(x[1] - target_acc))
        diff = abs(best_acc - target_acc)
        if diff <= 1.5:
            if verbose:
                print(f"    Found within +/-1.5%: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%)")
        else:
            if verbose:
                print(f"    NOT within +/-1.5%, closest: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%, diff: {diff:.2f}%)")
    else:
        best_params, best_acc = max(results, key=lambda x: x[1])
        if verbose:
            print(f"    Best: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s")

    return best_params, best_acc, runtime / (len(k_values) * len(lamb_values))


def tune_tca(Xs, Ys, Xt, Yt, k_values, lamb_values, target_acc=None, workers=1, verbose=True):
    """Grid search for TCA (sequential or parallel based on workers)."""
    if workers > 1:
        return tune_tca_parallel(Xs, Ys, Xt, Yt, k_values, lamb_values, target_acc, workers, verbose)

    start_time = time.time()
    results = []
    total = len(k_values) * len(lamb_values)

    if verbose:
        print(f"  Tuning TCA: {total} combinations...")

    for k in k_values:
        for lamb in tqdm(lamb_values, desc=f"TCA(k={k})", leave=False):
            acc = run_tca(Xs, Ys, Xt, Yt, k, lamb)
            results.append(({"k": k, "lamb": lamb}, acc))

    runtime = time.time() - start_time
    avg_time = runtime / total

    if target_acc is not None:
        best_params, best_acc = min(results, key=lambda x: abs(x[1] - target_acc))
        diff = abs(best_acc - target_acc)
        if diff <= 1.5:
            if verbose:
                print(f"    Found within +/-1.5%: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%)")
        else:
            if verbose:
                print(f"    NOT within +/-1.5%, closest: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%, diff: {diff:.2f}%)")
    else:
        best_params, best_acc = max(results, key=lambda x: x[1])
        if verbose:
            print(f"    Best: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s")

    return best_params, best_acc, runtime / total


def tune_tsl_parallel(Xs, Ys, Xt, Yt, k_values, lamb_values, target_acc=None, workers=4, verbose=True):
    """Grid search for TSL with ProcessPoolExecutor for true parallelism."""
    tasks = [(k, lamb) for k in k_values for lamb in lamb_values]
    total = len(tasks)

    if verbose:
        print(f"  Tuning TSL: {total} combinations with {workers} workers (ProcessPool)...")

    # Prepare tasks with data
    task_args = [(task, Xs, Ys, Xt, Yt) for task in tasks]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(_tsl_task, task_args), total=total, desc="  TSL", leave=False))

    runtime = time.time() - start_time
    avg_time = runtime / total

    if target_acc is not None:
        best_params, best_acc = min(results, key=lambda x: abs(x[1] - target_acc))
        diff = abs(best_acc - target_acc)
        if diff <= 1.5:
            if verbose:
                print(f"    Found within +/-1.5%: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%)")
        else:
            if verbose:
                print(f"    NOT within +/-1.5%, closest: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%, diff: {diff:.2f}%)")
    else:
        best_params, best_acc = max(results, key=lambda x: x[1])
        if verbose:
            print(f"    Best: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s")

    return best_params, best_acc, runtime / total


def tune_tsl(Xs, Ys, Xt, Yt, k_values, lamb_values, target_acc=None, workers=1, verbose=True):
    """Grid search for TSL (sequential or parallel based on workers)."""
    if workers > 1:
        return tune_tsl_parallel(Xs, Ys, Xt, Yt, k_values, lamb_values, target_acc, workers, verbose)

    start_time = time.time()
    results = []
    total = len(k_values) * len(lamb_values)

    if verbose:
        print(f"  Tuning TSL: {total} combinations...")

    for k in k_values:
        for lamb in tqdm(lamb_values, desc=f"TSL(k={k})", leave=False):
            acc = run_tsl(Xs, Ys, Xt, Yt, k, lamb)
            results.append(({"k": k, "lamb": lamb}, acc))

    runtime = time.time() - start_time
    avg_time = runtime / total

    if target_acc is not None:
        best_params, best_acc = min(results, key=lambda x: abs(x[1] - target_acc))
        diff = abs(best_acc - target_acc)
        if diff <= 1.5:
            if verbose:
                print(f"    Found within +/-1.5%: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%)")
        else:
            if verbose:
                print(f"    NOT within +/-1.5%, closest: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%, diff: {diff:.2f}%)")
    else:
        best_params, best_acc = max(results, key=lambda x: x[1])
        if verbose:
            print(f"    Best: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s")

    return best_params, best_acc, runtime / total


def tune_jda_parallel(Xs, Ys, Xt, Yt, k_values, lamb_values, target_acc=None, workers=4, verbose=True):
    """Grid search for JDA with ProcessPoolExecutor for true parallelism."""
    tasks = [(k, lamb) for k in k_values for lamb in lamb_values]
    total = len(tasks)

    if verbose:
        print(f"  Tuning JDA: {total} combinations with {workers} workers (ProcessPool)...")

    # Prepare tasks with data
    task_args = [(task, Xs, Ys, Xt, Yt) for task in tasks]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(_jda_task, task_args), total=total, desc="  JDA", leave=False))

    runtime = time.time() - start_time
    avg_time = runtime / total

    if target_acc is not None:
        best_params, best_acc = min(results, key=lambda x: abs(x[1] - target_acc))
        diff = abs(best_acc - target_acc)
        if diff <= 1.5:
            if verbose:
                print(f"    Found within +/-1.5%: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%)")
        else:
            if verbose:
                print(f"    NOT within +/-1.5%, closest: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%, diff: {diff:.2f}%)")
    else:
        best_params, best_acc = max(results, key=lambda x: x[1])
        if verbose:
            print(f"    Best: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s")

    return best_params, best_acc, runtime / total


def tune_jda(Xs, Ys, Xt, Yt, k_values, lamb_values, target_acc=None, workers=1, verbose=True):
    """Grid search for JDA (sequential or parallel based on workers)."""
    if workers > 1:
        return tune_jda_parallel(Xs, Ys, Xt, Yt, k_values, lamb_values, target_acc, workers, verbose)

    start_time = time.time()
    results = []
    total = len(k_values) * len(lamb_values)

    if verbose:
        print(f"  Tuning JDA: {total} combinations...")

    for k in k_values:
        for lamb in tqdm(lamb_values, desc=f"JDA(k={k})", leave=False):
            acc = run_jda(Xs, Ys, Xt, Yt, k, lamb, T=JDA_ITERS)
            results.append(({"k": k, "lamb": lamb}, acc))

    runtime = time.time() - start_time
    avg_time = runtime / total

    if target_acc is not None:
        best_params, best_acc = min(results, key=lambda x: abs(x[1] - target_acc))
        diff = abs(best_acc - target_acc)
        if diff <= 1.5:
            if verbose:
                print(f"    Found within +/-1.5%: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%)")
        else:
            if verbose:
                print(f"    NOT within +/-1.5%, closest: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s (target: {target_acc:.2f}%, diff: {diff:.2f}%)")
    else:
        best_params, best_acc = max(results, key=lambda x: x[1])
        if verbose:
            print(f"    Best: k={best_params['k']}, lamb={best_params['lamb']}, Acc={best_acc:.2f}%, AvgTime={avg_time:.3f}s")

    return best_params, best_acc, runtime / total


# ============== Main ==============
def main():
    parser = argparse.ArgumentParser(description="Parameter tuning for JDA methods")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset type: digit, coil, pie, surf")
    parser.add_argument("--src", type=str, required=True, help="Source domain name")
    parser.add_argument("--tar", type=str, required=True, help="Target domain name")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--methods", type=str, default="all", help="Methods: all or comma-separated (pca,gfk,tca,tsl,jda)")
    parser.add_argument("--compare-paper", action="store_true", help="Compare results with original paper")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file")
    parser.add_argument("--parallel", action="store_true", help="Run parameter search in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    verbose = True  # Enable verbose output

    print("="*60)
    print(f"Parameter Tuning: {args.src} -> {args.tar}")
    print(f"Dataset: {args.dataset}")
    print("="*60)

    # Load data
    print("\nLoading data...")
    Xs, Ys, Xt, Yt = load_preset_data(args.dataset, args.src, args.tar, args.data_dir)
    print(f"  Source: {Xs.shape}, Target: {Xt.shape}")

    # Determine lambda range based on dataset
    if args.dataset == "surf":
        lamb_values = [0.01, 0.1, 1.0, 10.0]  # Skip 100 for speed
    else:
        lamb_values = LAMBDA_VALUES

    # Parse methods
    if args.methods == "all":
        methods = ["pca", "gfk", "tca", "tsl", "jda"]
    else:
        methods = [m.strip().lower() for m in args.methods.split(',')]

    results = {}
    start_time = time.time()

    # Get paper target accuracies if comparing
    paper_key = (args.dataset, args.src, args.tar)
    paper_data = PAPER_RESULTS.get(paper_key, {})

    # Determine number of workers for internal parallelization
    # When parallel=True, use all workers for parameter search within each method
    # Methods run sequentially to avoid mixed output
    use_workers = args.workers if args.parallel else 1

    # Run methods sequentially (to avoid mixed output)
    # Each method's internal parameter search uses parallelization when workers > 1
    for method in methods:
        target_acc = None
        if args.compare_paper and paper_data:
            target_acc = paper_data.get(method.upper(), None)

        if method == "nn":
            if verbose:
                print(f"\n  Running NN (baseline)...")
            start = time.time()
            acc = run_nn(Xs, Ys, Xt, Yt)
            runtime = time.time() - start
            results["NN"] = {"params": {}, "acc": acc, "runtime": runtime}
            if verbose:
                print(f"    NN: {acc:.2f}%, AvgTime={avg_time:.3f}s")
            continue

        if verbose:
            if use_workers > 1:
                print(f"\n  Tuning {method.upper()} with {use_workers} workers (parallel parameter search)...")
            else:
                print(f"\n  Tuning {method.upper()} (sequential)...")

        try:
            if method == "pca":
                params, acc, runtime = tune_pca(Xs, Ys, Xt, Yt, K_VALUES_LARGE, target_acc=target_acc, workers=use_workers)
                results["PCA"] = {"params": params, "acc": acc, "runtime": runtime}
            elif method == "gfk":
                params, acc, runtime = tune_gfk(Xs, Ys, Xt, Yt, K_VALUES_LARGE, target_acc=target_acc, workers=use_workers)
                results["GFK"] = {"params": params, "acc": acc, "runtime": runtime}
            elif method == "tca":
                params, acc, runtime = tune_tca(Xs, Ys, Xt, Yt, K_VALUES_LARGE, lamb_values, target_acc=target_acc, workers=use_workers)
                results["TCA"] = {"params": params, "acc": acc, "runtime": runtime}
            elif method == "tsl":
                params, acc, runtime = tune_tsl(Xs, Ys, Xt, Yt, K_VALUES_SMALL, lamb_values, target_acc=target_acc, workers=use_workers)
                results["TSL"] = {"params": params, "acc": acc, "runtime": runtime}
            elif method == "jda":
                params, acc, runtime = tune_jda(Xs, Ys, Xt, Yt, K_VALUES_SMALL, lamb_values, target_acc=target_acc, workers=use_workers)
                results["JDA"] = {"params": params, "acc": acc, "runtime": runtime}
        except Exception as e:
            print(f"Error in {method}: {e}")
            import traceback
            traceback.print_exc()
            results[method.upper()] = {"params": {}, "acc": 0, "runtime": 0, "error": str(e)}

    total_time = time.time() - start_time

    # Print results
    print("\n" + "="*60)
    if args.compare_paper and paper_data:
        print("Tuning Results (Finding Parameters Closest to Paper)")
    else:
        print("Tuning Results (Finding Best Parameters for Maximum Accuracy)")
    print("="*60)

    # Get paper results for comparison
    paper_key = (args.dataset, args.src, args.tar)
    paper_data = PAPER_RESULTS.get(paper_key, {})

    if args.compare_paper and paper_data:
        print(f"{'Method':<8} {'k':<6} {'lamb':<8} {'Ours':<12} {'Paper':<10} {'Diff':<10} {'Time':<10}")
        print("-"*76)

        for method, data in results.items():
            k = data["params"].get("k", "-")
            lamb = data["params"].get("lamb", "-")
            our_acc = data["acc"]
            runtime = data.get("runtime", 0)
            paper_acc = paper_data.get(method, "-")

            if paper_acc != "-":
                diff = our_acc - paper_acc
                print(f"{method:<8} {str(k):<6} {str(lamb):<8} {our_acc:>8.2f}% {paper_acc:>6.2f}% {diff:>+6.2f}% {runtime:>8.1f}s")
            else:
                print(f"{method:<8} {str(k):<6} {str(lamb):<8} {our_acc:>8.2f}% {'N/A':<10} {'':>6} {runtime:>8.1f}s")
    else:
        print(f"{'Method':<8} {'Best k':<10} {'Best lamb':<10} {'Accuracy':<12} {'Time':<10}")
        print("-"*60)
        for method, data in results.items():
            k = data["params"].get("k", "-")
            lamb = data["params"].get("lamb", "-")
            runtime = data.get("runtime", 0)
            print(f"{method:<8} {str(k):<10} {str(lamb):<10} {data['acc']:>8.2f}% {runtime:>8.1f}s")

    print("-"*76)
    print(f"Total time: {total_time:.2f}s")

    # Save to CSV file
    if args.output:
        import csv
        file_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
        with open(args.output, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Task", "Dataset", "Method", "k", "lambda", "Accuracy", "Paper_Acc", "Diff", "Runtime"])
            task = f"{args.src} -> {args.tar}"
            for method, data in results.items():
                k = data["params"].get("k", "")
                lamb = data["params"].get("lamb", "")
                runtime = data.get("runtime", 0)
                our_acc = data["acc"]
                paper_acc = paper_data.get(method, "-")
                diff = our_acc - paper_acc if paper_acc != "-" else ""
                writer.writerow([task, args.dataset, method, k, lamb, f"{our_acc:.2f}", paper_acc, f"{diff:.2f}" if diff != "" else "", f"{runtime:.1f}"])

    return results


if __name__ == "__main__":
    main()
