# prepare_mnist_usps.py
# ------------------------------------
# 加载 MNIST & USPS，并按 JDA 习惯预处理：
# - MNIST：torchvision 加载，resize 16x16，可选随机采样子集
# - USPS：从本地 .h5 文件读取，可选随机采样子集
# - 两域一起做 zero-mean, unit-variance 标准化
# - 仅保存为 .mat 文件，供后续 JDA / MATLAB 使用
# 依赖：torch torchvision numpy scipy h5py
# ------------------------------------

import os
import numpy as np
from scipy.io import savemat
import h5py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ========= 可调参数（随机采样与路径） =========
SEED = 42
USE_SUBSET = True          # 是否做随机子集采样（与论文常用设定一致）
MNIST_SUBSET_SIZE = 2000   # MNIST 采样数量，None 表示用全部
USPS_SUBSET_SIZE = 1800    # USPS 采样数量，None 表示用全部
USPS_H5_PATH = r"D:\Desktop\mdasc\7404\COMP7404-Group16-JDA-Reproduction-main\data\usps.h5"
DATA_ROOT = "./data"
SAVE_DIR = "./prepared_mnist_usps"
MAT_FILENAME = "mnist_usps.mat"


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mnist_16x16(root: str = "./data", download: bool = True):
    """
    加载 MNIST，统一 resize 到 16x16，返回：
        X_mnist: [N, 256] float32
        y_mnist: [N] int64
    像素缩放到 [0, 1]。
    """
    tf = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),          # [0,1]
    ])

    mnist_train = datasets.MNIST(root=root, train=True, transform=tf, download=download)
    mnist_test = datasets.MNIST(root=root, train=False, transform=tf, download=download)

    loader_train = DataLoader(mnist_train, batch_size=1024, shuffle=False)
    loader_test = DataLoader(mnist_test, batch_size=1024, shuffle=False)

    xs, ys = [], []

    for imgs, labels in loader_train:
        # imgs: [B,1,16,16]
        b = imgs.size(0)
        xs.append(imgs.view(b, -1))  # [B,256]
        ys.append(labels)

    for imgs, labels in loader_test:
        b = imgs.size(0)
        xs.append(imgs.view(b, -1))
        ys.append(labels)

    X = torch.cat(xs, dim=0).numpy().astype(np.float32)   # [N,256], [0,1]
    y = torch.cat(ys, dim=0).numpy().astype(np.int64)

    return X, y


def _read_group_Xy(g):
    """从 h5 Group（如 train/test）里找特征和标签，返回 (X, y)。"""
    if isinstance(g, h5py.Dataset):
        return np.array(g, dtype=np.float32), None  # 仅特征，标签由外层提供
    keys = list(g.keys())
    for xname in ("X", "x", "data", "fea", "features", "images", "img"):
        if xname in keys:
            X = np.array(g[xname], dtype=np.float32)
            break
    else:
        raise KeyError(f"该组内未找到特征，键: {keys}")
    for yname in ("y", "Y", "labels", "label", "gnd", "target", "t"):
        if yname in keys:
            y = np.array(g[yname]).reshape(-1).astype(np.int64)
            return X, y
    return X, None  # 仅特征


def load_usps_from_h5(h5_path: str):
    """
    从本地 usps.h5 读取 USPS 数据。
    支持两种结构：
      1）顶层为 train / test 两组，每组内有特征和标签；
      2）顶层直接为 X/x/data 与 y/Y/labels。
    返回：
        X: [N, 256] float32
        y: [N] int64，类别 0..9
    """
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())

        if "train" in keys and "test" in keys:
            X_list, y_list = [], []
            for part in ("train", "test"):
                g = f[part]
                if isinstance(g, h5py.Dataset):
                    Xi = np.array(g, dtype=np.float32)
                    label_key = f"{part}_labels" if f"{part}_labels" in keys else f"{part}_label"
                    yi = np.array(f[label_key]).reshape(-1).astype(np.int64) if label_key in keys else None
                else:
                    Xi, yi = _read_group_Xy(g)
                if yi is None:
                    label_key = f"{part}_labels" if f"{part}_labels" in keys else f"{part}_label"
                    if label_key in keys:
                        yi = np.array(f[label_key]).reshape(-1).astype(np.int64)
                    else:
                        raise KeyError(f"未找到 {part} 的标签，顶层键: {keys}，组内键: {list(g.keys())}")
                X_list.append(Xi)
                y_list.append(yi)
            X = np.vstack(X_list)
            y = np.concatenate(y_list)
        else:
            for name in ("X", "x", "data", "fea", "features"):
                if name in keys:
                    X = np.array(f[name], dtype=np.float32)
                    break
            else:
                raise KeyError(f"USPS .h5 中未找到特征键，当前键: {keys}")
            for name in ("y", "Y", "labels", "label", "gnd"):
                if name in keys:
                    y = np.array(f[name]).reshape(-1).astype(np.int64)
                    break
            else:
                raise KeyError(f"USPS .h5 中未找到标签键，当前键: {keys}")

    # 若是 [256, N]，转成 [N, 256]
    if X.shape[0] == 256 and X.shape[1] != 256:
        X = X.T
    # 若有多维特征如 [N, 16, 16]，展平为 [N, 256]
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    # 若标签是 1..10，转为 0..9
    if y.min() >= 1:
        y = y - 1
    # 若特征值域不在 [0,1]，归一化到 [0,1]
    if X.max() > 1.0 or X.min() < 0.0:
        X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    return X, y


def random_subset(X: np.ndarray, y: np.ndarray, size: int, seed: int):
    """在 (X, y) 上随机无放回采样 size 条，用 seed 保证可复现。"""
    n = X.shape[0]
    if size is None or size >= n:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=size, replace=False)
    return X[idx], y[idx]


def standardize_joint(X_src: np.ndarray, X_tgt: np.ndarray):
    """
    在“源+目标”的所有样本上一起计算均值/方差，
    然后对两个域分别做 (x - mean) / std。
    这样保证两个域在同一标准化空间中，对 JDA 比较友好。
    """
    X_all = np.vstack([X_src, X_tgt])          # [N_src+N_tgt, d]
    mean = X_all.mean(axis=0, keepdims=True)  # [1,d]
    std = X_all.std(axis=0, keepdims=True)    # [1,d]
    std[std < 1e-6] = 1.0                     # 防止除零

    X_src_std = (X_src - mean) / std
    X_tgt_std = (X_tgt - mean) / std

    return X_src_std.astype(np.float32), X_tgt_std.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def main():
    set_seed(SEED)

    os.makedirs(SAVE_DIR, exist_ok=True)
    mat_path = os.path.join(SAVE_DIR, MAT_FILENAME)

    print("Loading MNIST...")
    X_mnist, y_mnist = load_mnist_16x16(root=DATA_ROOT, download=True)
    print("  MNIST (full):", X_mnist.shape, y_mnist.shape)

    if USE_SUBSET and MNIST_SUBSET_SIZE is not None:
        X_mnist, y_mnist = random_subset(X_mnist, y_mnist, MNIST_SUBSET_SIZE, seed=SEED)
        print("  MNIST (subset):", X_mnist.shape, y_mnist.shape)

    print("Loading USPS from .h5...")
    X_usps, y_usps = load_usps_from_h5(USPS_H5_PATH)
    print("  USPS (full):", X_usps.shape, y_usps.shape)

    if USE_SUBSET and USPS_SUBSET_SIZE is not None:
        X_usps, y_usps = random_subset(X_usps, y_usps, USPS_SUBSET_SIZE, seed=SEED)
        print("  USPS (subset):", X_usps.shape, y_usps.shape)

    print("Standardizing jointly (zero-mean, unit-variance)...")
    X_mnist_std, X_usps_std, mean, std = standardize_joint(X_mnist, X_usps)

    savemat(mat_path, {
        "X_mnist": X_mnist_std,
        "y_mnist": y_mnist.astype(np.int32),
        "X_usps": X_usps_std,
        "y_usps": y_usps.astype(np.int32),
        "mean": mean,
        "std": std,
    })
    print("Saved .mat to:", mat_path)


if __name__ == "__main__":
    main()