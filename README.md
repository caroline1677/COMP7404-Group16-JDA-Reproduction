# JDA Comparison Framework

A comprehensive framework for comparing transfer learning methods in domain adaptation tasks. Implements NN, PCA, TCA, GFK, TSL, and JDA algorithms with proper implementations based on the original papers.

## Overview

This framework reproduces experiments from the JDA (Joint Distribution Adaptation) paper and provides a unified interface for comparing different transfer learning methods.

### Implemented Methods

| Method | Full Name | Reference |
|--------|-----------|------------|
| NN | Nearest Neighbor | Baseline |
| PCA | Principal Component Analysis | Baseline |
| TCA | Transfer Component Analysis | Pan et al., TNN 2011 |
| GFK | Geodesic Flow Kernel | Gong et al., CVPR 2012 |
| TSL | Transfer Subspace Learning | Si et al., TKDE 2010 |
| JDA | Joint Distribution Adaptation | Long et al., ICCV 2013 |

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/jda-comparison.git
cd jda-comparison

# Install dependencies with UV
uv sync

# Or install in development mode
uv pip install -e .
```

### Using pip

```bash
pip install numpy scipy scikit-learn
```

## Quick Test

Verify your installation by running a simple test:

```bash
python jda_comparison.py --dataset digit --src USPS --tar MNIST
```

Expected output:
```
============================================================
Transfer Learning Comparison: USPS -> MNIST
Dataset: digit, Dim: 100, Lambda: 0.1, JDA Iter: 10, TSL Iter: 10
============================================================
| Method | Accuracy | Runtime (s) |
|--------|----------|-------------|
| NN     |   64.44% |       0.123 |
| PCA    |   65.06% |       0.234 |
| GFK    |   31.89% |       0.456 |
| TCA    |   58.11% |       0.789 |
| TSL    |   58.94% |       1.234 |
| JDA    |   72.44% |       2.345 |
```

## Dataset Structure

Place your data files in the following structure:

```
jda_project/
├── data/
│   ├── digit/
│   │   └── MNIST_vs_USPS.mat
│   ├── coil/
│   │   ├── COIL_1.mat
│   │   └── COIL_2.mat
│   ├── pie/
│   │   ├── PIE1.mat
│   │   ├── PIE2.mat
│   │   ├── PIE3.mat
│   │   ├── PIE4.mat
│   │   └── PIE5.mat
│   └── surf/
│       ├── amazon_zscore_SURF_L10.mat
│       ├── Caltech10_zscore_SURF_L10.mat
│       ├── dslr_zscore_SURF_L10.mat
│       └── webcam_zscore_SURF_L10.mat
```

## Usage

### Preset Dataset Mode (Recommended for Standard Datasets)

Run a single experiment using preset datasets:

```bash
python jda_comparison.py --dataset digit --src USPS --tar MNIST
python jda_comparison.py --dataset coil --src COIL1 --tar COIL2
python jda_comparison.py --dataset pie --src PIE1 --tar PIE4
python jda_comparison.py --dataset surf --src webcam --tar dslr
```

### Custom Data Mode (For Your Own Data)

Use your own .mat files by specifying file paths and variable names:

```bash
python jda_comparison.py \
    --src-file data/source.mat --src-feat X --src-label y \
    --tar-file data/target.mat --tar-feat X --tar-label y \
    --dim 100 --lamb 0.1
```

### Command Line Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| **Data Input (choose one mode)** | | | |
| `--dataset` | str | None | Dataset type: digit, coil, pie, surf |
| `--src` | str | None | Source domain name (with --dataset) |
| `--tar` | str | None | Target domain name (with --dataset) |
| `--data-dir` | str | data | Path to data directory |
| `--src-file` | str | None | Path to source .mat file (custom mode) |
| `--src-feat` | str | None | Variable name for source features |
| `--src-label` | str | None | Variable name for source labels |
| `--tar-file` | str | None | Path to target .mat file (custom mode) |
| `--tar-feat` | str | None | Variable name for target features |
| `--tar-label` | str | None | Variable name for target labels |
| **Method Parameters** | | | |
| `--dim` | int | 100 | Subspace dimensionality |
| `--lamb` | float | 0.1 | Regularization (TCA/TSL/JDA) |
| `--iter` | int | 10 | Default iterations for JDA and TSL |
| `--jda-iter` | int | 10 | Iterations for JDA specifically |
| `--tsl-iter` | int | 10 | Iterations for TSL specifically |
| **Method-Specific Parameters** | | | | Override per-method settings |
| `--pca-dim` | int | dim | Dimensionality for PCA |
| `--gfk-dim` | int | dim | Dimensionality for GFK |
| `--tca-dim` | int | dim | Dimensionality for TCA |
| `--tca-lamb` | float | lamb | Regularization for TCA |
| `--tsl-dim` | int | dim | Dimensionality for TSL |
| `--tsl-lamb` | float | lamb | Regularization for TSL |
| `--jda-dim` | int | dim | Dimensionality for JDA |
| `--jda-lamb` | float | lamb | Regularization for JDA |
| **Output Options** | | | |
| `--methods` | str | all | Methods: 'all' or comma-separated (nn,pca,tca,gfk,tsl,jda) |
| `--parallel` | flag | False | Run methods in parallel (multi-threaded) |
| `--workers` | int | 4 | Number of parallel workers (default: 4) |
| `--output` | str | None | Save results to CSV file |

### Examples

Run with custom parameters:
```bash
python jda_comparison.py --dataset surf --src webcam --tar dslr --dim 50 --lamb 1.0 --jda-iter 15
```

Run with method-specific parameters:
```bash
python jda_comparison.py --dataset digit --src USPS --tar MNIST \
    --methods pca,gfk,tca,tsl,jda \
    --pca-dim 40 \
    --gfk-dim 60 \
    --tca-dim 50 --tca-lamb 0.1 \
    --tsl-dim 80 --tsl-lamb 1.0 \
    --jda-dim 100 --jda-lamb 0.1
```

Run only specific methods:
```bash
python jda_comparison.py --dataset coil --src COIL1 --tar COIL2 --methods nn,pca,jda
```

Run in parallel (recommended for large datasets):
```bash
python jda_comparison.py --dataset pie --src PIE1 --tar PIE4 --parallel --workers 4
```

Save results to CSV:
```bash
python jda_comparison.py --dataset pie --src PIE1 --tar PIE4 --output results.csv
```

## Batch Processing

Run multiple experiments from a config file:

```bash
python run_experiments.py experiments_config.csv full_results.csv
```

### Configuration File Format

#### Preset Mode

```csv
dataset,src,tar,dim,lamb,iter
digit,USPS,MNIST,100,0.1,10
coil,COIL1,COIL2,100,0.1,10
pie,PIE1,PIE4,100,0.1,10
surf,webcam,dslr,100,1.0,10
```

#### Custom Mode

```csv
src_file,src_feat,src_label,tar_file,tar_feat,tar_label,dim,lamb,iter,jda_iter,tsl_iter
data/source1.mat,X,Y,data/target1.mat,X,Y,100,0.1,10,10,10
data/source2.mat,features,labels,data/target2.mat,features,labels,100,0.1,15,15,5
```

### Output Format

Results are saved in CSV format with accuracy and runtime for each method:

```csv
Task,NN_Acc,NN_Time,PCA_Acc,PCA_Time,GFK_Acc,GFK_Time,TCA_Acc,TCA_Time,TSL_Acc,TSL_Time,JDA_Acc,JDA_Time
USPS -> MNIST,64.44,0.123,65.06,0.234,31.89,0.456,58.11,0.789,58.94,1.234,72.44,2.345
```

## Parameter Tuning

Perform grid search to find optimal hyperparameters for each method:

```bash
# Tune all methods
python tune_parameters.py --dataset digit --src USPS --tar MNIST

# Tune specific methods
python tune_parameters.py --dataset digit --src USPS --tar MNIST --methods pca,gfk,tca

# Compare with original paper results
python tune_parameters.py --dataset digit --src USPS --tar MNIST --compare-paper

# Run with parallel (faster for large parameter search)
python tune_parameters.py --dataset digit --src USPS --tar MNIST --parallel --workers 4
```

### Search Space (as per paper)

| Method | k Range | λ Values |
|--------|---------|----------|
| PCA | 10,20,30,...,200 | - |
| GFK | 10,20,30,...,200 | - |
| TCA | 10,20,30,...,200 | 0.01, 0.1, 1, 10, 100 |
| TSL | 10,20,30,...,200 | 0.01, 0.1, 1, 10, 100 |
| JDA | 10,20,30,...,200 | 0.01, 0.1, 1, 10, 100 |

### Output Example

```
Method   k      λ        Ours       Paper      Diff
----------------------------------------------------------------------
PCA      40     -         66.78%   44.95%   +21.83%
GFK      60     -         65.00%   46.45%   +18.55%
TCA      50     0.1       58.89%   51.05%    +7.84%
TSL      70     1.0       59.12%   53.75%    +5.37%
JDA      100    0.1       72.44%   59.65%   +12.79%
```

**NOTE**: Using target domain labels for parameter tuning is only acceptable for research reproduction. In real-world scenarios, this is not feasible as target domain labels are unknown.

## Method Details

### 1. Nearest Neighbor (NN)
Baseline classifier using 1-NN with Euclidean distance.

### 2. Principal Component Analysis (PCA)
Dimensionality reduction using PCA followed by 1-NN classification.

### 3. Transfer Component Analysis (TCA)
- Adapts only the marginal distribution P(x)
- Uses MMD (Maximum Mean Discrepancy) to measure distribution distance
- Learns transfer components in RKHS

### 4. Geodesic Flow Kernel (GFK)
- Manifold-based domain adaptation
- Computes geodesic flow between source and target subspaces
- Uses kernel distance for classification: D = diag(K_ss) + diag(K_tt) - 2*K_st

### 5. Transfer Subspace Learning (TSL)
- Uses Bregman divergence (LogDet) for distribution adaptation
- Iteratively optimizes subspace to minimize divergence
- Maximizes variance while minimizing distribution mismatch
- Uses `--tsl-iter` for internal optimization iterations

### 6. Joint Distribution Adaptation (JDA)
- Adapts both marginal P(x) and conditional Q(y|x) distributions
- Iteratively refines target pseudo-labels
- Combines MMD for both marginal and conditional distributions
- Uses `--jda-iter` for pseudo-label refinement iterations

## Data Preprocessing

### Digit Datasets (USPS, MNIST)
- Raw pixel values (no normalization)
- Feature dimension varies by dataset

### COIL Dataset
- Raw image features
- 20 object classes, 72 images per object

### PIE Dataset
- Normalized to [0,1] by dividing by 255
- Face images from different poses

### Office SURF Dataset
- Pre-extracted SURF features
- Already z-score standardized

## Hyperparameters

| Dataset | Lambda | Notes |
|---------|--------|-------|
| digit (USPS->MNIST) | 0.1 | Raw pixels work best |
| coil (COIL1->COIL2) | 0.1 | Raw pixels work best |
| pie (PIE->PIE) | 0.1 | Normalize to [0,1] |
| surf (Office) | 1.0 | SURF already standardized |

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --dev

# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

### Code Style

This project follows PEP 8 guidelines:

```bash
# Format code
black .

# Lint code
flake8 .
```


