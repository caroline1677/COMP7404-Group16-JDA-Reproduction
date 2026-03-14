"""
Batch Experiment Runner for JDA Comparison Framework

This script reads experiment configurations from a CSV file and runs
multiple transfer learning experiments automatically.

Supports two modes:
1. Preset mode: Use --dataset with --src/--tar
2. Custom mode: Use --src-file/--tar-file with variable names

Usage:
    python run_experiments.py [config_csv] [output_csv]

Example:
    python run_experiments.py experiments_config.csv full_results.csv
"""

import argparse
import csv
import os
import subprocess
import sys
import time


# Default method order (as per paper)
DEFAULT_METHOD_ORDER = ["NN", "PCA", "GFK", "TCA", "TSL", "JDA"]


def run_single_experiment(args_list):
    """Run a single experiment using jda_comparison.py."""
    cmd = [sys.executable, "jda_comparison.py"] + args_list
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def parse_method_order(methods_str):
    """Parse method order from string like 'nn,pca,jda' or 'all'."""
    if methods_str.lower() == "all" or methods_str is None:
        return DEFAULT_METHOD_ORDER
    return [m.strip().upper() for m in methods_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Batch experiment runner for JDA comparison"
    )
    parser.add_argument("config", nargs="?", default="experiments_config.csv",
                        help="Input configuration CSV file (default: experiments_config.csv)")
    parser.add_argument("output", nargs="?", default="full_results.csv",
                        help="Output results CSV file (default: full_results.csv)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to data directory (default: data)")
    parser.add_argument("--methods", type=str, default="all",
                        help="Methods to run: 'all' or comma-separated list")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print verbose output")

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found!")
        print(f"\nPlease create a config file with one of these formats:")
        print("\n=== Preset Mode (recommended for standard datasets) ===")
        print("dataset,src,tar,dim,lamb,iter")
        print("digit,USPS,MNIST,100,0.1,10")
        print("coil,COIL1,COIL2,100,0.1,10")
        print("pie,PIE05,PIE27,100,0.1,10")
        print("surf,webcam,dslr,100,1.0,10")

        print("\n=== Custom Mode (for your own data) ===")
        print("src_file,src_feat,src_label,tar_file,tar_feat,tar_label,dim,lamb,iter")
        print("data/source.mat,X,Y,data/target.mat,X,Y,100,0.1,10")

        sys.exit(1)

    # Read config file
    experiments = []
    with open(args.config, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            experiments.append(row)

    print(f"Loaded {len(experiments)} experiment configurations")
    print(f"Output will be saved to: {args.output}")
    print("-" * 60)

    # Determine method order
    method_order = parse_method_order(args.methods)

    # Clear/create output file with header
    header = ["Task"]
    for m in method_order:
        header.extend([f"{m}_Acc", f"{m}_Time"])
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Run experiments
    for i, exp in enumerate(experiments):
        # Determine mode: preset or custom
        if 'dataset' in exp and exp.get('dataset'):
            # Preset mode
            dataset = exp.get('dataset', '')
            src = exp.get('src', '')
            tar = exp.get('tar', '')
            task_name = f"{src} -> {tar}"

            cmd_args = [
                "--dataset", dataset,
                "--src", src,
                "--tar", tar,
            ]
        elif 'src_file' in exp and exp.get('src_file'):
            # Custom mode
            src_file = exp.get('src_file', '')
            src_feat = exp.get('src_feat', 'X')
            src_label = exp.get('src_label', 'Y')
            tar_file = exp.get('tar_file', '')
            tar_feat = exp.get('tar_feat', 'X')
            tar_label = exp.get('tar_label', 'Y')
            task_name = f"{os.path.basename(src_file)} -> {os.path.basename(tar_file)}"

            cmd_args = [
                "--src-file", src_file,
                "--src-feat", src_feat,
                "--src-label", src_label,
                "--tar-file", tar_file,
                "--tar-feat", tar_feat,
                "--tar-label", tar_label,
            ]
        else:
            print(f"\n[{i+1}/{len(experiments)}] Skipping invalid config: {exp}")
            continue

        # Common parameters
        dim = exp.get('dim', '100')
        lamb = exp.get('lamb', '0.1')
        iter_val = exp.get('iter', '10')
        jda_iter = exp.get('jda_iter', iter_val)
        tsl_iter = exp.get('tsl_iter', iter_val)

        cmd_args.extend([
            "--dim", str(dim),
            "--lamb", str(lamb),
            "--iter", str(iter_val),
            "--data-dir", args.data_dir,
            "--methods", args.methods
        ])

        # Add method-specific iterations if present in config
        if 'jda_iter' in exp:
            cmd_args.extend(["--jda-iter", str(jda_iter)])
        if 'tsl_iter' in exp:
            cmd_args.extend(["--tsl-iter", str(tsl_iter)])

        print(f"\n[{i+1}/{len(experiments)}] {task_name}")

        if args.verbose:
            print(f"    dim={dim}, lamb={lamb}, iter={iter_val}")

        # Run experiment
        start_time = time.time()
        returncode, stdout, stderr = run_single_experiment(cmd_args)
        elapsed = time.time() - start_time

        if args.verbose:
            if stderr:
                print(f"    Error: {stderr[:200]}")
            else:
                print(f"    Completed in {elapsed:.1f}s")

        # Parse results from output
        results = {m: ('', '') for m in method_order}

        # Extract results from markdown table in output
        in_table = False
        for line in stdout.split('\n'):
            if '| Method |' in line:
                in_table = True
                continue
            if in_table and '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4 and parts[1] in method_order:
                    method = parts[1]
                    acc = parts[2].replace('%', '')
                    runtime = parts[3]
                    results[method] = (acc, runtime)

        # Write results to CSV
        row = [task_name]
        for m in method_order:
            acc, runtime = results[m]
            if acc:  # Only add if we have results
                row.extend([acc, runtime])

        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Print summary
        acc_strs = []
        for m in method_order:
            acc, _ = results[m]
            if acc:
                acc_strs.append(f"{m}={acc}%")
        print(f"    Results: {', '.join(acc_strs)}")

    print("\n" + "=" * 60)
    print(f"All experiments completed!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
