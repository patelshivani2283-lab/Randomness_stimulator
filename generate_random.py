

import os
import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats     # for mode (works well with continuous data fallback)

#  Configuration 
DATA_DIR = "data"
REPORT_DIR = "reports"
PLOT_DIR = os.path.join(REPORT_DIR, "plots")

for d in (DATA_DIR, REPORT_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)


#  Helper functions 
def save_dataframe_csv(df: pd.DataFrame, filename: str):
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved raw data: {path}")
    return path


def save_report(df_stats: pd.DataFrame, filename: str):
    path = os.path.join(REPORT_DIR, filename)
    df_stats.to_csv(path, index=False)
    print(f"Saved report: {path}")
    return path


def save_histogram(values, title: str, filename: str, bins: int = 30):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    outpath = os.path.join(PLOT_DIR, filename)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved histogram: {outpath}")
    return outpath


def calculate_statistics(series: pd.Series):
  
    s = series.dropna()
    if s.empty:
        return {}

    # Compute mode - handle continuous by rounding fallback
    try:
        mode_val, mode_count = stats.mode(s, nan_policy='omit')
        mode_val = float(mode_val[0]) if len(mode_val) > 0 else None
    except Exception:
        mode_val = None

    stats_dict = {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "mode": mode_val,
        "min": float(s.min()),
        "max": float(s.max()),
        "range": float(s.max() - s.min()),
        "variance": float(s.var(ddof=0)),
        "std_dev": float(s.std(ddof=0)),
    }
    return stats_dict

#  Generators 
def generate_uniform(n: int, low: float = 0.0, high: float = 1.0):
    arr = np.random.uniform(low, high, size=n)
    return arr


def generate_normal(n: int, mu: float = 0.0, sigma: float = 1.0):
    arr = np.random.normal(loc=mu, scale=sigma, size=n)
    return arr


def generate_binomial(n: int, trials: int = 1, p: float = 0.5):
    arr = np.random.binomial(n=trials, p=p, size=n)
    return arr


def generate_poisson(n: int, lam: float = 1.0):
    arr = np.random.poisson(lam=lam, size=n)
    return arr


#  Main simulation logic 
def run_simulation(
    distribution: str,
    count: int,
    params: dict,
    sample_name: str = None,
    seed: int = None
):
    

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        print(f"Seed set to {seed} (for reproducibility)")

    dist = distribution.lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_name = sample_name or f"{dist}_{count}_{ts}"

    if dist == "uniform":
        low = float(params.get("low", 0.0))
        high = float(params.get("high", 1.0))
        arr = generate_uniform(count, low, high)
    elif dist == "normal":
        mu = float(params.get("mu", 0.0))
        sigma = float(params.get("sigma", 1.0))
        arr = generate_normal(count, mu, sigma)
    elif dist == "binomial":
        trials = int(params.get("trials", 1))
        p = float(params.get("p", 0.5))
        arr = generate_binomial(count, trials, p)
    elif dist == "poisson":
        lam = float(params.get("lam", 1.0))
        arr = generate_poisson(count, lam)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Build DataFrame
    df = pd.DataFrame({"value": arr})
    raw_filename = f"{sample_name}_raw.csv"
    save_dataframe_csv(df, raw_filename)

    # Calculate statistics
    stats_dict = calculate_statistics(df["value"])
    stats_df = pd.DataFrame([stats_dict])
    stats_filename = f"{sample_name}_stats.csv"
    save_report(stats_df, stats_filename)

    # Save histogram
    hist_title = f"{distribution.title()} Distribution (n={count})"
    hist_file = f"{sample_name}_hist.png"
    save_histogram(df["value"], hist_title, hist_file)

    print("\n--- Summary ---")
    for k, v in stats_dict.items():
        print(f"{k:12}: {v}")
    print("----------------\n")

    return {
        "raw_data_path": os.path.join(DATA_DIR, raw_filename),
        "stats_path": os.path.join(REPORT_DIR, stats_filename),
        "hist_path": os.path.join(PLOT_DIR, hist_file),
        "stats": stats_dict,
    }


#  CLI Menu 
def input_int(prompt_text, default=None, min_val=None):
    while True:
        try:
            raw = input(prompt_text).strip()
            if raw == "" and default is not None:
                return default
            val = int(raw)
            if min_val is not None and val < min_val:
                print(f"Please enter an integer >= {min_val}")
                continue
            return val
        except ValueError:
            print("Please enter a valid integer.")


def input_float(prompt_text, default=None, min_val=None):
    while True:
        try:
            raw = input(prompt_text).strip()
            if raw == "" and default is not None:
                return default
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"Please enter a value >= {min_val}")
                continue
            return val
        except ValueError:
            print("Please enter a valid number.")


def main_menu():
    print("=== Randomness Simulator ===")
    print("Choose distribution:")
    print("1) Uniform")
    print("2) Normal")
    print("3) Binomial")
    print("4) Poisson")
    print("5) All (generate one example of each)")
    print("0) Exit")

    choice = input_int("Select option [1-5,0]: ", default=1, min_val=0)
    if choice == 0:
        print("Exiting.")
        return

    if choice == 5:
        # default example for 'All'
        seed = input_int("Optional seed (blank for random): ", default=None)
        count = input_int("Number of samples (default 1000): ", default=1000, min_val=1)
        # uniform
        run_simulation("uniform", count, {"low": 0, "high": 1}, sample_name=f"uniform_{count}", seed=seed)
        # normal
        run_simulation("normal", count, {"mu": 0, "sigma": 1}, sample_name=f"normal_{count}", seed=seed)
        # binomial
        run_simulation("binomial", count, {"trials": 10, "p": 0.5}, sample_name=f"binomial_{count}", seed=seed)
        # poisson
        run_simulation("poisson", count, {"lam": 2.0}, sample_name=f"poisson_{count}", seed=seed)
        return

    # Single distribution chosen
    dist_map = {1: "uniform", 2: "normal", 3: "binomial", 4: "poisson"}
    dist = dist_map.get(choice)

    count = input_int("Number of samples to generate (default 1000): ", default=1000, min_val=1)
    seed = input_int("Optional integer seed for reproducibility (blank for random): ", default=None)

    params = {}
    if dist == "uniform":
        low = input_float("Uniform low (default 0.0): ", default=0.0)
        high = input_float("Uniform high (default 1.0): ", default=1.0)
        if high <= low:
            print("High must be > low. Swapping values.")
            low, high = min(low, high), max(low, high)
        params = {"low": low, "high": high}

    elif dist == "normal":
        mu = input_float("Normal mean (mu) (default 0.0): ", default=0.0)
        sigma = input_float("Normal std dev (sigma) (default 1.0): ", default=1.0, min_val=0.0)
        params = {"mu": mu, "sigma": sigma}

    elif dist == "binomial":
        trials = input_int("Binomial trials (n) (default 1): ", default=1, min_val=1)
        p = input_float("Binomial p (prob of success) (0-1) (default 0.5): ", default=0.5)
        p = min(max(p, 0.0), 1.0)
        params = {"trials": trials, "p": p}

    elif dist == "poisson":
        lam = input_float("Poisson lambda (mean rate) (default 1.0): ", default=1.0)
        params = {"lam": lam}

    # optional sample name
    sample_name = input("Optional sample name prefix (blank = auto): ").strip() or None

    res = run_simulation(dist, count, params, sample_name=sample_name, seed=seed)
    print("Done. Files created:")
    print(res["raw_data_path"])
    print(res["stats_path"])
    print(res["hist_path"])


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
