from flask import Flask, render_template, request, send_from_directory
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_ROOT, "static")

# ---------------------------- DATA GENERATOR ----------------------------
def generate_data(dist, size, p):
    if dist == "uniform":
        return np.random.uniform(float(p["low"]), float(p["high"]), size)
    elif dist == "normal":
        return np.random.normal(float(p["mean"]), float(p["std"]), size)
    elif dist == "binomial":
        return np.random.binomial(int(p["n"]), float(p["p"]), size)
    elif dist == "poisson":
        return np.random.poisson(float(p["lam"]), size)
    else:
        return np.random.uniform(0, 1, size)

# ---------------------------- SAVE GRAPH ----------------------------
def save_histogram(data, title, out_path):
    plt.clf()
    plt.style.use("ggplot")  # safe style
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(data, bins=30, edgecolor='white', alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

# ---------------------------- STATS ----------------------------
def compute_stats(arr):
    return {
        "Count": len(arr),
        "Mean": round(float(np.mean(arr)), 4),
        "Median": round(float(np.median(arr)), 4),
        "Variance": round(float(np.var(arr)), 4),
        "StdDev": round(float(np.std(arr)), 4),
        "Min": round(float(np.min(arr)), 4),
        "Max": round(float(np.max(arr)), 4),
    }

# ---------------------------- MAIN ROUTE ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    graph = None
    graph2 = None
    stats = None
    stats2 = None
    compare_mode = False

    if request.method == "POST":
        mode = request.form.get("mode")
        compare_mode = (mode == "compare")

        # ---------------- Distribution 1 ----------------
        dist1 = request.form.get("dist1", "normal")
        size1 = int(request.form.get("size1", 500))

        params1 = {
            "low": request.form.get("low1", 0),
            "high": request.form.get("high1", 1),
            "mean": request.form.get("mean1", 0),
            "std": request.form.get("std1", 1),
            "n": request.form.get("n1", 10),
            "p": request.form.get("p1", 0.5),
            "lam": request.form.get("lam1", 4)
        }

        data1 = generate_data(dist1, size1, params1)
        stats = compute_stats(data1)

        graph_file = f"g1_{int(datetime.now().timestamp())}.png"
        save_histogram(data1, f"{dist1.capitalize()} Distribution (n={size1})",
                       os.path.join(STATIC_DIR, graph_file))
        graph = f"/static/{graph_file}"

        # ---------------- Distribution 2 ----------------
        if compare_mode:
            dist2 = request.form.get("dist2", "uniform")
            size2 = int(request.form.get("size2", 500))

            params2 = {
                "low": request.form.get("low2", 0),
                "high": request.form.get("high2", 1),
                "mean": request.form.get("mean2", 0),
                "std": request.form.get("std2", 1),
                "n": request.form.get("n2", 10),
                "p": request.form.get("p2", 0.5),
                "lam": request.form.get("lam2", 4)
            }

            data2 = generate_data(dist2, size2, params2)
            stats2 = compute_stats(data2)

            graph_file2 = f"g2_{int(datetime.now().timestamp())}.png"
            save_histogram(data2, f"{dist2.capitalize()} Distribution (n={size2})",
                           os.path.join(STATIC_DIR, graph_file2))
            graph2 = f"/static/{graph_file2}"

    return render_template("index.html",
                           graph=graph, graph2=graph2,
                           stats=stats, stats2=stats2,
                           compare_mode=compare_mode)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)

