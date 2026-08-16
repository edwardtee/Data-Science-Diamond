import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Navigate up 2 levels from testing.py to find the project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

METRICS_DIR = ROOT_DIR / "src" / "saved_models" / "metrics.json"
OUT_DIR = ROOT_DIR / "output"

# ---- Load data ----
with open(METRICS_DIR, "r") as f:
    metrics = json.load(f)

models = list(metrics.keys())
r2 = [metrics[m]["r2"] for m in models]
mae = [metrics[m]["mae"] for m in models]
rmse = [metrics[m]["rmse"] for m in models]

# Shorten labels for cleaner x-axis
short_names = [m.replace(" Regression", "").replace(" (Pro)", "") for m in models]

# ---- Colors: highlight best model per metric ----
def bar_colors(values, higher_is_better):
    best_idx = values.index(max(values) if higher_is_better else min(values))
    return ["#2E7D32" if i == best_idx else "#B0BEC5" for i in range(len(values))]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold")

metric_data = [
    ("R² Score", r2, True, axes[0]),
    ("MAE", mae, False, axes[1]),
    ("RMSE", rmse, False, axes[2]),
]

for title, values, higher_better, ax in metric_data:
    colors = bar_colors(values, higher_better)
    bars = ax.bar(short_names, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}" if val < 10 else f"{val:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT_DIR / "model_comparison.png", dpi=200, bbox_inches="tight")
print("Saved model_comparison.png")