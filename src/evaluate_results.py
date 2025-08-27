import pandas as pd
import matplotlib.pyplot as plt

# ========================
# Config
# ========================
files = {
    "Train (results.csv)": "results.csv",
    "Unseen Base (results_new.csv)": "results_new.csv",
    "Unseen Fine-tuned (results_finetuned.csv)": "results_finetuned.csv",
}

summary = {}

# ========================
# Compute averages
# ========================
for label, path in files.items():
    df = pd.read_csv(path)
    avg_cer = df["CER"].mean()
    avg_wer = df["WER"].mean()
    summary[label] = {"CER": avg_cer, "WER": avg_wer}

# Convert to DataFrame
summary_df = pd.DataFrame(summary).T
print(summary_df)

# Save summary CSV
summary_df.to_csv("model_performance_summary.csv")
print("✅ Summary saved to model_performance_summary.csv")

# ========================
# Plot bar chart
# ========================
ax = summary_df.plot(
    kind="bar",
    rot=25,
    figsize=(10, 6),
    color=["#1f77b4", "#ff7f0e"],  # blue & orange
)

plt.title("Lipreading Model Performance (Train vs Unseen vs Fine-tuned)")
plt.ylabel("Error Rate")
plt.xlabel("Dataset")

# Values ko bars ke upar likhna
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}",
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha="center", va="bottom", fontsize=9, rotation=0
    )

plt.legend(["CER", "WER"], loc="upper right")

# Save figure
plt.savefig("model_performance.png", dpi=300, bbox_inches="tight")
print("✅ Plot saved to model_performance.png")

plt.show()
