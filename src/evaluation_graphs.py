import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# ==============================
# Load results (finetuned wale ko example le raha hu)
# Tum apne hisaab se results.csv / results_new.csv bhi use kar sakte ho
# ==============================
df = pd.read_csv("results_finetuned.csv")

# ==============================
# 1. Character-level Confusion Matrix
# ==============================
y_true = []
y_pred = []

for _, row in df.iterrows():
    actual = row["Actual"].replace(" ", "")
    pred = row["Prediction"].replace(" ", "")
    min_len = min(len(actual), len(pred))
    
    for i in range(min_len):
        y_true.append(actual[i])
        y_pred.append(pred[i])

# unique characters
labels = sorted(list(set(y_true + y_pred)))

cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=False, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Character-level Confusion Matrix (unseen data Results after finetune)")
# plt.show()
plt.savefig("evaluation_finetune.png")

# ==============================
# 2. CER / WER Distribution Histogram
# ==============================
plt.figure(figsize=(10, 6))
plt.hist(df["CER"], bins=20, alpha=0.6, label="CER")
plt.hist(df["WER"], bins=20, alpha=0.6, label="WER")
plt.xlabel("Error Rate")
plt.ylabel("Number of Samples")
plt.title("Distribution of CER and WER (unseen data Results after finetune)")
plt.legend()
plt.savefig("Distribution of CER and WER (finetune Results).png")
