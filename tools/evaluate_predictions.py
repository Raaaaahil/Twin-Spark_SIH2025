import pandas as pd
import os

PRED_FILE = "results/predictions.csv"
LABEL_FILE = "data/labels.csv"

print("Loading predictions...")
df_pred = pd.read_csv(PRED_FILE)
print(f"Predictions loaded: {len(df_pred)}")

# Rename prediction column "image" to a standard name
df_pred.rename(columns={"image": "image_path"}, inplace=True)

print("Loading ground truth labels...")
df_labels = pd.read_csv(LABEL_FILE)

# Ensure label file also uses same image_path naming
df_labels.rename(columns={"image_path": "image_path", "label": "true_label"}, inplace=True)

# Merge using filename only (because predictions have long path)
df_pred["fname"] = df_pred["image_path"].apply(lambda p: os.path.basename(p))
df_labels["fname"] = df_labels["image_path"].apply(lambda p: os.path.basename(p))

df = df_pred.merge(df_labels, on="fname", suffixes=("_pred", "_gt"))
print(f"Merged rows: {len(df)}")

# Compare predicted vs true
df["correct"] = df["pred_label"] == df["true_label_gt"]
accuracy = df["correct"].mean()

print("\n========== FINAL ACCURACY ==========")
print(f"Top-1 Accuracy: {accuracy:.4f}")
print("====================================")

# Save merged results
df.to_csv("results/eval_merged.csv", index=False)
print("Saved merged evaluation -> results/eval_merged.csv")
