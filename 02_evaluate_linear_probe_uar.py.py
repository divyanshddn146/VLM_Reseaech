import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix

PER_VIDEO_PATH = "/home/Student/s4899464/Research_1/dfew_llava_probe_parallel_fresh/dfew_probe_per_video_predictions.csv"

LABELS = ["happy", "sad", "angry", "fear", "surprise", "neutral", "gross"]

df = pd.read_csv(PER_VIDEO_PATH)

print("Loaded rows:", len(df))
print("\nAvailable prompt/layer combinations:")
print(df[["prompt", "layer"]].drop_duplicates().sort_values(["prompt", "layer"]).to_string(index=False))

summary_rows = []

for (prompt, layer), g in df.groupby(["prompt", "layer"]):
    y_true = g["gt_label"].astype(str)
    y_pred = g["probe_pred_label"].astype(str)

    acc = accuracy_score(y_true, y_pred)

    # UAR = macro recall across all 7 classes
    uar = recall_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0
    )

    row = {
        "prompt": prompt,
        "layer": int(layer),
        "accuracy": acc,
        "uar": uar,
        "n": len(g),
    }

    for lab, rec in zip(LABELS, per_class_recall):
        row[f"recall_{lab}"] = rec

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows).sort_values(["prompt", "layer"])

print("\n" + "=" * 80)
print("ACCURACY + UAR FOR ALL AVAILABLE LAYERS")
print("=" * 80)
print(summary[["prompt", "layer", "accuracy", "uar", "n"]].round(4).to_string(index=False))

print("\n" + "=" * 80)
print("BEST LAYER BY UAR")
print("=" * 80)
best_uar = summary.loc[summary["uar"].idxmax()]
print(best_uar.round(4).to_string())

print("\n" + "=" * 80)
print("BEST LAYER BY ACCURACY")
print("=" * 80)
best_acc = summary.loc[summary["accuracy"].idxmax()]
print(best_acc.round(4).to_string())

out_path = "/home/Student/s4899464/Research_1/dfew_llava_probe_parallel_fresh/dfew_probe_accuracy_uar_all_layers.csv"
summary.to_csv(out_path, index=False)
print("\nSaved:", out_path)