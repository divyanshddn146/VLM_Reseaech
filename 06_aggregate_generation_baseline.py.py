import os
import glob
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

OUT_DIR = "/home/Student/s4899464/Research_1/dfew_generation_baseline"

LABELS = ["happy", "sad", "angry", "fear", "surprise", "neutral", "gross"]

files = sorted(glob.glob(os.path.join(OUT_DIR, "generation_predictions_shard_*_of_*.csv")))

print("Found files:", len(files))
for f in files:
    print(f)

if len(files) == 0:
    raise RuntimeError("No prediction shard files found.")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Remove duplicates if any
df = df.sort_values(["fold", "video_id"]).drop_duplicates(["fold", "video_id"], keep="last")

all_path = os.path.join(OUT_DIR, "ALL_generation_predictions.csv")
df.to_csv(all_path, index=False)

print("\nTotal rows:", len(df))
print("Unparsed:", int(df["parsed_label"].isna().sum()))
print("Used retry:", int(df["used_retry"].sum()))

summary_rows = []
per_class_rows = []

for fold, g in df.groupby("fold"):
    g_eval = g.dropna(subset=["parsed_label"]).copy()

    y_true = g_eval["gt_label"].astype(str)
    y_pred = g_eval["parsed_label"].astype(str)

    acc = accuracy_score(y_true, y_pred)
    uar = recall_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    per_recalls = recall_score(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    row = {
        "fold": int(fold),
        "n_total": len(g),
        "n_parsed": len(g_eval),
        "n_unparsed": int(g["parsed_label"].isna().sum()),
        "accuracy": acc,
        "uar": uar,
    }

    for lab, rec in zip(LABELS, per_recalls):
        row[f"recall_{lab}"] = rec

    summary_rows.append(row)

    for lab, rec in zip(LABELS, per_recalls):
        lab_g = g_eval[g_eval["gt_label"] == lab]
        correct = int((lab_g["parsed_label"] == lab).sum())
        total = int(len(lab_g))

        per_class_rows.append({
            "fold": int(fold),
            "emotion": lab,
            "correct": correct,
            "total": total,
            "recall": rec,
        })

summary = pd.DataFrame(summary_rows).sort_values("fold")
per_class = pd.DataFrame(per_class_rows)

summary_path = os.path.join(OUT_DIR, "generation_fold_summary.csv")
per_class_path = os.path.join(OUT_DIR, "generation_per_class.csv")

summary.to_csv(summary_path, index=False)
per_class.to_csv(per_class_path, index=False)

mean_row = {
    "method": "direct_generation_baseline",
    "folds": summary["fold"].nunique(),
    "mean_accuracy": summary["accuracy"].mean(),
    "std_accuracy": summary["accuracy"].std(),
    "mean_uar": summary["uar"].mean(),
    "std_uar": summary["uar"].std(),
    "mean_unparsed": summary["n_unparsed"].mean(),
}

for lab in LABELS:
    mean_row[f"mean_recall_{lab}"] = summary[f"recall_{lab}"].mean()

final = pd.DataFrame([mean_row])
final_path = os.path.join(OUT_DIR, "FINAL_generation_5fold_result.csv")
final.to_csv(final_path, index=False)

print("\n" + "=" * 100)
print("FOLD SUMMARY")
print("=" * 100)
print(summary.round(4).to_string(index=False))

print("\n" + "=" * 100)
print("FINAL 5-FOLD GENERATION BASELINE")
print("=" * 100)
print(final.round(4).to_string(index=False))

print("\nSaved:")
print(all_path)
print(summary_path)
print(per_class_path)
print(final_path)

print("\nUnparsed examples:")
unparsed = df[df["parsed_label"].isna()]
if len(unparsed) > 0:
    print(unparsed[["fold", "video_id", "gt_label", "generation_text", "retry_text"]].head(20).to_string(index=False))
else:
    print("None")