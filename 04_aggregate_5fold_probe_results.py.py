import os
import glob
import pandas as pd

OUT_DIR = "/home/Student/s4899464/Research_1/dfew_llava_probe_parallel_fresh"
RESULT_DIR = os.path.join(OUT_DIR, "five_fold_8methods_results")

summary_files = glob.glob(os.path.join(RESULT_DIR, "fold_*", "*", "summary.csv"))
per_class_files = glob.glob(os.path.join(RESULT_DIR, "fold_*", "*", "per_class.csv"))

print("Found summary files:", len(summary_files))
print("Found per-class files:", len(per_class_files))

if len(summary_files) == 0:
    raise RuntimeError("No summary files found.")

summary_df = pd.concat([pd.read_csv(p) for p in summary_files], ignore_index=True)
per_class_df = pd.concat([pd.read_csv(p) for p in per_class_files], ignore_index=True)

all_summary_path = os.path.join(RESULT_DIR, "ALL_fold_method_summary.csv")
all_per_class_path = os.path.join(RESULT_DIR, "ALL_fold_method_per_class.csv")

summary_df.to_csv(all_summary_path, index=False)
per_class_df.to_csv(all_per_class_path, index=False)

agg = (
    summary_df
    .groupby("method")
    .agg(
        folds=("fold", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_uar=("uar", "mean"),
        std_uar=("uar", "std"),
        mean_recall_happy=("recall_happy", "mean"),
        mean_recall_sad=("recall_sad", "mean"),
        mean_recall_angry=("recall_angry", "mean"),
        mean_recall_fear=("recall_fear", "mean"),
        mean_recall_surprise=("recall_surprise", "mean"),
        mean_recall_neutral=("recall_neutral", "mean"),
        mean_recall_gross=("recall_gross", "mean"),
        mean_test_n=("test_n", "mean"),
    )
    .reset_index()
)

agg["combined_score"] = 0.5 * agg["mean_accuracy"] + 0.5 * agg["mean_uar"]

agg_path = os.path.join(RESULT_DIR, "FINAL_5fold_mean_std.csv")
agg.to_csv(agg_path, index=False)

print("\nSaved:")
print(all_summary_path)
print(all_per_class_path)
print(agg_path)

print("\n" + "=" * 100)
print("FINAL 5-FOLD RESULTS SORTED BY COMBINED SCORE")
print("=" * 100)

cols = [
    "method",
    "folds",
    "mean_accuracy",
    "std_accuracy",
    "mean_uar",
    "std_uar",
    "combined_score",
    "mean_recall_gross",
    "mean_recall_fear",
    "mean_recall_surprise",
]

print(
    agg.sort_values("combined_score", ascending=False)[cols]
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 100)
print("SORTED BY ACCURACY")
print("=" * 100)

print(
    agg.sort_values("mean_accuracy", ascending=False)[cols]
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 100)
print("SORTED BY UAR")
print("=" * 100)

print(
    agg.sort_values("mean_uar", ascending=False)[cols]
    .round(4)
    .to_string(index=False)
)