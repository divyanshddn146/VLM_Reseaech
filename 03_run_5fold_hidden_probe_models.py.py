import os
import json
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler


# ============================================================
# CONFIG
# ============================================================

DFEW_ROOT = "/home/Student/s4899464/DFEW-part2"
DFEW_SPLIT_ROOT = os.path.join(DFEW_ROOT, "EmoLabel_DataSplit")

OUT_DIR = "/home/Student/s4899464/Research_1/dfew_llava_probe_parallel_fresh"
CACHE_DIR = os.path.join(OUT_DIR, "fresh_hidden_cache")

RESULT_DIR = os.path.join(OUT_DIR, "five_fold_8methods_results")
os.makedirs(RESULT_DIR, exist_ok=True)

PROMPT_KEY = "P1_minimal"

VAL_RATIO_FROM_TRAIN = 0.20
RANDOM_SEED = 42

LABELS = ["happy", "sad", "angry", "fear", "surprise", "neutral", "gross"]

DFEW_ID_TO_EMO = {
    1: "happy",
    2: "sad",
    3: "neutral",
    4: "angry",
    5: "surprise",
    6: "gross",
    7: "fear",
}

EMO_TO_ID = {emo: i for i, emo in enumerate(LABELS)}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MLP_EPOCHS = 100
MLP_LR = 1e-3
MLP_BATCH_SIZE = 64
MLP_WEIGHT_DECAY = 1e-4
MLP_PATIENCE = 18
MLP_HIDDEN = 512
MLP_DROPOUT = 0.25


# ============================================================
# 8 FINAL METHODS
# ============================================================

METHODS = [
    {
        "name": "accuracy_mlp_concat_L16_20_26",
        "feature_type": "concat",
        "layers": [16, 20, 26],
        "balanced_sampler": False,
        "weight_power": 0.0,
        "select_metric": "val_acc",
    },
    {
        "name": "bridge_mlp_concat_L20_22_26",
        "feature_type": "concat",
        "layers": [20, 22, 26],
        "balanced_sampler": False,
        "weight_power": 0.0,
        "select_metric": "val_acc",
    },
    {
        "name": "weighted_mlp_L20",
        "feature_type": "single",
        "layers": [20],
        "balanced_sampler": False,
        "weight_power": 1.0,
        "select_metric": "val_acc",
    },
    {
        "name": "weighted_mlp_concat_key_layers",
        "feature_type": "concat",
        "layers": [13, 16, 20, 22, 26, 31],
        "balanced_sampler": False,
        "weight_power": 1.0,
        "select_metric": "val_acc",
    },
    {
        "name": "weighted_mlp_concat_L16_20_26",
        "feature_type": "concat",
        "layers": [16, 20, 26],
        "balanced_sampler": False,
        "weight_power": 1.0,
        "select_metric": "val_acc",
    },
    {
        "name": "weighted_mlp_L22",
        "feature_type": "single",
        "layers": [22],
        "balanced_sampler": False,
        "weight_power": 1.0,
        "select_metric": "val_acc",
    },
    {
        "name": "balanced_sampler_mlp_L26_p025",
        "feature_type": "single",
        "layers": [26],
        "balanced_sampler": True,
        "weight_power": 0.25,
        "select_metric": "val_uar",
    },
    {
        "name": "tradeoff_mlp_mean_L20_31_p000",
        "feature_type": "mean",
        "layers": list(range(20, 32)),
        "balanced_sampler": False,
        "weight_power": 0.0,
        "select_metric": "val_uar",
    },
]


# ============================================================
# SEED
# ============================================================

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DFEW SPLIT
# ============================================================

def load_dfew_split(set_id: int):
    train_csv = os.path.join(
        DFEW_SPLIT_ROOT,
        "train(single-labeled)",
        f"set_{set_id}.csv",
    )

    test_csv = os.path.join(
        DFEW_SPLIT_ROOT,
        "test(single-labeled)",
        f"set_{set_id}.csv",
    )

    if not os.path.exists(train_csv):
        raise FileNotFoundError(train_csv)

    if not os.path.exists(test_csv):
        raise FileNotFoundError(test_csv)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df["video_name"] = train_df["video_name"].astype(int)
    test_df["video_name"] = test_df["video_name"].astype(int)

    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    train_df = train_df[train_df["label"].isin(DFEW_ID_TO_EMO.keys())].copy()
    test_df = test_df[test_df["label"].isin(DFEW_ID_TO_EMO.keys())].copy()

    train_df, val_df = train_test_split(
        train_df,
        test_size=VAL_RATIO_FROM_TRAIN,
        stratify=train_df["label"],
        random_state=RANDOM_SEED,
    )

    def df_to_records(df, split):
        records = []
        for _, row in df.iterrows():
            video_id = int(row["video_name"])
            label_id = int(row["label"])
            gt_label = DFEW_ID_TO_EMO[label_id]

            records.append({
                "split": split,
                "video_id": video_id,
                "video_name": f"{video_id}.mp4",
                "gt_label": gt_label,
                "gt_idx": EMO_TO_ID[gt_label],
                "dfew_label_id": label_id,
            })

        return records

    train_records = df_to_records(train_df, "train")
    val_records = df_to_records(val_df, "val")
    test_records = df_to_records(test_df, "test")

    return train_records, val_records, test_records


# ============================================================
# HIDDEN LOADING
# ============================================================

def cache_path(video_name):
    base = os.path.splitext(video_name)[0]
    return os.path.join(CACHE_DIR, f"{base}_{PROMPT_KEY}_hiddens.pt")


def load_hidden_tensor(video_name):
    path = cache_path(video_name)

    if not os.path.exists(path):
        return None

    try:
        t = torch.load(path, map_location="cpu", weights_only=True)

        if torch.is_tensor(t) and t.ndim == 2:
            return t.float()

    except Exception as e:
        print(f"Failed loading {path}: {e}")

    return None


def get_feature_from_hidden(hidden_tensor, method_spec):
    layers = method_spec["layers"]
    feature_type = method_spec["feature_type"]

    if feature_type == "single":
        return hidden_tensor[layers[0]].numpy().astype(np.float32)

    if feature_type == "mean":
        return hidden_tensor[layers].mean(dim=0).numpy().astype(np.float32)

    if feature_type == "concat":
        return hidden_tensor[layers].reshape(-1).numpy().astype(np.float32)

    raise ValueError(f"Unknown feature_type: {feature_type}")


def build_matrix(records, method_spec):
    X = []
    y = []
    kept = []
    missing = 0

    for r in records:
        t = load_hidden_tensor(r["video_name"])

        if t is None:
            missing += 1
            continue

        try:
            feat = get_feature_from_hidden(t, method_spec)
        except Exception:
            missing += 1
            continue

        X.append(feat)
        y.append(r["gt_idx"])
        kept.append(r)

    if len(X) == 0:
        return None, None, [], missing

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y, kept, missing


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    uar = recall_score(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        average="macro",
        zero_division=0,
    )

    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        average=None,
        zero_division=0,
    )

    out = {
        "accuracy": float(acc),
        "uar": float(uar),
    }

    for i, lab in enumerate(LABELS):
        out[f"recall_{lab}"] = float(per_class_recall[i])

    return out


def make_per_class_rows(fold_id, method_name, y_true, y_pred):
    rows = []

    for i, lab in enumerate(LABELS):
        mask = y_true == i
        total = int(mask.sum())

        if total == 0:
            correct = 0
            recall = 0.0
        else:
            correct = int((y_pred[mask] == y_true[mask]).sum())
            recall = correct / total

        rows.append({
            "fold": fold_id,
            "method": method_name,
            "emotion": lab,
            "correct": correct,
            "total": total,
            "recall": recall,
        })

    return rows


# ============================================================
# MLP MODEL
# ============================================================

class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=7, dropout=0.25):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# CLASS WEIGHTS + BALANCED SAMPLER
# ============================================================

def make_loss_weights(y_train, power=0.0):
    counts = np.bincount(y_train, minlength=len(LABELS)).astype(np.float32)

    base_weights = counts.sum() / (len(LABELS) * np.maximum(counts, 1.0))
    base_weights = np.clip(base_weights, 0.5, 10.0)

    weights = base_weights ** float(power)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32), counts, base_weights


def make_balanced_sampler(y_train):
    y_np = np.asarray(y_train)
    class_counts = np.bincount(y_np, minlength=len(LABELS)).astype(np.float32)

    class_sample_weights = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = class_sample_weights[y_np]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler, class_counts, class_sample_weights


# ============================================================
# TRAIN
# ============================================================

def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    method_spec,
):
    input_dim = X_train.shape[1]

    model = MLPProbe(
        input_dim=input_dim,
        hidden_dim=MLP_HIDDEN,
        num_classes=len(LABELS),
        dropout=MLP_DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MLP_LR,
        weight_decay=MLP_WEIGHT_DECAY,
    )

    loss_weights, class_counts, base_loss_weights = make_loss_weights(
        y_train,
        power=method_spec["weight_power"],
    )

    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(DEVICE))

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)

    if method_spec["balanced_sampler"]:
        sampler, _, sampler_class_weights = make_balanced_sampler(y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=MLP_BATCH_SIZE,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
        )
    else:
        sampler_class_weights = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=MLP_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    best_score = -1.0
    best_val_acc = -1.0
    best_val_uar = -1.0
    best_state = None
    bad = 0

    for epoch in range(MLP_EPOCHS):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            val_logits = model(X_val_t)
            val_pred = val_logits.argmax(dim=1).cpu().numpy()

        val_true = y_val_t.cpu().numpy()

        val_acc = accuracy_score(val_true, val_pred)
        val_uar = recall_score(
            val_true,
            val_pred,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        )

        if method_spec["select_metric"] == "val_acc":
            score = val_acc
        elif method_spec["select_metric"] == "val_uar":
            score = val_uar
        elif method_spec["select_metric"] == "combined":
            score = 0.5 * val_acc + 0.5 * val_uar
        else:
            raise ValueError(method_spec["select_metric"])

        if score > best_score:
            best_score = score
            best_val_acc = val_acc
            best_val_uar = val_uar
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if bad >= MLP_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        test_logits = model(X_test_t)
        test_probs = torch.softmax(test_logits, dim=1)
        test_pred = test_probs.argmax(dim=1).cpu().numpy()

    extra = {
        "best_val_accuracy": float(best_val_acc),
        "best_val_uar": float(best_val_uar),
        "best_val_score": float(best_score),
        "class_counts": class_counts.tolist(),
        "loss_base_weights": base_loss_weights.tolist(),
        "loss_final_weights": loss_weights.numpy().tolist(),
        "sampler_class_weights": None if sampler_class_weights is None else sampler_class_weights.tolist(),
    }

    return test_pred, extra


# ============================================================
# TASK RUNNER
# ============================================================

def run_task(task_id: int):
    num_methods = len(METHODS)

    fold_id = task_id // num_methods + 1
    method_idx = task_id % num_methods

    if fold_id < 1 or fold_id > 5:
        raise ValueError(f"Bad fold_id={fold_id} from task_id={task_id}")

    method_spec = METHODS[method_idx]
    method_name = method_spec["name"]

    seed_everything(RANDOM_SEED + task_id)

    print("=" * 80)
    print("DFEW 5-FOLD ONE-TASK RUN")
    print("=" * 80)
    print(f"Task ID: {task_id}")
    print(f"Fold: {fold_id}")
    print(f"Method index: {method_idx}")
    print(f"Method name: {method_name}")
    print(f"Method spec: {json.dumps(method_spec)}")
    print(f"Device: {DEVICE}")
    print(f"Cache: {CACHE_DIR}")

    train_records, val_records, test_records = load_dfew_split(fold_id)

    print("\nSplit sizes before cache filtering:")
    print(f"Train: {len(train_records)}")
    print(f"Val:   {len(val_records)}")
    print(f"Test:  {len(test_records)}")

    X_train, y_train, train_kept, miss_train = build_matrix(train_records, method_spec)
    X_val, y_val, val_kept, miss_val = build_matrix(val_records, method_spec)
    X_test, y_test, test_kept, miss_test = build_matrix(test_records, method_spec)

    print("\nUsable after cache filtering:")
    print(f"Train usable: {0 if X_train is None else len(X_train)} | missing: {miss_train}")
    print(f"Val usable:   {0 if X_val is None else len(X_val)} | missing: {miss_val}")
    print(f"Test usable:  {0 if X_test is None else len(X_test)} | missing: {miss_test}")

    if X_train is None or X_val is None or X_test is None:
        raise RuntimeError("Missing matrices. Cache may be incomplete.")

    print(f"Feature dim: {X_train.shape[1]}")

    y_pred, extra = train_mlp(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        method_spec=method_spec,
    )

    metrics = compute_metrics(y_test, y_pred)

    summary_row = {
        "task_id": task_id,
        "fold": fold_id,
        "method_index": method_idx,
        "method": method_name,
        "feature_type": method_spec["feature_type"],
        "layers": json.dumps(method_spec["layers"]),
        "balanced_sampler": method_spec["balanced_sampler"],
        "weight_power": method_spec["weight_power"],
        "select_metric": method_spec["select_metric"],
        "feature_dim": X_train.shape[1],
        "train_n": len(X_train),
        "val_n": len(X_val),
        "test_n": len(X_test),
        "missing_train": miss_train,
        "missing_val": miss_val,
        "missing_test": miss_test,
        "best_val_accuracy": extra["best_val_accuracy"],
        "best_val_uar": extra["best_val_uar"],
        "best_val_score": extra["best_val_score"],
        **metrics,
    }

    per_class = make_per_class_rows(
        fold_id=fold_id,
        method_name=method_name,
        y_true=y_test,
        y_pred=y_pred,
    )

    task_dir = os.path.join(RESULT_DIR, f"fold_{fold_id}", method_name)
    os.makedirs(task_dir, exist_ok=True)

    summary_path = os.path.join(task_dir, "summary.csv")
    per_class_path = os.path.join(task_dir, "per_class.csv")

    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    pd.DataFrame(per_class).to_csv(per_class_path, index=False)

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(pd.DataFrame([summary_row]).round(4).to_string(index=False))
    print("\nSaved summary:", summary_path)
    print("Saved per-class:", per_class_path)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    args = parser.parse_args()

    run_task(args.task_id)


if __name__ == "__main__":
    main()