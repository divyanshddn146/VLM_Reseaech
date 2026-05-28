"""
DFEW-only LLaVA-NeXT-Video-7B Linear Probe Evaluation
Data-parallel fresh hidden extraction
====================================================

NO FERV39K.
NO causal patching.
NO ablation.
NO old hidden-state reuse during extraction.

This script supports three modes:

1. extract_only
   - Loads LLaVA.
   - Processes only this shard of videos.
   - Saves fresh hidden states.
   - Does NOT train probe.

2. probe_only
   - Does NOT load LLaVA.
   - Reads saved hidden states from all shards.
   - Trains/evaluates linear probe.

3. extract_and_probe
   - Single-job version.
   - Extracts and then trains probe.

Recommended workflow:

Step 1: run 3 Slurm array jobs:
    RUN_MODE=extract_only
    SHARD_COUNT=3
    SHARD_INDEX=0,1,2

Step 2: after extraction finishes:
    RUN_MODE=probe_only

Expected DFEW structure:

/home/Student/s4899464/DFEW-part2/
    EmoLabel_DataSplit/
        train(single-labeled)/
            set_1.csv
            ...
        test(single-labeled)/
            set_1.csv
            ...
    Clip/
        original/
            part_1/
            part_2/
            ...
            part_11/
"""

import os
import gc
import copy
from typing import Dict, List, Optional
from collections import defaultdict

# ============================================================
# ENV / RUN MODE
# ============================================================

RUN_MODE = os.environ.get("RUN_MODE", "extract_and_probe")
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
SHARD_COUNT = int(os.environ.get("SHARD_COUNT", "1"))

# For array jobs, Slurm gives each task one GPU.
# So do NOT force CUDA_VISIBLE_DEVICES=0,1,2 here.
# Each job should see one GPU.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"

# Your real DFEW path
DFEW_ROOT = "/home/Student/s4899464/DFEW-part2"

DFEW_SPLIT_ROOT = os.path.join(DFEW_ROOT, "EmoLabel_DataSplit")
DFEW_ORIGINAL_DIR = os.path.join(DFEW_ROOT, "Clip", "original")

OUT_DIR = "/home/Student/s4899464/Research_1/dfew_llava_probe_parallel_fresh"
CACHE_DIR = os.path.join(OUT_DIR, "fresh_hidden_cache")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Official DFEW fold: 1 to 5
DFEW_SET_ID = 1

# Official DFEW gives train/test only.
# We create validation from train.
VAL_RATIO_FROM_TRAIN = 0.20

# Start with P1 only.
PROMPTS_TO_RUN = [
    "P1_minimal",
    # "P3_reasoning_then_answer",
]

KEY_LAYERS = [13, 16, 20, 22, 26, 31]

T_FRAMES = 16

PROBE_EPOCHS = 50
PROBE_LR = 1e-3
PROBE_BATCH_SIZE = 32
PROBE_WEIGHT_DECAY = 1e-4
PATIENCE = 20

RANDOM_SEED = 42

# Behavior by mode
if RUN_MODE == "extract_only":
    RUN_EXTRACTION = True
    FORCE_REEXTRACT = True

elif RUN_MODE == "probe_only":
    RUN_EXTRACTION = False
    FORCE_REEXTRACT = False

elif RUN_MODE == "extract_and_probe":
    RUN_EXTRACTION = True
    FORCE_REEXTRACT = True

else:
    raise ValueError(
        f"Unknown RUN_MODE={RUN_MODE}. "
        "Use extract_only, probe_only, or extract_and_probe."
    )


# ============================================================
# LABELS
# ============================================================

LABELS = ["happy", "sad", "angry", "fear", "surprise", "neutral", "gross"]
LABEL_SET = set(LABELS)
LABEL_STR = ", ".join(LABELS)

DFEW_ID_TO_EMO = {
    1: "happy",
    2: "sad",
    3: "neutral",
    4: "angry",
    5: "surprise",
    6: "gross",   # DFEW disgust mapped to gross
    7: "fear",
}

EMO_TO_ID = {emo: idx for idx, emo in enumerate(LABELS)}

PROMPTS = {
    "P1_minimal": [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {
                    "type": "text",
                    "text": (
                        f"Identify the primary emotion. "
                        f"Answer with exactly ONE word from this list of 7 options: {LABEL_STR}."
                    ),
                },
            ],
        }
    ],
    "P3_reasoning_then_answer": [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {
                    "type": "text",
                    "text": (
                        f"Briefly explain the person's emotion, "
                        f"then answer with exactly one label from: {LABEL_STR}."
                    ),
                },
            ],
        }
    ],
}


# ============================================================
# UTILS
# ============================================================

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_layers(model):
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "layers")
    ):
        return model.language_model.model.layers

    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers

    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers

    raise RuntimeError("Could not locate transformer layers.")


def to_device_dtype(batch, device):
    out = {}

    for k, v in batch.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue

        if v.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.bool):
            out[k] = v.to(device)
        else:
            out[k] = v.to(device=device, dtype=torch.bfloat16)

    return out


def prompt_to_text(processor, prompt_messages):
    tok = processor.tokenizer

    if not getattr(tok, "chat_template", None):
        tok.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}"
            "{% for content in message['content'] %}"
            "{% if content['type'] == 'text' %}{{ content['text'] }}"
            "{% elif content['type'] == 'video' %}<video>\n{% endif %}"
            "{% endfor %}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}ASSISTANT:{% endif %}"
        )

    text = tok.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    has_video = any(
        c.get("type") == "video"
        for m in prompt_messages
        for c in m.get("content", [])
    )

    if has_video and "<video>" not in text:
        text = "<video>\n" + text

    return text


# ============================================================
# DFEW VIDEO INDEXING
# ============================================================

def build_video_path_index(original_dir: str) -> Dict[int, str]:
    """
    Builds video_id -> full path.

    Supports:
        part1
        part_1
        Part1
        Part_1

    Your path is like:
        /home/Student/s4899464/DFEW-part2/Clip/original/part_11
    """

    if not os.path.exists(original_dir):
        raise FileNotFoundError(f"Video original dir not found: {original_dir}")

    video_index = {}

    print("\n" + "=" * 80)
    print("SCANNING DFEW VIDEO FOLDERS")
    print("=" * 80)
    print(f"Original dir: {original_dir}")

    for part_num in range(1, 12):
        possible_names = [
            f"part{part_num}",
            f"part_{part_num}",
            f"Part{part_num}",
            f"Part_{part_num}",
        ]

        part_dir = None

        for folder_name in possible_names:
            candidate = os.path.join(original_dir, folder_name)
            if os.path.exists(candidate):
                part_dir = candidate
                break

        if part_dir is None:
            print(f"⚠️ Missing folder for part {part_num}")
            continue

        count = 0

        for f in os.listdir(part_dir):
            if not f.lower().endswith(".mp4"):
                continue

            name = os.path.splitext(f)[0]

            try:
                video_id = int(name)
            except ValueError:
                print(f"⚠️ Skipping non-numeric filename: {f}")
                continue

            full_path = os.path.join(part_dir, f)

            if video_id in video_index:
                print(f"⚠️ Duplicate video id {video_id}:")
                print(f"old: {video_index[video_id]}")
                print(f"new: {full_path}")

            video_index[video_id] = full_path
            count += 1

        print(f"part {part_num:<2}: {count} videos")

    print(f"\nTotal indexed videos: {len(video_index)}")

    if len(video_index) == 0:
        raise RuntimeError("No videos found. Check DFEW_ORIGINAL_DIR.")

    return video_index


def load_official_dfew_split(set_id: int):
    video_index = build_video_path_index(DFEW_ORIGINAL_DIR)

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
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    print("\n" + "=" * 80)
    print("LOADING OFFICIAL DFEW SPLIT")
    print("=" * 80)
    print(f"Train CSV: {train_csv}")
    print(f"Test CSV:  {test_csv}")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    required_cols = {"video_name", "label"}

    if not required_cols.issubset(set(train_df.columns)):
        raise ValueError("Train CSV must contain columns: video_name,label")

    if not required_cols.issubset(set(test_df.columns)):
        raise ValueError("Test CSV must contain columns: video_name,label")

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["video_name"] = train_df["video_name"].astype(int)
    test_df["video_name"] = test_df["video_name"].astype(int)

    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    train_df = train_df[train_df["label"].isin(DFEW_ID_TO_EMO.keys())].copy()
    test_df = test_df[test_df["label"].isin(DFEW_ID_TO_EMO.keys())].copy()

    print("\nOfficial train distribution before validation split:")
    print(train_df["label"].value_counts().sort_index().to_string())

    print("\nOfficial test distribution:")
    print(test_df["label"].value_counts().sort_index().to_string())

    train_df, val_df = train_test_split(
        train_df,
        test_size=VAL_RATIO_FROM_TRAIN,
        stratify=train_df["label"],
        random_state=RANDOM_SEED,
    )

    def df_to_records(df: pd.DataFrame, split_name: str):
        records = []
        missing_video = 0

        for _, row in df.iterrows():
            video_id = int(row["video_name"])
            label_id = int(row["label"])

            gt_label = DFEW_ID_TO_EMO.get(label_id)

            if gt_label not in LABEL_SET:
                continue

            video_name = f"{video_id}.mp4"
            video_path = video_index.get(video_id)

            if video_path is None or not os.path.exists(video_path):
                missing_video += 1
                continue

            records.append({
                "split": split_name,
                "video_id": video_id,
                "video_name": video_name,
                "video_path": video_path,
                "gt_label": gt_label,
                "gt_idx": EMO_TO_ID[gt_label],
                "dfew_label_id": label_id,
                "official_set": set_id,
            })

        print("\n" + "-" * 80)
        print(f"{split_name.upper()} RECORDS")
        print("-" * 80)
        print(f"CSV rows:       {len(df)}")
        print(f"usable videos:  {len(records)}")
        print(f"missing videos: {missing_video}")

        if len(records) > 0:
            dist = pd.Series([r["gt_label"] for r in records]).value_counts()
            print("label distribution:")
            print(dist.reindex(LABELS).fillna(0).astype(int).to_string())

        return records

    train_records = df_to_records(train_df, "train")
    val_records = df_to_records(val_df, "val")
    test_records = df_to_records(test_df, "test")

    return train_records, val_records, test_records


# ============================================================
# VIDEO LOADING
# ============================================================

def load_video_frames(path: str, T: int = 16):
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total < T:
            cap.release()
            return None

        idxs = np.linspace(0, total - 1, T).astype(int).tolist()
        frames = []

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret:
                cap.release()
                return None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        cap.release()

        if len(frames) != T:
            return None

        return tuple(frames)

    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e}")
        return None


# ============================================================
# HIDDEN EXTRACTION
# ============================================================

class HiddenStateExtractor:
    def __init__(self, model, processor, cache_dir: str):
        self.model = model
        self.processor = processor
        self.cache_dir = cache_dir
        self.layers = get_layers(model)
        self.num_layers = len(self.layers)
        self.input_device = next(model.parameters()).device

        os.makedirs(cache_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("HIDDEN STATE EXTRACTOR")
        print("=" * 80)
        print(f"Input device: {self.input_device}")
        print(f"Language layers: {self.num_layers}")

    def get_cache_path(self, video_name: str, prompt_key: str) -> str:
        safe_name = os.path.splitext(video_name)[0]
        return os.path.join(self.cache_dir, f"{safe_name}_{prompt_key}_hiddens.pt")

    @torch.no_grad()
    def extract_and_cache(
        self,
        video_path: str,
        video_name: str,
        prompt_key: str,
        prompt_struct: List[Dict],
    ) -> bool:

        cache_path = self.get_cache_path(video_name, prompt_key)

        if os.path.exists(cache_path) and FORCE_REEXTRACT:
            os.remove(cache_path)

        if os.path.exists(cache_path) and not FORCE_REEXTRACT:
            return True

        frames = load_video_frames(video_path, T=T_FRAMES)

        if frames is None:
            print(f"⚠️ Could not load frames: {video_path}")
            return False

        prompt_str = prompt_to_text(self.processor, prompt_struct)

        try:
            batch = self.processor(
                text=[prompt_str],
                videos=[list(frames)],
                return_tensors="pt",
                padding=True,
            )

            inputs = to_device_dtype(batch, self.input_device)

            attn = inputs["attention_mask"]
            true_length = int(attn.sum().item())
            last_token_idx = true_length - 1

            hidden_states_by_layer = {}
            handles = []

            def make_hook(layer_idx: int):
                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, (tuple, list)) else out
                    hidden_states_by_layer[layer_idx] = (
                        h[:, last_token_idx, :]
                        .detach()
                        .float()
                        .cpu()
                    )
                return hook_fn

            for layer_idx in range(self.num_layers):
                handles.append(
                    self.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
                )

            try:
                _ = self.model(**inputs)

                if len(hidden_states_by_layer) != self.num_layers:
                    print(
                        f"⚠️ Missing layers for {video_name}: "
                        f"{len(hidden_states_by_layer)}/{self.num_layers}"
                    )
                    return False

                hidden_tensor = torch.stack(
                    [
                        hidden_states_by_layer[layer_idx].squeeze(0)
                        for layer_idx in range(self.num_layers)
                    ],
                    dim=0,
                ).contiguous()

                torch.save(hidden_tensor, cache_path)
                return True

            finally:
                for h in handles:
                    h.remove()

                del inputs
                del batch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️ Extract failed for {video_name}: {e}")
            return False

    def extract_all(self, records: List[Dict], prompts_dict: Dict[str, List[Dict]]):
        print("\n" + "=" * 80)
        print("EXTRACTING HIDDEN STATES FOR THIS SHARD")
        print("=" * 80)
        print(f"Records in this shard: {len(records)}")
        print(f"FORCE_REEXTRACT: {FORCE_REEXTRACT}")

        for prompt_key, prompt_struct in prompts_dict.items():
            print("\n" + "-" * 80)
            print(f"Prompt: {prompt_key}")
            print("-" * 80)

            extracted = 0
            failed = 0
            skipped = 0
            deleted = 0

            for r in tqdm(records, desc=f"Shard {SHARD_INDEX}/{SHARD_COUNT} {prompt_key}"):
                cache_path = self.get_cache_path(r["video_name"], prompt_key)

                if os.path.exists(cache_path) and not FORCE_REEXTRACT:
                    skipped += 1
                    continue

                if os.path.exists(cache_path) and FORCE_REEXTRACT:
                    os.remove(cache_path)
                    deleted += 1

                ok = self.extract_and_cache(
                    video_path=r["video_path"],
                    video_name=r["video_name"],
                    prompt_key=prompt_key,
                    prompt_struct=prompt_struct,
                )

                if ok:
                    extracted += 1
                else:
                    failed += 1

            print(f"skipped existing: {skipped}")
            print(f"deleted existing: {deleted}")
            print(f"newly extracted:  {extracted}")
            print(f"failed:           {failed}")


# ============================================================
# HIDDEN LOADING FOR PROBE
# ============================================================

def load_hidden(video_name: str, prompt_key: str, layer_idx: int) -> Optional[np.ndarray]:
    safe_name = os.path.splitext(video_name)[0]
    path = os.path.join(CACHE_DIR, f"{safe_name}_{prompt_key}_hiddens.pt")

    if not os.path.exists(path):
        return None

    try:
        t = torch.load(path, map_location="cpu", weights_only=True)

        if t.ndim == 2 and 0 <= layer_idx < t.shape[0]:
            return t[layer_idx].numpy().astype(np.float32)

        return None

    except Exception as e:
        print(f"⚠️ Failed to load hidden for {video_name}, L{layer_idx}: {e}")
        return None


def build_matrix(records: List[Dict], prompt_key: str, layer_idx: int):
    X = []
    y = []
    kept_records = []
    missing_hidden = 0

    for r in records:
        h = load_hidden(r["video_name"], prompt_key, layer_idx)

        if h is None:
            missing_hidden += 1
            continue

        X.append(h)
        y.append(r["gt_idx"])
        kept_records.append(r)

    if len(X) == 0:
        return None, None, [], missing_hidden

    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y, kept_records, missing_hidden


# ============================================================
# LINEAR PROBE
# ============================================================

class EmotionProbe:
    """
    Linear probe:
        nn.Linear(hidden_dim, 7)

    Trained using:
        Adam + CrossEntropyLoss + weight decay + early stopping
    """

    def __init__(self, hidden_dim: int, num_classes: int = 7):
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.classifier: Optional[nn.Linear] = None

    def train_probe(self, X_train, y_train, X_val, y_val, device):
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes).to(device)

        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=PROBE_LR,
            weight_decay=PROBE_WEIGHT_DECAY,
        )

        criterion = nn.CrossEntropyLoss()

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        best_val_acc = -1.0
        best_state = None
        bad = 0

        for epoch in range(PROBE_EPOCHS):
            self.classifier.train()
            perm = torch.randperm(len(X_train), device=device)

            for start in range(0, len(X_train), PROBE_BATCH_SIZE):
                idx = perm[start:start + PROBE_BATCH_SIZE]
                xb = X_train[idx]
                yb = y_train[idx]

                logits = self.classifier(xb)
                loss = criterion(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.classifier.eval()

            with torch.no_grad():
                val_logits = self.classifier(X_val)
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_val).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.classifier.state_dict())
                bad = 0
            else:
                bad += 1

            if bad >= PATIENCE:
                break

        if best_state is not None:
            self.classifier.load_state_dict(best_state)

        return float(best_val_acc)

    def predict_with_probs(self, X, device):
        self.classifier.eval()
        X = X.to(device)

        with torch.no_grad():
            logits = self.classifier(X)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1)

        return (
            pred_idx.cpu().numpy(),
            probs.cpu().numpy(),
            logits.cpu().numpy(),
        )


# ============================================================
# EVALUATION
# ============================================================

def per_class_accuracy(pred_labels: List[str], gt_labels: List[str]):
    counts = defaultdict(lambda: [0, 0])

    for pred, gt in zip(pred_labels, gt_labels):
        counts[gt][1] += 1
        if pred == gt:
            counts[gt][0] += 1

    out = {}

    for lab in LABELS:
        correct, total = counts[lab]
        acc = correct / total if total > 0 else 0.0

        out[lab] = {
            "correct": correct,
            "total": total,
            "accuracy": acc,
        }

    return out


def run_probe_for_layer(
    train_records,
    val_records,
    test_records,
    prompt_key,
    layer_idx,
    device,
):
    print("\n" + "=" * 80)
    print(f"TRAIN/EVAL PROBE | prompt={prompt_key} | layer={layer_idx}")
    print("=" * 80)

    X_train, y_train, train_kept, miss_train = build_matrix(train_records, prompt_key, layer_idx)
    X_val, y_val, val_kept, miss_val = build_matrix(val_records, prompt_key, layer_idx)
    X_test, y_test, test_kept, miss_test = build_matrix(test_records, prompt_key, layer_idx)

    print(f"Train usable: {0 if X_train is None else len(X_train)} | missing hidden: {miss_train}")
    print(f"Val usable:   {0 if X_val is None else len(X_val)} | missing hidden: {miss_val}")
    print(f"Test usable:  {0 if X_test is None else len(X_test)} | missing hidden: {miss_test}")

    if X_train is None or X_val is None or X_test is None:
        print("⚠️ Missing hidden states. Skipping.")
        return [], [], None

    probe = EmotionProbe(hidden_dim=X_train.shape[1], num_classes=len(LABELS))

    val_acc = probe.train_probe(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        device=device,
    )

    pred_idx, probs, logits = probe.predict_with_probs(X_test, device)

    pred_labels = [LABELS[i] for i in pred_idx]
    gt_labels = [r["gt_label"] for r in test_kept]

    correct_flags = [int(p == g) for p, g in zip(pred_labels, gt_labels)]

    test_acc = float(np.mean(correct_flags))
    test_correct = int(sum(correct_flags))
    test_total = int(len(correct_flags))

    pc = per_class_accuracy(pred_labels, gt_labels)

    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy:       {test_acc:.4f}")
    print(f"Correct:             {test_correct}/{test_total}")

    per_video_rows = []

    for i, r in enumerate(test_kept):
        row = {
            "prompt": prompt_key,
            "layer": layer_idx,
            "split": "test",
            "official_set": r["official_set"],
            "video_id": r["video_id"],
            "video_name": r["video_name"],
            "video_path": r["video_path"],
            "dfew_label_id": r["dfew_label_id"],
            "gt_label": r["gt_label"],
            "gt_idx": r["gt_idx"],
            "probe_pred_label": pred_labels[i],
            "probe_pred_idx": int(pred_idx[i]),
            "correct": correct_flags[i],
            "probe_confidence": float(probs[i, pred_idx[i]]),
        }

        for j, lab in enumerate(LABELS):
            row[f"prob_{lab}"] = float(probs[i, j])
            row[f"logit_{lab}"] = float(logits[i, j])

        per_video_rows.append(row)

    per_class_rows = []

    for lab in LABELS:
        per_class_rows.append({
            "prompt": prompt_key,
            "layer": layer_idx,
            "emotion": lab,
            "correct": pc[lab]["correct"],
            "total": pc[lab]["total"],
            "accuracy": pc[lab]["accuracy"],
            "overall_test_accuracy": test_acc,
            "val_accuracy": val_acc,
        })

    summary_row = {
        "prompt": prompt_key,
        "layer": layer_idx,
        "official_set": DFEW_SET_ID,
        "train_n": len(X_train),
        "val_n": len(X_val),
        "test_n": len(X_test),
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_correct": test_correct,
        "test_total": test_total,
    }

    return per_video_rows, per_class_rows, summary_row


# ============================================================
# MAIN
# ============================================================

def main():
    seed_everything(RANDOM_SEED)

    print("\n" + "=" * 80)
    print("DFEW LLaVA LINEAR PROBE — PARALLEL EXTRACTION VERSION")
    print("=" * 80)
    print(f"RUN_MODE: {RUN_MODE}")
    print(f"SHARD_INDEX: {SHARD_INDEX}")
    print(f"SHARD_COUNT: {SHARD_COUNT}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"DFEW_ROOT: {DFEW_ROOT}")
    print(f"DFEW_ORIGINAL_DIR: {DFEW_ORIGINAL_DIR}")
    print(f"DFEW_SPLIT_ROOT: {DFEW_SPLIT_ROOT}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"CACHE_DIR: {CACHE_DIR}")
    print(f"RUN_EXTRACTION: {RUN_EXTRACTION}")
    print(f"FORCE_REEXTRACT: {FORCE_REEXTRACT}")

    train_records, val_records, test_records = load_official_dfew_split(
        set_id=DFEW_SET_ID
    )

    if len(train_records) == 0 or len(val_records) == 0 or len(test_records) == 0:
        raise RuntimeError("Train/val/test records are empty.")

    all_records = train_records + val_records + test_records

    split_meta_path = os.path.join(OUT_DIR, "dfew_split_metadata.csv")

    # Only write metadata once or in probe mode.
    if RUN_MODE != "extract_only" or SHARD_INDEX == 0:
        pd.DataFrame(all_records).to_csv(split_meta_path, index=False)
        print(f"\nSaved split metadata: {split_meta_path}")

    prompts_dict = {k: PROMPTS[k] for k in PROMPTS_TO_RUN}

    # Shard records for extraction only.
    extract_records = all_records[SHARD_INDEX::SHARD_COUNT]

    print("\n" + "=" * 80)
    print("SHARD INFORMATION")
    print("=" * 80)
    print(f"Total records: {len(all_records)}")
    print(f"Records for this shard: {len(extract_records)}")

    # ========================================================
    # Extraction
    # ========================================================
    if RUN_EXTRACTION:
        print("\n" + "=" * 80)
        print("LOADING LLaVA FOR EXTRACTION")
        print("=" * 80)

        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()

        processor = LlavaNextVideoProcessor.from_pretrained(
            MODEL_ID,
            use_fast=True,
        )

        tokenizer = processor.tokenizer

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        extractor = HiddenStateExtractor(
            model=model,
            processor=processor,
            cache_dir=CACHE_DIR,
        )

        extractor.extract_all(
            records=extract_records,
            prompts_dict=prompts_dict,
        )

        print("\nDeleting LLaVA model after extraction...")

        del extractor
        del model
        del processor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if RUN_MODE == "extract_only":
        print("\nExtraction-only mode complete. Probe training not run in this job.")
        return

    # ========================================================
    # Probe training
    # ========================================================
    print("\n" + "=" * 80)
    print("TRAINING LINEAR PROBES")
    print("=" * 80)

    probe_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Probe device: {probe_device}")

    all_per_video_rows = []
    all_per_class_rows = []
    all_summary_rows = []

    for prompt_key in PROMPTS_TO_RUN:
        for layer_idx in KEY_LAYERS:
            per_video_rows, per_class_rows, summary_row = run_probe_for_layer(
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
                prompt_key=prompt_key,
                layer_idx=layer_idx,
                device=probe_device,
            )

            all_per_video_rows.extend(per_video_rows)
            all_per_class_rows.extend(per_class_rows)

            if summary_row is not None:
                all_summary_rows.append(summary_row)

    per_video_df = pd.DataFrame(all_per_video_rows)
    per_class_df = pd.DataFrame(all_per_class_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    per_video_path = os.path.join(OUT_DIR, "dfew_probe_per_video_predictions.csv")
    per_class_path = os.path.join(OUT_DIR, "dfew_probe_per_class_accuracy.csv")
    summary_path = os.path.join(OUT_DIR, "dfew_probe_summary_accuracy.csv")

    per_video_df.to_csv(per_video_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("SAVED OUTPUTS")
    print("=" * 80)
    print(f"Per-video predictions: {per_video_path}")
    print(f"Per-class accuracy:    {per_class_path}")
    print(f"Summary accuracy:      {summary_path}")
    print(f"Split metadata:        {split_meta_path}")
    print(f"Hidden cache:          {CACHE_DIR}")

    print("\nSUMMARY:")
    if len(summary_df) > 0:
        print(summary_df.round(4).to_string(index=False))
    else:
        print("No summary results produced. Maybe hidden extraction is incomplete.")

    print("\nDone.")


if __name__ == "__main__":
    main()