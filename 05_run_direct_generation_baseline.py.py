import os
import re
import gc
import argparse
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from sklearn.metrics import accuracy_score, recall_score


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"

DFEW_ROOT = "/home/Student/s4899464/DFEW-part2"
DFEW_SPLIT_ROOT = os.path.join(DFEW_ROOT, "EmoLabel_DataSplit")
DFEW_ORIGINAL_DIR = os.path.join(DFEW_ROOT, "Clip", "original")

OUT_DIR = "/home/Student/s4899464/Research_1/dfew_generation_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

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

T_FRAMES = 16
MAX_NEW_TOKENS = 32

# If first generation cannot be parsed, retry once with stricter prompt
DO_RETRY_IF_UNPARSED = True


# ============================================================
# PROMPTS
# ============================================================

PROMPT_MAIN = [
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {
                "type": "text",
                "text": (
                    f"Identify the primary emotion in the video. "
                    f"Answer with exactly ONE word from this list: {LABEL_STR}."
                ),
            },
        ],
    }
]

PROMPT_RETRY = [
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {
                "type": "text",
                "text": (
                    "You must choose exactly one emotion label for this video.\n"
                    f"Allowed labels: {LABEL_STR}.\n"
                    "Do not explain. Do not write a sentence. "
                    "Return only one label."
                ),
            },
        ],
    }
]


# ============================================================
# VIDEO INDEXING
# ============================================================

def build_video_path_index(original_dir: str) -> Dict[int, str]:
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
            print(f"Missing folder for part {part_num}")
            continue

        count = 0

        for f in os.listdir(part_dir):
            if not f.lower().endswith(".mp4"):
                continue

            name = os.path.splitext(f)[0]

            try:
                video_id = int(name)
            except ValueError:
                continue

            video_index[video_id] = os.path.join(part_dir, f)
            count += 1

        print(f"part {part_num:<2}: {count} videos")

    print(f"Total indexed videos: {len(video_index)}")

    if len(video_index) == 0:
        raise RuntimeError("No videos found.")

    return video_index


def load_all_test_records():
    video_index = build_video_path_index(DFEW_ORIGINAL_DIR)

    records = []

    for set_id in range(1, 6):
        test_csv = os.path.join(
            DFEW_SPLIT_ROOT,
            "test(single-labeled)",
            f"set_{set_id}.csv",
        )

        if not os.path.exists(test_csv):
            raise FileNotFoundError(test_csv)

        df = pd.read_csv(test_csv)
        df["video_name"] = df["video_name"].astype(int)
        df["label"] = df["label"].astype(int)
        df = df[df["label"].isin(DFEW_ID_TO_EMO.keys())].copy()

        missing = 0

        for _, row in df.iterrows():
            video_id = int(row["video_name"])
            label_id = int(row["label"])
            gt_label = DFEW_ID_TO_EMO[label_id]

            video_path = video_index.get(video_id)

            if video_path is None or not os.path.exists(video_path):
                missing += 1
                continue

            records.append({
                "fold": set_id,
                "video_id": video_id,
                "video_name": f"{video_id}.mp4",
                "video_path": video_path,
                "dfew_label_id": label_id,
                "gt_label": gt_label,
            })

        print(f"Fold {set_id}: test rows={len(df)}, missing videos={missing}")

    print("\nTotal test records across 5 folds:", len(records))
    return records


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
        print(f"Failed to load {path}: {e}")
        return None


# ============================================================
# PROMPT FORMAT
# ============================================================

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
# TEXT-ONLY SELF EVALUATOR / PARSER
# ============================================================

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


SYNONYM_TO_LABEL = {
    # happy
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "joyful": "happy",
    "smile": "happy",
    "smiling": "happy",

    # sad
    "sad": "sad",
    "sadness": "sad",
    "unhappy": "sad",
    "sorrow": "sad",
    "crying": "sad",

    # angry
    "angry": "angry",
    "anger": "angry",
    "mad": "angry",
    "furious": "angry",
    "annoyed": "angry",

    # fear
    "fear": "fear",
    "fearful": "fear",
    "afraid": "fear",
    "scared": "fear",
    "frightened": "fear",
    "terrified": "fear",

    # surprise
    "surprise": "surprise",
    "surprised": "surprise",
    "shock": "surprise",
    "shocked": "surprise",
    "astonished": "surprise",

    # neutral
    "neutral": "neutral",
    "calm": "neutral",
    "blank": "neutral",
    "expressionless": "neutral",

    # gross/disgust
    "gross": "gross",
    "disgust": "gross",
    "disgusted": "gross",
    "disgusting": "gross",
    "revulsion": "gross",
    "repulsed": "gross",
}


def parse_emotion_from_text(text: str) -> Optional[str]:
    """
    Text-only self evaluator.

    It reads only the generated text and maps it to one of:
    happy, sad, angry, fear, surprise, neutral, gross.

    Returns None if no valid emotion can be detected.
    """

    if text is None:
        return None

    norm = normalize_text(text)

    if len(norm) == 0:
        return None

    # 1. If output is exactly one valid label/synonym
    if norm in SYNONYM_TO_LABEL:
        return SYNONYM_TO_LABEL[norm]

    # 2. Prefer phrases like "answer is X", "emotion is X", "label is X"
    priority_patterns = [
        r"(?:answer|emotion|label|prediction)\s+(?:is|:)?\s+(happy|sad|angry|fear|fearful|afraid|scared|surprise|surprised|neutral|gross|disgust|disgusted)",
        r"(?:the person is|person appears|looks)\s+(happy|sad|angry|fearful|afraid|scared|surprised|neutral|disgusted)",
    ]

    for pat in priority_patterns:
        m = re.search(pat, norm)
        if m:
            word = m.group(1)
            return SYNONYM_TO_LABEL.get(word)

    # 3. Find all emotion/synonym occurrences
    found = []

    tokens = norm.split()

    for tok in tokens:
        if tok in SYNONYM_TO_LABEL:
            found.append(SYNONYM_TO_LABEL[tok])

    # Also catch multi-word-like content after normalization
    for syn, label in SYNONYM_TO_LABEL.items():
        if re.search(rf"\b{re.escape(syn)}\b", norm):
            found.append(label)

    if not found:
        return None

    # Remove duplicates while preserving order
    ordered = []
    for x in found:
        if x not in ordered:
            ordered.append(x)

    # If only one emotion detected, use it
    if len(ordered) == 1:
        return ordered[0]

    # If multiple emotions detected, choose the last one.
    # Often model says: "not sad, it is happy" or explains then final answer.
    return ordered[-1]


# ============================================================
# GENERATION
# ============================================================

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


@torch.no_grad()
def generate_for_video(model, processor, video_path, prompt_messages):
    frames = load_video_frames(video_path, T=T_FRAMES)

    if frames is None:
        return None, "frame_load_failed"

    prompt_str = prompt_to_text(processor, prompt_messages)

    batch = processor(
        text=[prompt_str],
        videos=[list(frames)],
        return_tensors="pt",
        padding=True,
    )

    device = next(model.parameters()).device
    inputs = to_device_dtype(batch, device)

    input_len = inputs["input_ids"].shape[1]

    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            use_cache=True,
        )

        gen_ids = output_ids[0, input_len:]
        gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        del inputs, batch, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return gen_text, "ok"

    except Exception as e:
        del inputs, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return None, f"generation_failed: {e}"


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    shard_id = args.shard_id
    num_shards = args.num_shards

    print("=" * 80)
    print("DFEW LLaVA GENERATION BASELINE")
    print("=" * 80)
    print(f"Shard: {shard_id}/{num_shards}")
    print(f"Model: {MODEL_ID}")
    print(f"Output dir: {OUT_DIR}")

    all_records = load_all_test_records()
    all_records = sorted(all_records, key=lambda r: (r["fold"], r["video_id"]))

    records = [
        r for i, r in enumerate(all_records)
        if i % num_shards == shard_id
    ]

    print(f"Total records: {len(all_records)}")
    print(f"Records in this shard: {len(records)}")

    print("\nLoading model...")
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

    rows = []

    for r in tqdm(records, desc=f"generation shard {shard_id}/{num_shards}"):
        gen_text, status = generate_for_video(
            model=model,
            processor=processor,
            video_path=r["video_path"],
            prompt_messages=PROMPT_MAIN,
        )

        parsed = parse_emotion_from_text(gen_text) if gen_text is not None else None
        used_retry = False
        retry_text = None
        retry_status = None
        retry_parsed = None

        if parsed is None and DO_RETRY_IF_UNPARSED:
            used_retry = True

            retry_text, retry_status = generate_for_video(
                model=model,
                processor=processor,
                video_path=r["video_path"],
                prompt_messages=PROMPT_RETRY,
            )

            retry_parsed = parse_emotion_from_text(retry_text) if retry_text is not None else None

            if retry_parsed is not None:
                parsed = retry_parsed

        correct = int(parsed == r["gt_label"]) if parsed is not None else 0

        rows.append({
            "fold": r["fold"],
            "video_id": r["video_id"],
            "video_name": r["video_name"],
            "video_path": r["video_path"],
            "gt_label": r["gt_label"],
            "dfew_label_id": r["dfew_label_id"],
            "generation_text": gen_text,
            "generation_status": status,
            "parsed_label": parsed,
            "used_retry": used_retry,
            "retry_text": retry_text,
            "retry_status": retry_status,
            "retry_parsed_label": retry_parsed,
            "correct": correct,
            "unparsed": int(parsed is None),
        })

        # save periodically
        if len(rows) % 100 == 0:
            tmp_path = os.path.join(
                OUT_DIR,
                f"generation_predictions_shard_{shard_id}_of_{num_shards}.csv",
            )
            pd.DataFrame(rows).to_csv(tmp_path, index=False)

    out_path = os.path.join(
        OUT_DIR,
        f"generation_predictions_shard_{shard_id}_of_{num_shards}.csv",
    )

    pd.DataFrame(rows).to_csv(out_path, index=False)

    print("\nSaved:", out_path)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()