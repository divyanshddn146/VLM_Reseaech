# README — DFEW Emotion Recognition with Frozen LLaVA-NeXT-Video Hidden States

## Overview

This code evaluates whether frozen hidden states from **LLaVA-NeXT-Video-7B** contain useful information for video emotion recognition on the DFEW dataset.

Main model:

```text
llava-hf/LLaVA-NeXT-Video-7B-hf
````

The pipeline is:

1. Run LLaVA-NeXT-Video on DFEW videos.
2. Extract final-token hidden states from the language-model layers.
3. Train lightweight classifiers on frozen hidden states.
4. Evaluate using WAR/accuracy and UAR.
5. Compare hidden-state classifiers against the model’s direct text-generation baseline.

No full VLM fine-tuning is performed.

---

## Important note before running

Several scripts contain **absolute file paths** from my local/HPC environment.

Before running the code, please update the path variables near the top of each script, especially:

```text
DFEW_ROOT
DFEW_SPLIT_ROOT
DFEW_ORIGINAL_DIR
OUT_DIR
CACHE_DIR
RESULT_DIR
```

These should be changed to match the dataset location, hidden-state cache location, and output directory on the machine where the code is being run.

---

# Files included

## 1. `01_extract_hidden_states_and_linear_probe.py`

### Purpose

This is the main LLaVA-NeXT-Video script for hidden-state extraction and linear probing.

It does two main things:

1. Extracts hidden states from LLaVA-NeXT-Video.
2. Trains and evaluates a linear probe on the saved hidden states.

The script supports three modes:

```text
extract_only
probe_only
extract_and_probe
```

Recommended workflow:

```text
1. Run extract_only to save hidden states.
2. Run probe_only to train and evaluate the linear probe without reloading the full VLM.
```

### Main result

Best Set 1 linear-probe result:

```text
Layer 26 linear probe:
71.7% WAR / 59.3% UAR
```

This shows that useful emotion-recognition information is linearly recoverable from frozen LLaVA-NeXT hidden states.

---

## 2. `02_evaluate_linear_probe_uar.py`

### Purpose

This is a post-processing script for the linear-probe predictions from `01_extract_hidden_states_and_linear_probe.py`.

It computes:

```text
Accuracy / WAR
UAR
Per-class recall
Best layer by WAR
Best layer by UAR
```

### Expected result

```text
L26:
~71.7% WAR / ~59.3% UAR
```

This script is used to confirm the linear-probe baseline.

---

## 3. `03_run_5fold_hidden_probe_models.py`

### Purpose

This is the main final 5-fold evaluation script.

It evaluates selected hidden-state classifier configurations across the official DFEW folds.

Each run handles:

```text
one fold + one method
```

This makes it suitable for Slurm array jobs.

### Methods evaluated

The 8 evaluated methods are:

```text
1. accuracy_mlp_concat_L16_20_26
2. bridge_mlp_concat_L20_22_26
3. weighted_mlp_L20
4. weighted_mlp_concat_key_layers
5. weighted_mlp_concat_L16_20_26
6. weighted_mlp_L22
7. balanced_sampler_mlp_L26_p025
8. tradeoff_mlp_mean_L20_31_p000
```

### What it does

For each fold and method, the script:

1. Loads the official DFEW train/test split.
2. Creates a validation split from the training set.
3. Loads the saved LLaVA-NeXT hidden states.
4. Builds the selected feature representation.
5. Trains the selected lightweight classifier.
6. Evaluates on the official test split.
7. Saves fold-level summary and per-class results.

This is the main script for the final hidden-state probing evaluation.

---

## 4. `04_aggregate_5fold_probe_results.py`

### Purpose

This script aggregates the outputs from `03_run_5fold_hidden_probe_models.py`.

It computes:

```text
Mean WAR/accuracy across 5 folds
Standard deviation of WAR
Mean UAR across 5 folds
Standard deviation of UAR
Mean per-class recall
Combined score = (WAR + UAR) / 2
```

### Main final 5-fold hidden-state results

| Result type   | Method                           | WAR / Accuracy |        UAR |
| ------------- | -------------------------------- | -------------: | ---------: |
| Best WAR      | `accuracy_mlp_concat_L16_20_26`  |     71.5 ± 1.3 | 59.7 ± 1.5 |
| Best tradeoff | `weighted_mlp_concat_key_layers` |     70.4 ± 1.3 | 61.7 ± 2.2 |
| Best UAR      | `balanced_sampler_mlp_L26_p025`  |     64.8 ± 1.5 | 62.3 ± 1.6 |

The best accuracy-focused model uses an MLP on concatenated hidden states from:

```text
L16, L20, L26
```

The best tradeoff model uses a weighted MLP on key layers:

```text
L13, L16, L20, L22, L26, L31
```

The best UAR-focused model uses balanced sampling on:

```text
L26
```

---

## 5. `05_run_direct_generation_baseline.py`

### Purpose

This script runs the direct generation baseline for LLaVA-NeXT-Video.

Instead of training a classifier on hidden states, it asks the VLM to directly generate an emotion label from the video.

The model is prompted to answer with one label from:

```text
happy, sad, angry, fear, surprise, neutral, gross
```

### What it does

For each video in the DFEW test folds, the script:

1. Loads the video frames.
2. Prompts LLaVA-NeXT-Video to generate one emotion label.
3. Parses the generated text into one of the seven emotion classes.
4. If the output cannot be parsed, reruns the same video with a stricter label-only prompt.
5. Saves prediction shards for later aggregation.

This baseline measures how well the model performs when used normally as a text-generating VLM.

---

## 6. `06_aggregate_generation_baseline.py`

### Purpose

This script aggregates the direct generation baseline outputs from `05_run_direct_generation_baseline.py`.

It reads the generated prediction shards and computes:

```text
Fold-level WAR/accuracy
Fold-level UAR
Per-class recall
Final 5-fold generation baseline result
```

This script does not run LLaVA-NeXT itself. It only evaluates the saved generation predictions.

`06_aggregate_generation_baseline.py` should be used after `05_run_direct_generation_baseline.py`.

---

# Recommended running order

## Step 1 — Extract hidden states

Run:

```text
01_extract_hidden_states_and_linear_probe.py
```

with:

```text
RUN_MODE=extract_only
```

This saves the frozen LLaVA-NeXT hidden states.

---

## Step 2 — Run linear probe

Run:

```text
01_extract_hidden_states_and_linear_probe.py
```

with:

```text
RUN_MODE=probe_only
```

Then run:

```text
02_evaluate_linear_probe_uar.py
```

to compute UAR and identify the best linear-probe layer.

Expected result:

```text
Best Set 1 linear probe:
Layer 26
~71.7% WAR / ~59.3% UAR
```

---

## Step 3 — Run final 5-fold hidden-state probing

Run:

```text
03_run_5fold_hidden_probe_models.py
```

across the 5 folds and 8 selected methods.

Then run:

```text
04_aggregate_5fold_probe_results.py
```

to aggregate the final 5-fold results.

Expected final results:

```text
Best WAR:
accuracy_mlp_concat_L16_20_26
~71.5% WAR / ~59.7% UAR

Best tradeoff:
weighted_mlp_concat_key_layers
~70.4% WAR / ~61.7% UAR

Best UAR:
balanced_sampler_mlp_L26_p025
~64.8% WAR / ~62.3% UAR
```

---

## Step 4 — Run direct generation baseline

Run:

```text
05_run_direct_generation_baseline.py
```

to generate emotion-label predictions from LLaVA-NeXT-Video directly.

Then run:

```text
06_aggregate_generation_baseline.py
```

to compute the 5-fold generation baseline metrics.

This gives the direct-output comparison against the hidden-state classifiers.

---

# Metrics

DFEW papers commonly report:

```text
WAR = Weighted Average Recall = overall accuracy
UAR = Unweighted Average Recall = macro average of per-class recall
```

In this code:

```text
accuracy ≈ WAR
uar = macro recall across the seven emotion classes
```

The seven labels used are:

```text
happy, sad, angry, fear, surprise, neutral, gross
```

The DFEW `disgust` label is mapped to:

```text
gross
```

---

# Short interpretation

The results suggest that frozen LLaVA-NeXT-Video hidden states contain useful emotion-recognition information.

A simple linear probe reaches:

```text
71.7% WAR / 59.3% UAR
```

on DFEW Set 1.

The final 5-fold hidden-state probing result reaches:

```text
71.5% WAR
```

without fine-tuning the full VLM.

Class-balanced training improves UAR and minority-class recall, especially for difficult classes such as `fear` and `gross`, but reduces overall WAR.

The strongest practical tradeoff is:

```text
weighted_mlp_concat_key_layers:
70.4% WAR / 61.7% UAR
```

The direct generation baseline is included to compare hidden-state classifiers against the model’s normal text-output behavior.

