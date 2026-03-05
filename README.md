# RoadState-VLM

> **Neuro-Symbolic Structured Road Perception via Geometric Chain-of-Thought and Discrete State Transitions**

[![Paper](https://img.shields.io/badge/Paper-KDD%202026-blue)](https://github.com/eadst/KDD-2026-RoadState-VLM)
[![Model](https://img.shields.io/badge/Base%20Model-Qwen3--VL--8B--Thinking-orange)](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
[![Framework](https://img.shields.io/badge/Framework-LlamaFactory-green)](https://github.com/hiyouga/LLaMA-Factory)
[![Status](https://img.shields.io/badge/Status-Under%20Review-yellow)]()

This repository accompanies the paper:

**RoadState-VLM: Neuro-Symbolic Structured Road Perception via Geometric Chain-of-Thought and Discrete State Transitions** *(KDD 2026, under review)*

📦 Repository: **https://github.com/eadst/KDD-2026-RoadState-VLM**

---

## Overview

RoadState-VLM is a vision-language model fine-tuned for **structured road scene understanding** in autonomous driving. Given two consecutive front-view frames (a previous frame and a current frame) along with the previous road state, the model performs:

1. **Geometric Chain-of-Thought Reasoning** — step-by-step scene observation, temporal analysis, and logical deduction enclosed within `<think>...</think>`.
2. **Discrete State Transition** — maintains a symbolic road state machine across frames, inferring the current structured state from incremental visual evidence.

The output is a structured JSON object describing the current road state, including lane count, ego-vehicle position, and multi-dimensional lane-change feasibility.

---

## Task Definition

### Input
| Item | Description |
|------|-------------|
| **Previous Frame** | The immediately preceding dashcam image |
| **Current Frame** | The current dashcam image |
| **Previous State** | The structured road state of the previous frame (JSON) |

### Output State Schema

```json
{
  "lane_count": "<integer: total number of same-direction motor vehicle lanes>",
  "ego_lane_index": "<integer: 1-indexed from left, >50% occupancy rule during lane change>",
  "lane_changeability": {
    "struct":   { "left": "yes/no", "right": "yes/no" },
    "vehicle":  { "left": "yes/no", "right": "yes/no" },
    "obstacle": { "left": "yes/no", "right": "yes/no" },
    "all":      { "left": "yes/no", "right": "yes/no" }
  }
}
```

### Feasibility Dimensions
| Dimension | Description |
|-----------|-------------|
| `struct` | Road boundaries, lane markings (solid/dashed), guardrails, curbs |
| `vehicle` | Adjacent vehicles (occupying, paralleling, or approaching) |
| `obstacle` | Pedestrians, cyclists, construction, roadblocks |
| `all` | **AND** of all three dimensions; `"yes"` only if all three are `"yes"` |

### Key Rules
- **Lane counting**: `separator_count + 1` (same-direction separators only; excludes road boundaries, non-motor lanes)
- **Ego position**: numbered left-to-right from 1; during lane change, the lane with >50% occupancy is used
- **Temporal stability**: maintain previous state unless clear visual evidence of change exists

---

## Reasoning Workflow

The model follows a structured three-step Chain-of-Thought inside `<think>`:

```
Step 1 — Global Scene Observation
  Describe road type, boundaries, markings (color/type),
  traffic distribution, and obstacles.

Step 2 — Temporal & Dynamic Analysis
  Compare previous and current frames:
  - Identify ego-motion and lateral shifts
  - Detect dynamic triggers (new objects, marking transitions)
  - Apply stability bias unless clear evidence of change

Step 3 — Logical Deduction
  Derive lane_count → ego_lane_index → feasibility per dimension
  Apply 3-dimension AND logic for the "all" field
```

After `</think>`, the model outputs **only** the final JSON (no extra text).

---

## Repository Structure

```
RoadState-VLM/
├── README.md
├── data_example/
│   ├── images/                  # Sample dashcam image pairs (image1.png – image5.png)
│   ├── dataset/
│   │   └── train_sft_cot_pred.jsonl   # Example SFT training data (CoT + prediction)
│   └── prompt/
│       └── train.example.txt    # System & user prompt template
└── LlamaFactory/
    ├── train/
    │   └── train.yaml           # LoRA SFT training configuration
    ├── merge/
    │   └── merge.yaml           # LoRA adapter merge configuration
    └── infer/
        └── infer.yaml           # Inference configuration
```

---

## Model & Training

### Base Model
- **[Qwen/Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)**

### Training Configuration (`LlamaFactory/train/train.yaml`)

| Parameter | Value |
|-----------|-------|
| Stage | SFT (Supervised Fine-Tuning) |
| Fine-tuning method | LoRA (`lora_rank: 8`, `lora_target: all`) |
| Thinking mode | Disabled (`enable_thinking: false`) |
| Template | `qwen3_vl` |
| Dataset | `roadstate_dataset` |
| Image max pixels | 1,568,000 |
| Cutoff length | 8,192 tokens |
| Batch size | 1 (×4 gradient accumulation = effective 4) |
| Learning rate | 1e-5 (cosine scheduler, warmup 0.1) |
| Epochs | 2 |
| Precision | BF16 |

### Merge LoRA Adapter (`LlamaFactory/merge/merge.yaml`)

```yaml
model_name_or_path: /Qwen/Qwen3-VL-8B-Thinking
adapter_name_or_path: /saves/qwen3-vl-8b-thinking
export_dir: saves/qwen3_vl_thinking_merged
export_size: 5
export_device: cpu
```

### Inference (`LlamaFactory/infer/infer.yaml`)

```yaml
model_name_or_path: saves/qwen3_vl_thinking_merged
template: qwen3_vl
infer_backend: huggingface
```

---

## Quick Start

### 1. Install LlamaFactory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 2. Prepare Dataset

Place your dataset (in LlamaFactory multi-modal SFT format) under `data/` and register it in `data/dataset_info.json` with the name `roadstate_dataset`.

Refer to `data_example/dataset/train_sft_cot_pred.jsonl` and `data_example/prompt/train.example.txt` for data format and prompt structure.

### 3. Train

```bash
llamafactory-cli train LlamaFactory/train/train.yaml
```

### 4. Merge LoRA Adapter

```bash
llamafactory-cli export LlamaFactory/merge/merge.yaml
```

### 5. Inference

```bash
llamafactory-cli chat LlamaFactory/infer/infer.yaml
```

---

## Data Example

The `data_example/` directory contains:
- **5 sample dashcam images** (`image1.png` – `image5.png`) illustrating typical road scenarios
- **A sample JSONL file** (`train_sft_cot_pred.jsonl`) with one CoT+prediction training entry
- **The full prompt template** (`train.example.txt`) used for SFT, including system instructions, few-shot state definitions, reasoning workflow, and output format constraints

---

## Citation

> BibTeX will be provided upon publication.

If you use this work before formal publication, please cite the preprint or reference the GitHub repository:

```
@misc{roadstate-vlm-2026,
  title  = {RoadState-VLM: Neuro-Symbolic Structured Road Perception via
             Geometric Chain-of-Thought and Discrete State Transitions},
  year   = {2026},
  note   = {KDD 2026, under review. \url{https://github.com/eadst/KDD-2026-RoadState-VLM}}
}
```

---

## Status

🚧 **Work in progress** — codebase, full dataset, and model checkpoints will be released in phases after the review process.

| Component | Status |
|-----------|--------|
| Data examples & prompt template | ✅ Released |
| LlamaFactory training configs | ✅ Released |
| Full training dataset | 🔜 Coming soon |
| Model checkpoint (merged) | 🔜 Coming soon |
| Evaluation code | 🔜 Coming soon |
