# xr-multimodal-emotion-vlm
Multimodal emotion recognition for XR environments using vision-language models.

## Repository Structure
```
xr-multimodal-emotion-vlm/
├── configs/
│   ├── data/
│   │   ├── face.yaml
│   │   ├── audio.yaml          # (future)
│   │   ├── gaze_pupil.yaml     # (future)
│   │   └── head_motion.yaml    # (future)
│   │
│   ├── model/
│   │   ├── gemma4.yaml
│   │   ├── vision_lora.yaml
│   │   └── vision_backbone.yaml
│   │
│   └── experiment/
│       ├── face_only.yaml
│       ├── audio_only.yaml     # (future)
│       └── multimodal_fusion.yaml  # (future)
│
├── data/
│   ├── raw/
│   │   ├── face/
│   │   ├── audio/              # (future)
│   │   ├── gaze_pupil/         # (future)
│   │   └── head_motion/        # (future)
│   ├── processed/
│   │   ├── face/
│   │   ├── audio/              # (future)
│   │   ├── gaze_pupil/         # (future)
│   │   └── head_motion/        # (future)
│   └── annotations/
│       └── labels.csv
│
├── models/               
│   └── pretrained/
│       ├── face/
│       ├── audio/              # (future)
│       ├── gaze_pupil/         # (future)
│       ├── head_motion/        # (future)
│       └── vlm/
│
├── src/
│   ├── data/
│   │   ├── face/
│   │   ├── audio/              # (future)
│   │   ├── gaze_pupil/         # (future)
│   │   ├── head_motion/        # (future)
│   │   └── multimodal_dataset.py   # (future)
│   │
│   ├── models/
│   │   ├── face/
│   │   ├── audio/              # (future)
│   │   ├── gaze_pupil/         # (future)
│   │   ├── head_motion/        # (future)
│   │   ├── fusion/             # (future)
│   │   └── vlm/
│   │
│   ├── pipelines/
│   │   ├── run_unimodal.py
│   │   ├── run_multimodal_fusion.py  # (future)
│   │   └── run_vlm_inference.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       └── report.py
│
├── scripts/
│
├── experiments/
│   └── face/
│       ├── exp01_gemma4_direct/
│       ├── exp02_vision_lora/
│       └── exp03_vision_assisted/
│
├── results/
│   └── face/
│       ├── metrics/
│       ├── predictions/
│       └── figures/
│
└── notebooks/
    └── face_analysis.ipynb
```

## Download AffectNet
AffectNet face data can be downloaded from Hugging Face with:

```bash
pip install huggingface_hub
python3 scripts/download_affectnet.py "owner/affectnet"
```

Replace `"owner/affectnet"` with the actual Hugging Face dataset repo ID.
For example, if the dataset URL is `https://huggingface.co/datasets/owner/affectnet`,
the repo ID is `owner/affectnet`.

By default, files are saved to:

```text
data/raw/face/affectnet/
```

## Download RAF-DB
RAF-DB face data can be downloaded from Kaggle with:

```bash
pip install kagglehub
python3 scripts/download_rafdb.py "owner/raf-db-dataset"
```

Replace `"owner/raf-db-dataset"` with the actual Kaggle dataset ID.
For example, if the dataset URL is
`https://www.kaggle.com/datasets/owner/raf-db-dataset`, the dataset ID is
`owner/raf-db-dataset`.

By default, files are saved to:

```text
data/raw/face/rafdb/
```

To overwrite an existing local download:

```bash
python3 scripts/download_rafdb.py "owner/raf-db-dataset" --force-download
```

## Analyze AffectNet
Check label counts and image-size distributions before training:

```bash
python3 scripts/analyze_affectnet.py \
  --dataset data/raw/face/affectnet \
  --validation-ratio 0.1 \
  --json-output results/face/affectnet_summary.json
```

## Analyze RAF-DB
Check label counts and image-size distributions before training:

```bash
python3 scripts/analyze_rafdb.py \
  --dataset data/raw/face/rafdb \
  --validation-ratio 0.1 \
  --json-output results/face/rafdb_summary.json
```

## Train DINOv2 on Face Emotion Data
Install the training dependencies:

```bash
pip install -r requirements.txt
```

Then train a frozen-backbone DINOv2 emotion classifier on AffectNet:

```bash
python3 -m src.pipelines.train_face_dinov2 \
  --dataset data/raw/face/affectnet \
  --dataset-type affectnet \
  --validation-ratio 0.1 \
  --model-name facebook/dinov2-base \
  --epochs 5 \
  --batch-size 16 \
  --output-dir models/trained/face/dinov2_affectnet \
  --model-filename model_ep5_bs16.pt
```

To train on RAF-DB:

```bash
python3 -m src.pipelines.train_face_dinov2 \
  --dataset data/raw/face/rafdb \
  --dataset-type rafdb \
  --validation-ratio 0.1 \
  --model-name facebook/dinov2-base \
  --epochs 5 \
  --batch-size 16 \
  --output-dir models/trained/face/dinov2_rafdb \
  --model-filename model_ep5_bs16.pt
```

The model uses Hugging Face `AutoImageProcessor` and `AutoModel` to load
`facebook/dinov2-base`, takes the CLS token as the image feature, and trains a
small classification head for the detected dataset labels.

All supported face-emotion datasets are normalized to the same label ids before
training:

```text
0: Happiness
1: Sadness
2: Anger
3: Surprise
4: Fear
5: Disgust
6: Neutral
```

To fine-tune DINOv2 with LoRA later:

```bash
python3 -m src.pipelines.train_face_dinov2 \
  --dataset data/raw/face/affectnet \
  --dataset-type affectnet \
  --validation-ratio 0.1 \
  --model-name facebook/dinov2-base \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --epochs 5 \
  --batch-size 16 \
  --lr 0.0001 \
  --output-dir models/trained/face/dinov2_affectnet \
  --model-filename model_lora_ep5_bs16.pt
```

Use `--unfreeze-backbone` only when you want full backbone fine-tuning. For a
first baseline, keep the backbone frozen; for the next step, use LoRA.

## Visualize Training Metrics
After training, plot the saved history file:

```bash
python3 scripts/plot_training_metrics.py \
  models/trained/face/dinov2_affectnet/model_lora_ep5_bs16/history.json
```

This creates two PNG files next to the history JSON:

```text
plots.png
class_metrics.png
```

Use custom output paths if needed:

```bash
python3 scripts/plot_training_metrics.py \
  models/trained/face/dinov2_affectnet/model_lora_ep5_bs16/history.json \
  --output models/trained/face/dinov2_affectnet/model_lora_ep5_bs16/plots.png \
  --class-output models/trained/face/dinov2_affectnet/model_lora_ep5_bs16/class_metrics.png
```


## Experiments
### 1. Direct VLM
```
face video → Gemma 4 → emotion
```
### 2. Vision Model (LoRA)
```
face video → Vision Model (LoRA) → emotion
```
### 3. Vision-Assisted VLM
```
face video → Vision Model (LoRA) → prediction
face video + prediction → Gemma 4 → emotion
```
