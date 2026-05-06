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

## Analyze AffectNet
Check label counts and image-size distributions before training:

```bash
python3 scripts/analyze_affectnet.py \
  --dataset data/raw/face/affectnet
```

To save the analysis as JSON:

```bash
python3 scripts/analyze_affectnet.py \
  --dataset data/raw/face/affectnet \
  --json-output results/face/affectnet_summary.json
```

## Train DINOv2 on AffectNet
Install the training dependencies:

```bash
pip install -r requirements.txt
```

Then train a frozen-backbone DINOv2 emotion classifier:

```bash
python3 -m src.pipelines.train_face_dinov2 \
  --dataset data/raw/face/affectnet \
  --validation-ratio 0.1 \
  --model-name facebook/dinov2-base \
  --epochs 5 \
  --batch-size 16 \
  --model-filename model_ep5_bs16.pt
```

The model uses Hugging Face `AutoImageProcessor` and `AutoModel` to load
`facebook/dinov2-base`, takes the CLS token as the image feature, and trains a
small classification head for AffectNet labels.

If the Hugging Face AffectNet repo uses different column names, pass them
explicitly:

```bash
python3 -m src.pipelines.train_face_dinov2 \
  --dataset data/raw/face/affectnet \
  --image-column image \
  --label-column label
```

To fine-tune DINOv2 with LoRA later:

```bash
python3 -m src.pipelines.train_face_dinov2 \
  --dataset data/raw/face/affectnet \
  --validation-ratio 0.1 \
  --model-name facebook/dinov2-base \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --epochs 5 \
  --batch-size 16 \
  --lr 0.0001 \
  --model-filename model_lora_ep5_bs16.pt
```

Use `--unfreeze-backbone` only when you want full backbone fine-tuning. For a
first baseline, keep the backbone frozen; for the next step, use LoRA.

## Visualize Training Metrics
After training, plot the saved history file:

```bash
python3 scripts/plot_training_metrics.py \
  models/trained/face/dinov2_affectnet/model_lora_ep5_bs16_history.json
```

This creates two PNG files next to the history JSON:

```text
model_lora_ep5_bs16_plots.png
model_lora_ep5_bs16_class_metrics.png
```

Use custom output paths if needed:

```bash
python3 scripts/plot_training_metrics.py \
  models/trained/face/dinov2_affectnet/model_lora_ep5_bs16_history.json \
  --output models/trained/face/dinov2_affectnet/model_lora_ep5_bs16_plots.png \
  --class-output models/trained/face/dinov2_affectnet/model_lora_ep5_bs16_class_metrics.png
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
