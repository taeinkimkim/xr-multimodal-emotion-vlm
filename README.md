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
  --pretrained-dir models/pretrained/face/dinov2 \
  --pool-mode pooler \
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
  --pretrained-dir models/pretrained/face/dinov2 \
  --pool-mode pooler \
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
  --pretrained-dir models/pretrained/face/dinov2 \
  --pool-mode pooler \
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
  models/trained/face/dinov2_affectnet/model_lora_ep5_bs16/history.json \
  --title-name "DINOv2 AffectNet"
```

This creates two PNG files next to the history JSON:

```text
plots.png
class_metrics.png
```

## Quantize Gemma 4 E4B-it
Install the dependencies:

```bash
pip install -r requirements.txt
```

Download `google/gemma-4-E4B-it` from Hugging Face and save a 4-bit NF4
quantized copy to disk:

```bash
python3 scripts/quantize_gemma4.py \
  --model-id google/gemma-4-E4B-it \
  --output-dir models/pretrained/vlm/gemma-4-E4B-it-4bit \
  --device-map auto
```

To load the saved model in code:

```python
from src.models.vlm.gemma4 import load_quantized

model, processor = load_quantized("models/pretrained/vlm/gemma-4-E4B-it-4bit")
```

## Download Gemma 4 E2B-it (bfloat16, no quantization)
Install the dependencies:

```bash
pip install -r requirements.txt
```

Download `google/gemma-4-E2B-it` from Hugging Face and save the original
bfloat16 model to disk:

```bash
python3 scripts/download_gemma4.py \
  --model-id google/gemma-4-E2B-it \
  --output-dir models/pretrained/vlm/gemma-4-E2B-it-16bit \
  --device-map auto
```

To load the saved model in code:

```python
from src.models.vlm.gemma4 import load_bf16

model, processor = load_bf16("models/pretrained/vlm/gemma-4-E2B-it-16bit")
```

## Experiments

### Exp 1 — Direct VLM
A baseline experiment that feeds face images directly into the VLM to infer
emotion, with no vision model involved.

```
face image → VLM → emotion
```
```bash
python3 scripts/run_exp01_vlm_direct.py \
  --test-dir data/raw/face/balanced_rafdb/test \
  --vlm-model-dir models/pretrained/vlm/gemma-4-E2B-it-16bit \
  --prompt-id 2 \
  --max-new-tokens 512
```

### Exp 2 — Vision Model
Runs a vision model alone to classify facial emotion.
Measures the standalone accuracy of the vision model as a reference.

```
face image → Vision Model → emotion
```
```bash
python3 scripts/run_exp02_vision.py \
  --test-dir data/raw/face/balanced_rafdb/test \
  --model-dir models/trained/face/dinov2_balanced_rafdb/model_lora_ep10_bs16 \
  --backbone-pretrained-dir models/pretrained/face/dinov2 \
  --pool-mode pooler \
  --lora-r 8 --lora-alpha 16
```

### Exp 3 — Vision-Assisted VLM
The vision model's prediction is passed as additional context to the VLM.
Tests whether the vision model's prior judgment improves the VLM's final
emotion prediction.

```
face image → Vision Model → prediction
face image + prediction → VLM → emotion
```
```bash
python3 scripts/run_exp03_vision_assisted_vlm.py \
  --test-dir data/raw/face/balanced_rafdb/test \
  --vision-model-dir models/trained/face/dinov2_balanced_rafdb/model_lora_ep10_bs16 \
  --vlm-model-dir models/pretrained/vlm/gemma-4-E2B-it-16bit \
  --backbone-pretrained-dir models/pretrained/face/dinov2 \
  --prompt-id 2 \
  --max-new-tokens 512 \
  --pool-mode pooler \
  --lora-r 8 --lora-alpha 16
```

### Visualize Results
Launches a Gradio UI to browse experiment results sample by sample.
Displays the input image, predicted emotion, VLM response, and a
patch-level PCA visualization of the vision model's feature space.

```bash
python3 scripts/visualize_exp_results.py \
  --vision-model-dir models/trained/face/dinov2_balanced_rafdb/model_lora_ep10_bs16 \
  --vlm-model-dir models/pretrained/vlm/gemma-4-E2B-it-16bit
```
