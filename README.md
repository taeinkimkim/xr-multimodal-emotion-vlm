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
