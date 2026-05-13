"""Vision model inference runner: classification + CLS token extraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from src.models.face.dinov2_emotion import Dinov2EmotionClassifier, _pool_dinov2_output


class Dinov2EmotionRunner:
    def __init__(
        self,
        model_dir: Path | str,
        backbone_name: str = "facebook/dinov2-base",
        backbone_pretrained_dir: Path | str | None = Path("models/pretrained/face/dinov2"),
        lora_r: int = 8,
        lora_alpha: int = 16,
        device: str = "cuda",
    ) -> None:
        model_dir = Path(model_dir)

        with (model_dir / "labels.json").open(encoding="utf-8") as f:
            label_data = json.load(f)
        self.id2label: dict[int, str] = {int(k): v for k, v in label_data["id2label"].items()}
        self.label2id: dict[str, int] = label_data["label2id"]
        num_labels = len(self.id2label)

        self.device = device

        pretrained_dir = Path(backbone_pretrained_dir) if backbone_pretrained_dir else None
        processor_source = (
            str(pretrained_dir)
            if pretrained_dir is not None
            and (pretrained_dir / "preprocessor_config.json").exists()
            else backbone_name
        )
        self.processor = AutoImageProcessor.from_pretrained(processor_source)

        self.model = Dinov2EmotionClassifier(
            model_name=backbone_name,
            num_labels=num_labels,
            freeze_backbone=True,
            use_lora=True,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            pretrained_dir=pretrained_dir,
        )
        state_dict = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        print(f"Dinov2EmotionRunner loaded from {model_dir} (device={device})")

    @torch.no_grad()
    def run(self, image_path: Path | str) -> dict:
        """
        Returns:
            predicted_label (str), confidence (float),
            cls_token (np.ndarray, shape [hidden_dim]),
            probs (dict[str, float])
        """
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device)

        backbone_out = self.model.backbone(pixel_values=pixel_values)
        cls_token = _pool_dinov2_output(backbone_out)          # (1, hidden)
        logits = self.model.classifier(cls_token)              # (1, num_labels)

        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(probs.argmax())

        return {
            "predicted_label": self.id2label[pred_idx],
            "confidence": float(probs[pred_idx]),
            "cls_token": cls_token[0].cpu().numpy().astype(np.float32),
            "probs": {self.id2label[i]: float(p) for i, p in enumerate(probs)},
        }
