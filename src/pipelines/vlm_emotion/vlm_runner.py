"""VLM inference runner for direct and vision-assisted emotion experiments."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.vlm.gemma4 import load_quantized

EMOTION_LABELS = ("Happiness", "Sadness", "Anger", "Surprise", "Fear", "Disgust", "Neutral")

_LABEL_RE = re.compile(
    r"Emotion\s*:\s*(" + "|".join(EMOTION_LABELS) + r")\b",
    re.IGNORECASE,
)

_DIRECT_PROMPT = """\
Look at this facial image and identify the person's emotion.

Choose exactly one emotion from: Happiness, Sadness, Anger, Surprise, Fear, Disgust, Neutral.

Reply in this exact format:
Emotion: <label>
Reasoning: <2–3 sentences describing the facial features you observed>\
"""

_ASSISTED_PROMPT_TMPL = """\
A vision model has analyzed this facial image and provided the following context:
- Predicted emotion: {vision_label}
- Feature summary: {feature_summary}

Using this context alongside your own analysis, identify the most likely emotion.

Choose exactly one emotion from: Happiness, Sadness, Anger, Surprise, Fear, Disgust, Neutral.

Reply in this exact format:
Emotion: <label>
Reasoning: <2–3 sentences describing your analysis>\
"""


def _format_features(cls_token: np.ndarray, top_k: int = 10) -> str:
    """Convert a CLS token vector to a compact text description."""
    norm = float(np.linalg.norm(cls_token))
    mean = float(cls_token.mean())
    std  = float(cls_token.std())
    top_indices = np.argsort(np.abs(cls_token))[-top_k:][::-1]
    top_vals = ", ".join(f"dim{i}:{cls_token[i]:+.3f}" for i in top_indices)
    return f"norm={norm:.3f}, mean={mean:.4f}, std={std:.4f} | top activations: [{top_vals}]"


class Gemma4Runner:
    def __init__(
        self,
        model_dir: Path | str,
        device_map: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        self.model, self.processor = load_quantized(model_dir, device_map=device_map)
        self.max_new_tokens = max_new_tokens
        self.model.eval()

    def run_direct(self, image_path: Path | str) -> dict:
        """Exp 1: image → Gemma 4 → {response, predicted_label}."""
        return self._generate(image_path, _DIRECT_PROMPT)

    def run_vision_assisted(
        self,
        image_path: Path | str,
        vision_label: str,
        cls_token: np.ndarray,
    ) -> dict:
        """Exp 3: image + vision_label + cls_token → Gemma 4 → {response, predicted_label}."""
        prompt = _ASSISTED_PROMPT_TMPL.format(
            vision_label=vision_label,
            feature_summary=_format_features(cls_token),
        )
        return self._generate(image_path, prompt)

    def _generate(self, image_path: Path | str, prompt: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))

        messages = [
            {
                "role": "system",
                "content": "You are an emotion recognition assistant.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        raw_response = self.processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=False,
        )

        try:
            response = self.processor.parse_response(raw_response).strip()
        except Exception:
            response = raw_response.strip()

        return {"response": response, "predicted_label": _parse_emotion(response)}


def _parse_emotion(text: str) -> str | None:
    match = _LABEL_RE.search(text)
    if not match:
        return None
    raw = match.group(1)
    for label in EMOTION_LABELS:
        if label.lower() == raw.lower():
            return label
    return None
