"""DINOv2 facial emotion classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class Dinov2EmotionClassifier(nn.Module):
    """DINOv2 backbone with a small classification head."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        pretrained_dir: Path | None = None,
    ) -> None:
        super().__init__()
        model_source = _resolve_pretrained_source(model_name, pretrained_dir)
        cache_dir = str(pretrained_dir) if pretrained_dir is not None else None
        self.config = AutoConfig.from_pretrained(model_source, cache_dir=cache_dir)
        self.backbone = AutoModel.from_pretrained(model_source, cache_dir=cache_dir)
        if pretrained_dir is not None and model_source == model_name:
            pretrained_dir.mkdir(parents=True, exist_ok=True)
            self.backbone.save_pretrained(pretrained_dir, safe_serialization=False)

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"Model {model_name!r} does not expose config.hidden_size")

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        if use_lora:
            self.backbone = _apply_lora(
                self.backbone,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = _pool_dinov2_output(outputs)
        return self.classifier(pooled)


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts."""

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def _resolve_pretrained_source(model_name: str, pretrained_dir: Path | None) -> str:
    if pretrained_dir is None:
        return model_name

    has_config = (pretrained_dir / "config.json").exists()
    has_weights = any(
        (pretrained_dir / filename).exists()
        for filename in ("pytorch_model.bin", "model.safetensors")
    )
    return str(pretrained_dir) if has_config and has_weights else model_name


def _pool_dinov2_output(outputs: object) -> torch.Tensor:
    pooler_output = getattr(outputs, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output

    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is None:
        raise ValueError("DINOv2 output does not include last_hidden_state")
    return last_hidden_state[:, 0]


def _apply_lora(
    backbone: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
) -> nn.Module:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:
        raise ImportError("Install peft to train with --use-lora") from error

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["query", "value"],
    )
    return get_peft_model(backbone, config)
