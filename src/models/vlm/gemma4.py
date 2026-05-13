"""Gemma 4 VLM quantization and loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

MODEL_ID = "google/gemma-4-E4B-it"
SAVE_SUBDIR = "gemma-4-E4B-it-4bit"
SAVE_SUBDIR_BF16 = "gemma-4-E4B-it-16bit"


def bnb_4bit_config() -> BitsAndBytesConfig:
    """NF4 double-quant bitsandbytes config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def quantize_and_save(
    model_id: str = MODEL_ID,
    output_dir: Path | str = Path("models/pretrained/vlm") / SAVE_SUBDIR,
    device_map: str = "auto",
) -> None:
    """Load model_id with 4-bit NF4 quantization and save to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_id!r} with 4-bit NF4 quantization ...")
    config = bnb_4bit_config()

    model: PreTrainedModel = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        quantization_config=config,
        device_map=device_map,
        dtype=torch.bfloat16,
    )
    processor: ProcessorMixin = AutoProcessor.from_pretrained(model_id)

    print(f"Saving quantized model to {output_dir} ...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Done. Quantized model saved → {output_dir}")


def load_quantized(
    model_dir: Path | str,
    device_map: str = "auto",
) -> tuple[PreTrainedModel, ProcessorMixin]:
    """Return (model, processor) from a previously saved 4-bit quantized model."""
    model_dir = Path(model_dir)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_dir),
        device_map=device_map,
        dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(str(model_dir))
    return model, processor


def download_and_save(
    model_id: str = MODEL_ID,
    output_dir: Path | str = Path("models/pretrained/vlm") / SAVE_SUBDIR_BF16,
    device_map: str = "auto",
) -> None:
    """Load model_id in bfloat16 (no quantization) and save to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_id!r} in bfloat16 (no quantization) ...")
    model: PreTrainedModel = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=torch.bfloat16,
    )
    processor: ProcessorMixin = AutoProcessor.from_pretrained(model_id)

    print(f"Saving model to {output_dir} ...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Done. Model saved → {output_dir}")


def load_bf16(
    model_dir: Path | str,
    device_map: str = "auto",
) -> tuple[PreTrainedModel, ProcessorMixin]:
    """Return (model, processor) from a previously saved bfloat16 model."""
    model_dir = Path(model_dir)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_dir),
        device_map=device_map,
        dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(str(model_dir))
    return model, processor
