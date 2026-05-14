#!/usr/bin/env python3
"""Gradio UI: browse VLM emotion experiment results sample by sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.vlm_emotion.pca_viz import PatchPCAViz
from src.pipelines.vlm_emotion.vision_runner import Dinov2EmotionRunner

# ─────────────────────────── helpers (stateless) ─────────────────────

def _load_json(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _accuracy(results: list[dict] | None, key: str = "predicted_label") -> str:
    if not results:
        return "—"
    n = len(results)
    correct = sum(r.get(key) == r["true_label"] for r in results)
    return f"{correct / n:.1%}  ({correct}/{n})"


def _safe_get(results: list[dict] | None, idx: int) -> dict | None:
    return results[idx] if (results and idx < len(results)) else None


# ─────────────────────────── global state (set in main) ──────────────

exp1: list[dict] | None = None
exp2: list[dict] | None = None
exp3: list[dict] | None = None
vision_runner: Dinov2EmotionRunner | None = None
patch_pca: PatchPCAViz = PatchPCAViz()
master: list[dict] = []
N: int = 0
METRICS_MD: str = ""


# ─────────────────────────── sample update logic ─────────────────────

def update_sample(idx: int) -> tuple[Any, ...]:
    idx = int(idx) % N if N else 0

    row = master[idx] if master else {}
    image_path = row.get("image_path")
    true_label = row.get("label") or row.get("true_label", "—")

    r1 = _safe_get(exp1, idx)
    exp1_pred     = (r1 or {}).get("predicted_label") or "—"
    exp1_response = (r1 or {}).get("response") or "(Exp 1 not run)"

    r2 = _safe_get(exp2, idx)
    exp2_pred = (r2 or {}).get("predicted_label") or "—"
    exp2_conf = f"{(r2 or {}).get('confidence', 0):.1%}" if r2 else "—"

    r3 = _safe_get(exp3, idx)
    exp3_vision_input = (r3 or {}).get("vision_predicted_label") or "—"
    exp3_pred         = (r3 or {}).get("predicted_label") or "—"
    exp3_response     = (r3 or {}).get("response") or "(Exp 3 not run)"

    pca_img = None
    if vision_runner is not None and image_path:
        from PIL import Image as PILImage
        img = PILImage.open(image_path).convert("RGB")
        pixel_values = vision_runner.processor(images=img, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(vision_runner.device)
        pca_img = patch_pca.plot(vision_runner.model.backbone, pixel_values)

    return (
        image_path, true_label,
        exp1_pred, exp1_response,
        exp2_pred, exp2_conf, pca_img,
        exp3_vision_input, exp3_pred, pca_img, exp3_response,
    )


def _navigate(idx: int, delta: int = 0, rand: bool = False) -> list[Any]:
    new_idx = int(np.random.randint(N)) if rand else (int(idx) + delta) % N
    return [new_idx, *update_sample(new_idx)]


# ─────────────────────────── Gradio UI ───────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500&display=swap');
* {
    font-family: 'Noto Sans KR', sans-serif !important;
}
"""


def build_demo(vlm_name: str, vision_name: str) -> Any:
    import gradio as gr

    with gr.Blocks(title="VLM Emotion Experiments", css=_CSS) as demo:
        gr.Markdown("# VLM Emotion Experiment Results")
        gr.Markdown(METRICS_MD)

        # ── Navigator ──────────────────────────────────────────────
        with gr.Row():
            prev_btn = gr.Button("◀ Prev",    scale=1)
            idx_box  = gr.Number(label=f"Sample index  (0 – {N - 1})", value=0,
                                 precision=0, scale=3)
            next_btn = gr.Button("Next ▶",    scale=1)
            rand_btn = gr.Button("🎲 Random", scale=1)

        # ── Top row: image + true label ────────────────────────────
        with gr.Row():
            image_out      = gr.Image(label="Input Image", scale=2, height=300)
            true_label_out = gr.Textbox(label="True Label", scale=1)

        # ── Experiment tabs ────────────────────────────────────────
        with gr.Tabs():
            with gr.TabItem(f"Exp 1 · {vlm_name} Direct"):
                exp1_pred_out     = gr.Textbox(label=f"{vlm_name} Predicted Emotion")
                exp1_response_out = gr.Textbox(label=f"{vlm_name} Response", lines=7)

            with gr.TabItem(f"Exp 2 · {vision_name}"):
                with gr.Row():
                    exp2_pred_out = gr.Textbox(label="Predicted Emotion", scale=1)
                    exp2_conf_out = gr.Textbox(label="Confidence",         scale=1)
                pca2_out = gr.Image(label=f"{vision_name} Patch-level PCA", height=380)

            with gr.TabItem(f"Exp 3 · {vision_name} + {vlm_name}"):
                exp3_vision_input_out = gr.Textbox(
                    label=f"{vision_name} prediction fed into {vlm_name}")
                with gr.Row():
                    exp3_pred_out = gr.Textbox(label=f"{vlm_name} Predicted Emotion", scale=1)
                    pca3_out      = gr.Image(label=f"{vision_name} Patch-level PCA",
                                             scale=2, height=350)
                exp3_response_out = gr.Textbox(label=f"{vlm_name} Response", lines=7)

        all_outputs = [
            image_out, true_label_out,
            exp1_pred_out, exp1_response_out,
            exp2_pred_out, exp2_conf_out, pca2_out,
            exp3_vision_input_out, exp3_pred_out, pca3_out, exp3_response_out,
        ]
        nav_outputs = [idx_box] + all_outputs

        next_btn.click(fn=lambda i: _navigate(i, delta=+1), inputs=idx_box, outputs=nav_outputs)
        prev_btn.click(fn=lambda i: _navigate(i, delta=-1), inputs=idx_box, outputs=nav_outputs)
        rand_btn.click(fn=lambda i: _navigate(i, rand=True), inputs=idx_box, outputs=nav_outputs)
        idx_box.submit(fn=update_sample, inputs=idx_box, outputs=all_outputs)
        demo.load(fn=lambda: update_sample(0), outputs=all_outputs)

    return demo


# ─────────────────────────── entry point ─────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vision-model-dir", type=Path,
        default=Path("models/trained/face/dinov2_balanced_rafdb/model_lora_ep10_bs16"),
        help="Vision model directory used in the experiments (default: %(default)s)",
    )
    parser.add_argument(
        "--vlm-model-dir", type=Path,
        default=Path("models/pretrained/vlm/gemma-4-E4B-it-4bit"),
        help="VLM model directory used in the experiments (default: %(default)s)",
    )
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    return parser.parse_args()


def main() -> None:
    global exp1, exp2, exp3, vision_runner, master, N, METRICS_MD

    args = parse_args()

    vision_model_dir = args.vision_model_dir
    vlm_model_dir    = args.vlm_model_dir

    # Derive model name labels (match the naming used in experiment scripts)
    vision_name = vision_model_dir.parent.name   # e.g. "dinov2_balanced_rafdb"
    vlm_name    = vlm_model_dir.name             # e.g. "gemma-4-E4B-it-4bit"

    # Reconstruct experiment output paths (mirrors each runner script)
    exp_dirs = {
        1: Path("experiments/face/exp01_vlm_direct") / vlm_model_dir.name,
        2: (Path("experiments/face/exp02_vision")
            / vision_model_dir.parent.name
            / vision_model_dir.name),
        3: (Path("experiments/face/exp03_vision_assisted_vlm")
            / f"{vision_model_dir.parent.name}+{vlm_model_dir.name}"
            / vision_model_dir.name),
    }

    # Load results
    exp1 = _load_json(exp_dirs[1] / "results.json")
    exp2 = _load_json(exp_dirs[2] / "results.json")
    exp3 = _load_json(exp_dirs[3] / "results.json")

    master = exp2 or exp1 or exp3 or []
    N = len(master)

    # Load vision backbone and fit patch PCA for cross-image color consistency
    try:
        import torch
        from PIL import Image as PILImage

        vision_runner = Dinov2EmotionRunner(
            model_dir=vision_model_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        max_fit = min(len(master), 200)
        fit_indices = (
            np.random.choice(len(master), max_fit, replace=False)
            if len(master) > max_fit
            else range(len(master))
        )
        print(f"Fitting patch PCA on {max_fit} images ...")
        pixel_values_list = []
        for i in fit_indices:
            img_path = master[i].get("image_path")
            if img_path:
                img = PILImage.open(img_path).convert("RGB")
                pv = vision_runner.processor(images=img, return_tensors="pt")["pixel_values"]
                pixel_values_list.append(pv.to(vision_runner.device))
        patch_pca.fit(vision_runner.model.backbone, pixel_values_list)
        print("Patch PCA fitted.")
    except Exception as e:
        print(f"Warning: could not load vision runner for PCA viz: {e}")

    # Metrics table
    METRICS_MD = (
        "| Experiment | Accuracy |\n"
        "|---|---|\n"
        f"| Exp 1 — {vlm_name} Direct | {_accuracy(exp1)} |\n"
        f"| Exp 2 — {vision_name} | {_accuracy(exp2)} |\n"
        f"| Exp 3 — {vision_name} + {vlm_name} ({vlm_name}) | {_accuracy(exp3)} |\n"
        f"| Exp 3 — {vision_name} + {vlm_name} ({vision_name}) | {_accuracy(exp3, 'vision_predicted_label')} |\n"
    )

    if N == 0:
        print(
            "No experiment results found for the given model paths.\n"
            f"  Exp 1 expected: {exp_dirs[1]}\n"
            f"  Exp 2 expected: {exp_dirs[2]}\n"
            f"  Exp 3 expected: {exp_dirs[3]}\n"
            "Run the corresponding experiment scripts first."
        )
        return

    print(f"Loaded {N} samples.")
    print(f"  vision : {vision_name}")
    print(f"  vlm    : {vlm_name}")
    print(f"Launching UI → http://localhost:{args.port}")

    demo = build_demo(vlm_name=vlm_name, vision_name=vision_name)
    demo.launch(server_port=args.port, share=args.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
