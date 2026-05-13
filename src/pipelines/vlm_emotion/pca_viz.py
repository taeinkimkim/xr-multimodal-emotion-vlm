"""PCA visualization for DINOv2 features."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

EMOTION_COLORS = {
    "Happiness": "#FFD700",
    "Sadness":   "#4169E1",
    "Anger":     "#DC143C",
    "Surprise":  "#FF8C00",
    "Fear":      "#800080",
    "Disgust":   "#2E8B57",
    "Neutral":   "#808080",
}


class PCAViz:
    def __init__(self) -> None:
        from sklearn.decomposition import PCA as _PCA
        self._pca = _PCA(n_components=2, random_state=42)
        self._feat2d: np.ndarray | None = None
        self._labels: list[str] = []
        self._var_ratio: tuple[float, float] = (0.0, 0.0)

    def fit(self, features: np.ndarray, labels: list[str]) -> None:
        """Fit PCA on (N, D) feature matrix and cache 2D projections."""
        self._feat2d = self._pca.fit_transform(features)
        self._labels = labels
        self._var_ratio = (
            float(self._pca.explained_variance_ratio_[0]),
            float(self._pca.explained_variance_ratio_[1]),
        )

    def plot(self, current_idx: int) -> Image.Image:
        """Return a PIL Image of the 2D scatter with current_idx highlighted."""
        if self._feat2d is None:
            raise RuntimeError("Call fit() before plot()")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4), dpi=110)

        unique_labels = sorted(set(self._labels))
        for label in unique_labels:
            mask = np.array([lb == label for lb in self._labels])
            color = EMOTION_COLORS.get(label, "#888888")
            ax.scatter(
                self._feat2d[mask, 0], self._feat2d[mask, 1],
                c=color, label=label, alpha=0.35, s=16, zorder=2,
            )

        from matplotlib.legend_handler import HandlerPathCollection

        cx, cy = self._feat2d[current_idx]
        curr_sc = ax.scatter(cx, cy, c="black", s=220, marker="*", zorder=5, label="current")
        ax.annotate(
            "current", (cx, cy),
            xytext=(6, 4), textcoords="offset points", fontsize=7,
        )

        ax.legend(
            fontsize=6, markerscale=1.5, loc="best",
            handler_map={curr_sc: HandlerPathCollection(numpoints=1, sizes=[28])},
        )
        ax.set_xlabel(f"PC1 ({self._var_ratio[0]:.1%})")
        ax.set_ylabel(f"PC2 ({self._var_ratio[1]:.1%})")
        ax.set_title("DINOv2 Feature Space (PCA)")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()


class PatchPCAViz:
    """Two-stage patch-level PCA RGB visualization with cross-image color consistency.

    Call fit() once on a representative set of images to fix the PCA basis,
    then call plot() per image — same semantic region → same color across images.

    Stage 1: PCA(3) on all patches → PC1 separates background from foreground.
    Stage 2: PCA(3) on foreground patches only → 3 components mapped to RGB.
    """

    def __init__(self) -> None:
        self._pca1 = None
        self._pca2 = None
        self._pc1_min: float = 0.0
        self._pc1_max: float = 1.0
        self._feat2_p5: np.ndarray | None = None   # 5th-percentile per channel
        self._feat2_p95: np.ndarray | None = None  # 95th-percentile per channel
        self._fg_is_high_pc1: bool = True  # auto-detected in fit()
        self._bg_threshold: float = 0.2
        self._patch_size: int = 14

    @property
    def is_fitted(self) -> bool:
        return self._pca1 is not None and self._pca2 is not None

    def fit(
        self,
        backbone: object,
        pixel_values_list: list,
        patch_size: int = 14,
        bg_threshold: float = 0.35,
    ) -> None:
        """Fit PCA on patch tokens pooled from multiple images.

        Args:
            backbone: HuggingFace DINOv2 AutoModel (eval mode).
            pixel_values_list: list of (1, C, H, W) tensors, already preprocessed.
            patch_size: patch size of the ViT backbone (default 14).
            bg_threshold: PC1 threshold (after min-max norm) to separate background.
        """
        import torch
        from sklearn.decomposition import PCA

        self._patch_size = patch_size
        self._bg_threshold = bg_threshold

        all_patches: list[np.ndarray] = []
        with torch.no_grad():
            for pv in pixel_values_list:
                out = backbone(pixel_values=pv)
                all_patches.append(out.last_hidden_state[0, 1:].cpu().float().numpy())

        patches = np.concatenate(all_patches, axis=0)  # (N_total, hidden_size)

        # Stage 1: fit on all patches
        self._pca1 = PCA(n_components=3, random_state=42)
        feat1 = self._pca1.fit_transform(patches)
        pc1 = feat1[:, 0]
        self._pc1_min = float(pc1.min())
        self._pc1_max = float(pc1.max())
        pc1_norm = (pc1 - self._pc1_min) / (self._pc1_max - self._pc1_min + 1e-8)

        # Auto-detect PC1 orientation using center vs edge heuristic.
        # Face-centered images: center patches = face, edge patches = background.
        # If center mean PC1 > edge mean PC1 → face has high PC1 → fg = high PC1.
        _, _, H_px, W_px = pixel_values_list[0].shape
        H_p = H_px // patch_size
        W_p = W_px // patch_size
        ph = np.arange(H_p).repeat(W_p)
        pw = np.tile(np.arange(W_p), H_p)
        r = max(1, min(H_p, W_p) // 4)
        center_mask = (np.abs(ph - H_p // 2) <= r) & (np.abs(pw - W_p // 2) <= r)
        edge_mask   = (ph == 0) | (ph == H_p - 1) | (pw == 0) | (pw == W_p - 1)

        # Tile the per-patch masks across all images
        n_imgs = len(pixel_values_list)
        center_mask_all = np.tile(center_mask, n_imgs)
        edge_mask_all   = np.tile(edge_mask,   n_imgs)

        self._fg_is_high_pc1 = bool(
            pc1_norm[center_mask_all].mean() >= pc1_norm[edge_mask_all].mean()
        )

        # Stage 2: fit on foreground patches only
        fg = pc1_norm > bg_threshold if self._fg_is_high_pc1 else pc1_norm <= (1.0 - bg_threshold)

        self._pca2 = PCA(n_components=3, random_state=42)
        feat2 = self._pca2.fit_transform(patches[fg])
        # Percentile-based normalization for better color contrast
        self._feat2_p5  = np.percentile(feat2, 5,  axis=0)
        self._feat2_p95 = np.percentile(feat2, 95, axis=0)

    def plot(
        self,
        backbone: object,
        pixel_values: object,
        patch_size: int | None = None,
        saturation: float = 2.5,
    ) -> Image.Image:
        """Return a patch-level PCA RGB image using the fitted PCA basis.

        Falls back to per-image PCA if fit() has not been called.

        Args:
            backbone: HuggingFace DINOv2 AutoModel (eval mode).
            pixel_values: float tensor of shape (1, C, H, W), already preprocessed.
            patch_size: overrides the patch_size set during fit().
            saturation: PIL Color enhancement factor (1.0 = no change, >1 = more vivid).
        Returns:
            PIL Image (H, W, 3) where each patch is colored by its 3 PCA components.
        """
        import torch
        from PIL import ImageEnhance

        ps = patch_size or self._patch_size

        with torch.no_grad():
            outputs = backbone(pixel_values=pixel_values)

        patch_tokens = outputs.last_hidden_state[0, 1:].cpu().float().numpy()

        _, _, H_px, W_px = pixel_values.shape
        H_p, W_p = H_px // ps, W_px // ps

        if self.is_fitted:
            # Use shared PCA basis → consistent colors across images
            feat1 = self._pca1.transform(patch_tokens)
            pc1 = feat1[:, 0]
            pc1_norm = (pc1 - self._pc1_min) / (self._pc1_max - self._pc1_min + 1e-8)
            fg = pc1_norm > self._bg_threshold if self._fg_is_high_pc1 else pc1_norm <= (1.0 - self._bg_threshold)

            feat2 = self._pca2.transform(patch_tokens[fg])
            feat2 = (feat2 - self._feat2_p5) / (self._feat2_p95 - self._feat2_p5 + 1e-8)
            feat2 = np.clip(feat2, 0.0, 1.0)
        else:
            # Fallback: per-image PCA with auto-orientation
            from sklearn.decomposition import PCA
            feat1 = PCA(n_components=3, random_state=42).fit_transform(patch_tokens)
            pc1 = feat1[:, 0]
            pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)

            ph = np.arange(H_p).repeat(W_p)
            pw = np.tile(np.arange(W_p), H_p)
            r = max(1, min(H_p, W_p) // 4)
            center_mask = (np.abs(ph - H_p // 2) <= r) & (np.abs(pw - W_p // 2) <= r)
            edge_mask   = (ph == 0) | (ph == H_p - 1) | (pw == 0) | (pw == W_p - 1)
            fg_is_high  = bool(pc1_norm[center_mask].mean() >= pc1_norm[edge_mask].mean())
            fg = pc1_norm > self._bg_threshold if fg_is_high else pc1_norm <= (1.0 - self._bg_threshold)

            feat2 = PCA(n_components=3, random_state=42).fit_transform(patch_tokens[fg])
            for i in range(3):
                p5, p95 = np.percentile(feat2[:, i], [5, 95])
                feat2[:, i] = np.clip((feat2[:, i] - p5) / (p95 - p5 + 1e-8), 0.0, 1.0)

        rgb = np.zeros((H_p * W_p, 3), dtype=np.float32)
        rgb[fg] = feat2

        rgb_img = Image.fromarray((rgb.reshape(H_p, W_p, 3) * 255).astype(np.uint8))
        rgb_img = rgb_img.resize((W_px, H_px), Image.NEAREST)
        return ImageEnhance.Color(rgb_img).enhance(saturation)
