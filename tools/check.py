#!/usr/bin/env python3
"""
Quick visual QA for lung mask overlays, with a pleural‑biased outline for
presentation. Saves figures under ATM22/figs/<case>_overlay.png

Usage:
  python check.py ATM_147_0000
  python check.py  # runs a small demo list
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    binary_dilation,
    disk,
)
from scipy.ndimage import binary_fill_holes, binary_erosion
from skimage.segmentation import find_boundaries
from skimage.measure import label
import json

DATA_ROOT = Path("./ATM22/npy_files")
FIG_DIR = Path("./ATM22/figs")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def lung_window_auto(img2d: np.ndarray) -> np.ndarray:
    """Robust display windowing.
    - If values look HU-like, use a standard lung window (W≈1500, L≈-600).
    - Otherwise, use percentile [p1, p99] scaling to avoid washed-out images.
    Returns float image in [0,1].
    """
    x = img2d.astype(np.float32)
    p1, p50, p99 = np.percentile(x, [1, 50, 99])
    hu_like = (p1 < -700) and (p99 > 100) and (p50 < 200)
    if hu_like:
        lo, hi = -1350.0, 150.0
    else:
        lo, hi = p1, p99
        if hi - lo < 1e-3:
            lo, hi = p50 - 1.0, p50 + 1.0
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-6)


def clean_lung_mask_2d(mask2d: np.ndarray, sy: float, sx: float) -> np.ndarray:
    """Keep two largest components, aggressively remove tiny islands, close gaps.
    Area thresholds are spacing-aware (mm^2 -> pixels). This returns a cleaned
    mask but does not force-fill very large cavitations; visualization can fill.
    """
    m = mask2d.astype(bool)
    # spacing-aware thresholds (tuned for ~0.5–1.0 mm in-plane)
    min_island_mm2 = 300.0
    hole_fill_mm2 = 3000.0  # larger to suppress medium holes in 2D view
    px_area = max(sy * sx, 1e-6)
    m = remove_small_objects(m, min_size=int(round(min_island_mm2 / px_area)))
    m = remove_small_holes(m, area_threshold=int(round(hole_fill_mm2 / px_area)))
    # gentle morphological closing (~1 mm) to seal thin gaps to background
    rad = max(1, int(round(1.0 / max(1e-6, min(sy, sx)))))
    m = binary_closing(m, footprint=disk(rad))
    # keep two largest components
    lab = label(m, connectivity=2)
    if lab.max() > 2:
        areas = np.array([(lab == i).sum() for i in range(1, lab.max() + 1)])
        keep = np.argsort(areas)[-2:] + 1
        m = np.isin(lab, keep)
    return m


def pick_slice(mask: np.ndarray) -> int:
    areas = np.sum(mask > 0, axis=(1, 2))
    return int(np.argmax(areas)) if areas.max() > 0 else mask.shape[0] // 2


def slice_extent(case_id: str, ny: int, nx: int):
    meta = DATA_ROOT / "imagesTr" / f"{case_id}_meta.json"
    if meta.exists():
        spacing = json.loads(meta.read_text()).get("processed_spacing", [1.0, 1.0, 1.0])
        sx = float(spacing[0])  # mm along x (columns)
        sy = float(spacing[1])  # mm along y (rows)
    else:
        sy = sx = 1.0
    return [0, sx * nx, sy * ny, 0]  # left, right, bottom, top (mm)


def load_case(case_id: str):
    img = np.load(DATA_ROOT / "imagesTr" / f"{case_id}.npy")
    lung = np.load(DATA_ROOT / "lungsTr" / f"{case_id}.npy")
    return img, lung


def outer_perimeter(mask2d: np.ndarray) -> np.ndarray:
    # One-pixel outer perimeter of a binary mask
    return find_boundaries(mask2d.astype(np.uint8), mode="outer", background=0)


def pleural_perimeter(mask2d: np.ndarray, img2d: np.ndarray, grow_mm: float, sy: float, sx: float) -> np.ndarray:
    """Visualization-only perimeter: expand mask outward by ~grow_mm, clamp
    to a coarse thoracic body region estimated from intensities, then take a
    one‑pixel perimeter. This is for thesis figures only.
    """
    m = mask2d.astype(bool)
    m = remove_small_objects(m, min_size=1000)
    m = binary_closing(m, footprint=disk(2))
    m = binary_fill_holes(m)
    # coarse body via robust percentile
    p10, p50 = np.percentile(img2d.astype(np.float32), [10, 50])
    thr = p10 + 0.1 * (p50 - p10)
    body = img2d > thr
    # convert mm to pixels
    rad = max(1, int(round(grow_mm / max(1e-6, min(sy, sx)))))
    grown = binary_dilation(m, footprint=disk(rad)) & body # clamp to body
    return np.logical_xor(grown, binary_erosion(grown))

def coarse_body_mask_2d(img2d: np.ndarray) -> np.ndarray:
    """Very coarse thoracic body mask from robust percentiles on a single slice.
    Used to constrain visualization-only sealing so we don't bleed into the background.
    """
    p10, p50 = np.percentile(img2d.astype(np.float32), [10, 50])
    thr = p10 + 0.1 * (p50 - p10) # conservative threshold
    return img2d > thr

def air_candidate_2d(img2d: np.ndarray, body2d: np.ndarray) -> np.ndarray:
    """Estimate air-like pixels on a slice using robust thresholds.
    If HU-like, use ~-320 HU. Otherwise, use a percentile of intensities inside body.
    Returns a boolean mask.
    """
    x = img2d.astype(np.float32)
    vals = x[body2d]
    if vals.size == 0:
        return np.zeros_like(x, dtype=bool)
    p1, p50, p99 = np.percentile(vals, [1, 50, 99])
    hu_like = (p1 < -700) and (p99 > 100) and (p50 < 200)
    if hu_like:
        thr = -320.0
    else:
        # picks a conservative low-intensity percentile inside body as "air", using 35% to avoid excessive false positives
        thr = np.percentile(vals, 35)
    return (x <= thr) & body2d


def visualize(case_id: str):
    vol = np.load(DATA_ROOT / "imagesTr" / f"{case_id}.npy")
    mask = np.load(DATA_ROOT / "lungsTr" / f"{case_id}.npy")

    z = pick_slice(mask)
    # Build extent and spacing first for spacing-aware cleaning
    ext = slice_extent(case_id, vol.shape[1], vol.shape[2])
    sy = (ext[2] - ext[3]) / vol.shape[1]
    sx = (ext[1] - ext[0]) / vol.shape[2]

    img2d = lung_window_auto(vol[z])
    m2d = mask[z] > 0
    m2d_c = clean_lung_mask_2d(m2d, sy=sy, sx=sx)
    # For visualization, seal thin tunnels to exterior (~2.5 mm) within coarse body,
    # then fill cavities to avoid internal red outlines on diseased lungs.
    rad_close_mm = 2.5
    rad_close = max(1, int(round(rad_close_mm / max(1e-6, min(sy, sx)))))
    sealed = binary_closing(m2d_c, footprint=disk(rad_close))
    body2d = coarse_body_mask_2d(vol[z])
    sealed &= body2d

    # Add near-pleura air pixels (within ~4 mm of current mask) to cover undersegmented lobes
    rad_expand_mm = 4.0
    rad_expand = max(1, int(round(rad_expand_mm / max(1e-6, min(sy, sx)))))
    band = binary_dilation(sealed, footprint=disk(rad_expand))
    air2d = air_candidate_2d(vol[z], body2d)
    refined = sealed | (air2d & band)

    # Finally fill residual cavities, which are now definitely internal
    refined = binary_fill_holes(refined)
    m2d_vis = binary_fill_holes(refined)
    perim = outer_perimeter(m2d_vis)

    ext = slice_extent(case_id, img2d.shape[0], img2d.shape[1])
    sy = (ext[2] - ext[3]) / img2d.shape[0]
    sx = (ext[1] - ext[0]) / img2d.shape[1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img2d, cmap="gray", origin="upper", extent=ext, interpolation="none", aspect="equal")
    # fill, semi-transparent
    ax.imshow(np.ma.masked_where(~m2d_vis, m2d_vis), cmap="autumn", alpha=0.18, origin="upper", extent=ext)
    # exact cleaned outer outline (red), solid
    ax.contour(perim.astype(np.uint8), levels=[0.5], colors="r", linewidths=1.2,
               linestyles="solid", origin="upper", extent=ext)
    # pleural‑biased outline (yellow, dashed)
    # pleu = pleural_perimeter(m2d_c, vol[z], grow_mm=2.0, sy=sy, sx=sx)
    # ax.contour(pleu.astype(np.uint8), levels=[0.5], colors="y", linewidths=1.2,
    #            linestyles="dashed", origin="upper", extent=ext)

    # scale bar and cosmetics, assuming ~0.5–1.0 mm spacing
    x1 = ext[1] - 60
    x2 = ext[1] - 10
    y = ext[2] - 20
    ax.plot([x1, x2], [y, y], lw=3, color="w")
    ax.text((x1 + x2) / 2, y + 6, "50 mm", color="w", ha="center", va="bottom",
            fontsize=9, bbox=dict(facecolor="black", alpha=0.3, pad=1, edgecolor="none"))

    ax.set_title(f"Axial slice {z} with lung mask overlay for {case_id}")
    ax.axis("off")
    out = FIG_DIR / f"{case_id}_overlay.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    cases = sys.argv[1:] or ["ATM_503_0000"]
    for cid in cases:
        visualize(cid)


if __name__ == "__main__":
    main()
