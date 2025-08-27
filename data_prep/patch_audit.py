import os
import json
import gzip
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm
from scipy import ndimage
# skan (optional) for accurate skeleton geodesic lengths
try:
    from skan.csr import Skeleton  # type: ignore
except Exception:
    Skeleton = None  # type: ignore
# skeletonize_3d fallback (skimage >=0.23 may not expose 3D; 2D skeletonize slice-wise works acceptably)
try:
    from skimage.morphology import skeletonize_3d as _skeletonize_3d  # type: ignore
    def _skeletonize_3d_safe(arr: np.ndarray) -> np.ndarray:
        return _skeletonize_3d(arr)
except Exception:  # Fallback to 2D skeletonize applied per-slice
    from skimage.morphology import skeletonize as _skeletonize_2d  # type: ignore
    def _skeletonize_3d_safe(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 3:
            raise ValueError("Expected 3D array for skeletonization")
        out = np.zeros_like(arr, dtype=np.uint8)
        for z in range(arr.shape[0]):
            out[z] = _skeletonize_2d(arr[z] > 0).astype(np.uint8)
        return out


def _read_spacing_zyx_from_meta(meta_json_path: Path) -> Optional[Tuple[float, float, float]]:
    try:
        with open(meta_json_path, "r") as f:
            meta = json.load(f)
        # SimpleITK spacing is (x, y, z). Our arrays are (z, y, x) as saved in convert_to_numpy.py
        sp = meta.get("processed_spacing")
        if sp and isinstance(sp, (list, tuple)) and len(sp) == 3:
            x, y, z = float(sp[0]), float(sp[1]), float(sp[2])
            return (z, y, x)
    except Exception:
        pass
    return None

def _generate_patch_positions(shape_zyx: Tuple[int, int, int], patch_size: Tuple[int, int, int], stride: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    z_max = max(1, shape_zyx[0] - patch_size[0] + 1)
    y_max = max(1, shape_zyx[1] - patch_size[1] + 1)
    x_max = max(1, shape_zyx[2] - patch_size[2] + 1)
    positions: List[Tuple[int, int, int]] = []
    for z in range(0, z_max, stride[0]):
        for y in range(0, y_max, stride[1]):
            for x in range(0, x_max, stride[2]):
                positions.append((z, y, x))
    #include the tail if not divisible by stride
    tail_z = z_max - 1
    tail_y = y_max - 1
    tail_x = x_max - 1
    if positions and positions[-1] != (tail_z, tail_y, tail_x):
        positions.append((tail_z, tail_y, tail_x))
    return positions


def _extract_patch(volume_zyx: np.ndarray, start_zyx: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> np.ndarray:
    z0, y0, x0 = start_zyx
    dz, dy, dx = patch_size
    return volume_zyx[z0:z0+dz, y0:y0+dy, x0:x0+dx]


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    p = probs[(probs > 0) & np.isfinite(probs)]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p + eps)).sum())


class PatchBankManager:
    def __init__(self) -> None:
        pass
    def save_patch_bank(self, patch_bank: List[Dict], output_path: Path) -> None:
        # Try to infer patch_size/stride from any descriptor, else leave None
        inferred_ps = None
        inferred_stride = None
        # Look for optional hints in descriptors
        for p in patch_bank:
            if isinstance(p.get("position_zyx"), (list, tuple)):
                # position alone doesn't encode size; use auditor's defaults if embedded later
                pass
        meta = {
            "total_patches": len(patch_bank),
            "fg_patches": int(sum(1 for p in patch_bank if p.get("fg_fraction", 0.0) > 0.0)),
            # These fields may be filled by caller context if present
            "patch_size": getattr(self, "_patch_size", None),
            "stride": getattr(self, "_stride", None),
        }
        bank = {"patch_descriptors": patch_bank, "metadata": meta}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(str(output_path), "wb") as f:
            pickle.dump(bank, f, protocol=pickle.HIGHEST_PROTOCOL)


class PatchAuditSystem:
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        stride: Tuple[int, int, int] = (32, 32, 32),
        min_lung_coverage: float = 0.10,
        max_patches_per_case: Optional[int] = None,
        use_skan: bool = False,
    ) -> None:
        self.patch_size = patch_size
        self.stride = stride
        self.min_lung_coverage = float(min_lung_coverage)
        self.max_patches_per_case = max_patches_per_case
        self.use_skan = bool(use_skan)

    # ---- Descriptors start here ----
    def compute_fg_fraction_unweighted(self, label_patch: np.ndarray) -> float:
        total = float(label_patch.size)
        if total <= 0:
            return 0.0
        return float((label_patch > 0).sum()) / total

    def compute_fg_fraction_center_weighted(self, label_patch: np.ndarray) -> float:
        #computes the fg fraction of the patch weighted by the distance from the center of the patch
        if (label_patch > 0).sum() == 0:
            return 0.0
        zc, yc, xc = np.array(label_patch.shape) // 2
        zz, yy, xx = np.ogrid[:label_patch.shape[0], :label_patch.shape[1], :label_patch.shape[2]]
        dist = np.sqrt((zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2)
        maxd = np.sqrt(zc ** 2 + yc ** 2 + xc ** 2) + 1e-6
        weights = np.exp(-0.5 * (dist / (0.3 * maxd)) ** 2)
        wf = float(((label_patch > 0) * weights).sum())
        wt = float(weights.sum()) + 1e-6
        return wf / wt

    def compute_components_metrics(self, label_patch: np.ndarray) -> Dict[str, float]:
        fg = (label_patch > 0).astype(np.uint8)
        if fg.sum() == 0:
            return {
                "n_components": 0.0,
                "lcc_fraction": 0.0,
                "size_entropy": 0.0,
            }
        # Use 26-connectivity for 3D airways
        lab, num = ndimage.label(fg, structure=ndimage.generate_binary_structure(3, 3))
        sizes = np.bincount(lab.ravel())
        sizes = sizes[1:] if sizes.size > 0 else sizes  # drop background
        total = float(sizes.sum()) + 1e-6
        lcc = float(sizes.max()) if sizes.size > 0 else 0.0
        probs = (sizes.astype(np.float64) / total) if total > 0 else np.array([])
        return {
            "n_components": float(int(num)),
            "lcc_fraction": (lcc / total) if total > 0 else 0.0,
            "size_entropy": _entropy(probs),
        }

    def compute_border_touch_fraction(self, label_patch: np.ndarray) -> float:
        fg = (label_patch > 0)
        if not fg.any():
            return 0.0
        zmax, ymax, xmax = np.array(label_patch.shape) - 1
        border = np.zeros_like(fg, dtype=bool)
        border[0, :, :] = True
        border[zmax, :, :] = True
        border[:, 0, :] = True
        border[:, ymax, :] = True
        border[:, :, 0] = True
        border[:, :, xmax] = True
        touch = (fg & border).sum()
        return float(touch) / float(fg.sum())

    def compute_skeleton_and_radii(self, label_patch: np.ndarray, spacing_zyx_mm: Optional[Tuple[float, float, float]]) -> Dict[str, float]:
        fg = (label_patch > 0).astype(np.uint8)
        if fg.sum() == 0:
            return {
                "skel_len_mm": 0.0,
                "radius_mean_mm": 0.0,
                "radius_median_mm": 0.0,
                "used_skan": False,
            }
        skel = _skeletonize_3d_safe(fg)
        n_skel_vox = int(skel.sum())
        if n_skel_vox < 2:
            # length is zero; no need to call Skan
            return {"skel_len_mm": 0.0, "radius_mean_mm": 0.0, "radius_median_mm": 0.0, "used_skan": False}
        if skel.sum() == 0:
            return {
                "skel_len_mm": 0.0,
                "radius_mean_mm": 0.0,
                "radius_median_mm": 0.0,
                "used_skan": False,
            }
        # EDT in physical space if spacing provided
        if spacing_zyx_mm is not None:
            edt = ndimage.distance_transform_edt(fg > 0, sampling=spacing_zyx_mm)
        else:
            edt = ndimage.distance_transform_edt(fg > 0)
        radii = edt[skel > 0]  # inscribed-sphere radius
        radius_mean_mm = float(np.mean(radii)) if radii.size > 0 else 0.0
        radius_median_mm = float(np.median(radii)) if radii.size > 0 else 0.0
        # Skeleton geodesic length (preferred: skan)
        skel_len_mm = 0.0
        used_skan = False
        if self.use_skan and Skeleton is not None:
            try:
                #Here, we are using the skan library to compute the skeleton geodesic length, which is more accurate than the voxel-count approximation.
                #skel_len_mm is the sum of the lengths of all branches in the skeleton.
                spacing_tuple = spacing_zyx_mm if spacing_zyx_mm is not None else (1.0, 1.0, 1.0)
                sk_obj = Skeleton(skel.astype(bool), spacing=spacing_tuple)
                lengths = sk_obj.path_lengths()  # array of branch lengths in physical units
                skel_len_mm = float(np.sum(lengths)) if lengths is not None and len(lengths) > 0 else 0.0
                used_skan = True
            except Exception as e:
                print(f"[patch_audit] skan path-length computation failed ({e}); falling back to voxel-count approximation")
        if not used_skan:
            # Fallback approximation: count skeleton voxels times mean spacing
            if spacing_zyx_mm is not None:
                voxel_step = float(np.mean(np.asarray(spacing_zyx_mm)))
            else:
                voxel_step = 1.0
            skel_len_mm = float(skel.sum()) * voxel_step
        return {
            "skel_len_mm": float(skel_len_mm),
            "radius_mean_mm": float(radius_mean_mm),
            "radius_median_mm": float(radius_median_mm),
            "used_skan": bool(used_skan),
        }

    def compute_topology_metrics(self, label_patch: np.ndarray) -> Dict[str, float]:
        fg = (label_patch > 0).astype(np.uint8)
        if fg.sum() == 0:
            return {
                "endpoint_count": 0.0,
                "bifurcation_count": 0.0,
                "branch_density": 0.0,
                "topology_score": 0.0,
            }
        skel = _skeletonize_3d_safe(fg)
        if skel.sum() == 0:
            return {
                "endpoint_count": 0.0,
                "bifurcation_count": 0.0,
                "branch_density": 0.0,
                "topology_score": 0.0,
            }
        # 26-connectivity kernel and exclude self by subtracting 1
        kernel = np.ones((3, 3, 3), dtype=np.float32)
        nb = ndimage.convolve(skel.astype(np.float32), kernel, mode="constant", cval=0.0)
        nb_on_skel = nb[skel > 0] - 1.0
        nb_on_skel = np.clip(nb_on_skel, 0.0, 26.0)
        endpoints = int((nb_on_skel == 1).sum())
        branches = int((nb_on_skel >= 3).sum())
        total = float(nb_on_skel.size)
        branch_density = (branches / total) if total > 0 else 0.0
        # Simple composite score (bounded)
        topology_score = min(2.0, 2.0 * branch_density + 1.0 * (endpoints / (total + 1e-6)))
        return {
            "endpoint_count": float(endpoints),
            "bifurcation_count": float(branches),
            "branch_density": float(branch_density),
            "topology_score": float(topology_score),
        }

    def compute_image_complexity(self, image_patch: np.ndarray, label_patch: np.ndarray) -> Dict[str, float]:
        fg = (label_patch > 0)
        if not fg.any():
            return {"complexity_score": 0.0}
        bg = ~fg
        fg_vals = image_patch[fg]
        bg_vals = image_patch[bg]
        if fg_vals.size < 10 or bg_vals.size < 10:
            return {"complexity_score": 0.0}
        fg_mean = float(np.mean(fg_vals))
        bg_mean = float(np.mean(bg_vals))
        contrast = abs(fg_mean - bg_mean)
        noise = float((np.std(fg_vals) + np.std(bg_vals)) / 2.0)
        snr = contrast / (noise + 1e-6)
        # Boundary region gradient magnitude
        boundary = ndimage.binary_dilation(fg, iterations=2) & (~fg)
        if boundary.any():
            gx = ndimage.sobel(image_patch, axis=0)
            gy = ndimage.sobel(image_patch, axis=1)
            gz = ndimage.sobel(image_patch, axis=2)
            grad = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
            boundary_grad = float(np.mean(grad[boundary]))
        else:
            boundary_grad = 0.0
        comp = (1.0 / (snr + 1e-1)) * (1.0 / (boundary_grad + 1e-1))
        comp = float(min(comp, 3.0))
        return {
            "complexity_score": comp,
            "contrast": float(contrast),
            "snr": float(snr),
            "boundary_gradient": float(boundary_grad),
        }

    def _compute_sampling_weight(self, fg_fraction: float, radius_class_weight: float, topology_score: float, complexity_score: float) -> float:
        base_weight = 0.1 if fg_fraction < 1e-3 else 1.0
        radius_weight = float(radius_class_weight)
        topology_weight = 1.0 + 0.5 * float(topology_score)
        complexity_weight = 1.0 + 0.3 * float(complexity_score)
        w = base_weight * radius_weight * topology_weight * complexity_weight
        return float(max(0.05, min(w, 5.0)))

    def _radius_class_weight(self, mean_radius_mm: float) -> Tuple[str, float]:
        # Coarse bins (adjust later after global quantiles are known)
        if mean_radius_mm > 8.0:
            return ("main", 0.6)
        if mean_radius_mm > 4.0:
            return ("segmental", 0.8)
        if mean_radius_mm > 2.0:
            return ("subsegmental", 1.0)
        return ("distal", 1.4)

    def audit_case(
        self,
        case_id: str,
        image_path: Path,
        label_path: Path,
        lung_path: Optional[Path],
        spacing_zyx_mm: Optional[Tuple[float, float, float]],
    ) -> List[Dict]:
        image = np.load(str(image_path), mmap_mode="r")
        label = np.load(str(label_path), mmap_mode="r")
        lung = np.load(str(lung_path), mmap_mode="r") if (lung_path and lung_path.exists()) else None
        assert image.shape == label.shape, f"Shape mismatch for {case_id}: {image.shape} vs {label.shape}"
        positions = _generate_patch_positions(image.shape, self.patch_size, self.stride)
        out: List[Dict] = []
        for (z, y, x) in tqdm(positions, desc=f"{case_id}: patches", leave=False):
            img_p = _extract_patch(image, (z, y, x), self.patch_size)
            lbl_p = _extract_patch(label, (z, y, x), self.patch_size)
            if lung is not None:
                lng_p = _extract_patch(lung, (z, y, x), self.patch_size)
                if float((lng_p > 0).sum()) < self.min_lung_coverage * float(lng_p.size):
                    continue
            # Descriptors
            fg_frac_unw = self.compute_fg_fraction_unweighted(lbl_p)
            fg_frac_center = self.compute_fg_fraction_center_weighted(lbl_p)
            comps = self.compute_components_metrics(lbl_p)
            border_touch = self.compute_border_touch_fraction(lbl_p)
            skel = self.compute_skeleton_and_radii(lbl_p, spacing_zyx_mm)
            topo = self.compute_topology_metrics(lbl_p)
            imgc = self.compute_image_complexity(img_p, lbl_p)
            # radius class
            rclass, rweight = self._radius_class_weight(skel.get("radius_mean_mm", 0.0))
            weight = self._compute_sampling_weight(
                fg_fraction=fg_frac_unw,
                radius_class_weight=rweight,
                topology_score=topo.get("topology_score", 0.0),
                complexity_score=imgc.get("complexity_score", 0.0),
            )
            out.append({
                "case_id": case_id,
                "position_zyx": (int(z), int(y), int(x)),
                "fg_fraction": float(fg_frac_unw),
                "fg_fraction_center_weighted": float(fg_frac_center),
                "n_components": float(comps["n_components"]),
                "lcc_fraction": float(comps["lcc_fraction"]),
                "size_entropy": float(comps["size_entropy"]),
                "border_touch_fraction": float(border_touch),
                "skel_len_mm": float(skel["skel_len_mm"]),
                "radius_mean_mm": float(skel["radius_mean_mm"]),
                "radius_median_mm": float(skel["radius_median_mm"]),
                "endpoint_count": float(topo["endpoint_count"]),
                "bifurcation_count": float(topo["bifurcation_count"]),
                "branch_density": float(topo["branch_density"]),
                "topology_score": float(topo["topology_score"]),
                "complexity_score": float(imgc.get("complexity_score", 0.0)),
                "radius_class": rclass,
                "sampling_weight": weight,
                "used_skan": bool(skel.get("used_skan", False)),
            })
        if self.max_patches_per_case is not None and len(out) > self.max_patches_per_case:
            rng = np.random.default_rng(123)
            idx = rng.choice(len(out), size=self.max_patches_per_case, replace=False)
            out = [out[i] for i in idx]
        return out


def _collect_files(npy_root: Path) -> List[Tuple[str, Path, Path, Optional[Path], Path]]:
    images_dir = npy_root / "imagesTr"
    labels_dir = npy_root / "labelsTr"
    lungs_dir = npy_root / "lungsTr"
    items: List[Tuple[str, Path, Path, Optional[Path], Path]] = []
    for img_path in sorted(images_dir.glob("*.npy")):
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.npy"
        if not lbl_path.exists():
            continue
        lung_path = lungs_dir / f"{stem}.npy"
        meta_json = images_dir / f"{stem}_meta.json"
        if not meta_json.exists():
            # try label meta
            alt = labels_dir / f"{stem}_meta.json"
            meta_json = alt if alt.exists() else meta_json
        items.append((stem, img_path, lbl_path, lung_path if lung_path.exists() else None, meta_json))
    return items


def _aggregate_global_stats(all_patch_descs: List[Dict]) -> Dict[str, Dict[str, float]]:
    # Compute quantiles for selected metrics to help bin design
    metrics = [
        "fg_fraction",
        "n_components",
        "lcc_fraction",
        "size_entropy",
        "border_touch_fraction",
        "skel_len_mm",
        "radius_mean_mm",
        "radius_median_mm",
        "endpoint_count",
        "bifurcation_count",
        "branch_density",
        "topology_score",
        "complexity_score",
    ]
    stats: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        vals = np.array([float(p.get(m, 0.0)) for p in all_patch_descs], dtype=np.float64)
        if vals.size == 0:
            stats[m] = {}
            continue
        q = np.quantile(vals, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        stats[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "q01": float(q[0]),
            "q05": float(q[1]),
            "q25": float(q[2]),
            "q50": float(q[3]),
            "q75": float(q[4]),
            "q95": float(q[5]),
            "q99": float(q[6]),
        }
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline patch audit generator (Phase 1)")
    parser.add_argument("--data_root", type=str, default="./ATM22/npy_files", help="Path to npy_files root (with imagesTr/labelsTr/lungsTr)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to write patch banks (default: <data_root>/patch_banks)")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96], help="Patch size in voxels (z y x)")
    parser.add_argument("--stride", type=int, nargs=3, default=[32, 32, 32], help="Stride in voxels (z y x)")
    parser.add_argument("--min_lung_coverage", type=float, default=0.10, help="Min fraction of lung voxels required inside patch")
    parser.add_argument("--max_patches_per_case", type=int, default=None, help="Optional cap per case for memory/time control")
    parser.add_argument("--limit_cases", type=int, default=None, help="Optionally limit number of cases for a dry run")
    parser.add_argument("--use_skan", action="store_true", help="Use skan to compute skeleton geodesic length (slower, more accurate)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir) if args.output_dir else (data_root / "patch_banks")
    out_dir.mkdir(parents=True, exist_ok=True)

    items = _collect_files(data_root)
    if args.limit_cases is not None:
        items = items[: int(args.limit_cases)]
    if not items:
        print(f"No cases found under {data_root}")
        return

    auditor = PatchAuditSystem(
        patch_size=tuple(int(x) for x in args.patch_size),
        stride=tuple(int(x) for x in args.stride),
        min_lung_coverage=float(args.min_lung_coverage),
        max_patches_per_case=args.max_patches_per_case,
        use_skan=bool(args.use_skan),
    )

    bank_mgr = PatchBankManager()
    # attach context so PatchBankManager can embed in metadata
    setattr(bank_mgr, "_patch_size", tuple(int(x) for x in args.patch_size))
    setattr(bank_mgr, "_stride", tuple(int(x) for x in args.stride))
    global_descs: List[Dict] = []

    for (stem, img_p, lbl_p, lung_p, meta_json) in tqdm(items, desc="Cases"):
        spacing_zyx = _read_spacing_zyx_from_meta(meta_json) if meta_json and meta_json.exists() else None
        patch_descs = auditor.audit_case(
            case_id=stem,
            image_path=img_p,
            label_path=lbl_p,
            lung_path=lung_p,
            spacing_zyx_mm=spacing_zyx,
        )
        bank_path = out_dir / f"{stem}.pkl.gz"
        bank_mgr.save_patch_bank(patch_descs, bank_path)
        global_descs.extend(patch_descs)

    # Save global stats / histograms
    stats = _aggregate_global_stats(global_descs)
    with open(out_dir / "global_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved global stats to {out_dir / 'global_stats.json'}")
    print(f"Completed patch audit for {len(items)} cases. Patch banks in: {out_dir}")


if __name__ == "__main__":
    main() 