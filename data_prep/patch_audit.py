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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
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
    # Extract a patch from a 3D volume given start coordinates and patch size
    z0, y0, x0 = start_zyx
    dz, dy, dx = patch_size
    return volume_zyx[z0:z0+dz, y0:y0+dy, x0:x0+dx]


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    # Compute entropy of a probability distribution
    p = probs[(probs > 0) & np.isfinite(probs)]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p + eps)).sum())


class PatchBankManager:
    def __init__(self) -> None:
        pass
    def save_patch_bank(self, patch_bank: List[Dict], output_path: Path) -> None:
        # infer patch_size/stride from any descriptor, else leave None
        inferred_ps = None
        inferred_stride = None
        # Look for optional hints in descriptors
        for p in patch_bank:
            if isinstance(p.get("position_zyx"), (list, tuple)):
                # position alone doesn't encode size; use auditor's defaults if embedded later
                pass
        # Metadata
        meta = {
            "total_patches": len(patch_bank),
            "fg_patches": int(sum(1 for p in patch_bank if p.get("fg_fraction", 0.0) > 0.0)),
            "patch_size": getattr(self, "_patch_size", None),
            "stride": getattr(self, "_stride", None),
        }
        # Save as compressed pickle
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
    # Initialize the PatchAuditSystem with parameters
        self.patch_size = patch_size
        self.stride = stride
        self.min_lung_coverage = float(min_lung_coverage)
        self.max_patches_per_case = max_patches_per_case
        self.use_skan = bool(use_skan)    
        self.RAD_Q50, self.RAD_Q95, self.RAD_Q99 = 0.725, 2.791, 4.808  # distal / subseg / seg / main

    # ---- Descriptors ----
    def compute_fg_fraction_unweighted(self, label_patch: np.ndarray) -> float:
        # This function computes the unweighted foreground fraction of the patch, i.e., the ratio of foreground voxels to total voxels.
        total = float(label_patch.size)
        if total <= 0:
            return 0.0
        return float((label_patch > 0).sum()) / total

    def compute_fg_fraction_center_weighted(self, label_patch: np.ndarray) -> float:
        #this function computes the fg fraction of the patch weighted by the distance from the center of the patch, giving more importance to voxels closer to the center.
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
        # Connected components metrics: count, LCC fraction, size entropy, using 26-connectivity for 3D. 
        fg = (label_patch > 0).astype(np.uint8)
        if fg.sum() == 0:
            return {
                "n_components": 0.0,
                "lcc_fraction": 0.0,
                "size_entropy": 0.0,
            }
        # Using 26-connectivity for 3D airways
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
        # Fraction of foreground voxels touching the patch border
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
        # Skeletonization and inscribed-sphere radii metrics, with optional spacing for physical units (mm)
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
        # Topology metrics from skeleton: endpoint count, bifurcation count, branch density, composite topology score. 
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
        # Count neighbors on skeleton excluding self
        nb_on_skel = nb[skel > 0] - 1.0
        # Clamp to [0, 26] to avoid spurious negatives
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

    def compute_landmark_masks(
        self,
        label_volume: np.ndarray,
        lung_volume: Optional[np.ndarray],
        spacing_zyx_mm: Optional[Tuple[float, float, float]],
    ) -> Dict[str, np.ndarray]:
        """
        Compute coarse anatomical landmark masks on the full volume with memory-safe ops:
        - bifurcation_mask: skeleton voxels with branching degree >= 3 (1-voxel dilation)
        - trachea_mask: large-radius skeleton voxels (>= RAD_Q95) in superior 20% lung extent (2-voxel dilation)

        Implementation notes:
        - Restrict computations to lung/label bounding box to reduce memory.
        - Compute distance transform only in superior region to identify trachea/main bronchi.
        """
        shp = tuple(label_volume.shape)
        out = {
            "bifurcation_mask": np.zeros(shp, dtype=bool),
            "trachea_mask": np.zeros(shp, dtype=bool),
        }
        # Define ROI bbox from lung if available, else from label
        if lung_volume is not None and lung_volume.shape == label_volume.shape and (lung_volume > 0).any():
            nz = np.where(lung_volume > 0)
        else:
            nz = np.where(label_volume > 0)
        if len(nz[0]) == 0:
            return out
        zmin, zmax = int(np.min(nz[0])), int(np.max(nz[0]))
        ymin, ymax = int(np.min(nz[1])), int(np.max(nz[1]))
        xmin, xmax = int(np.min(nz[2])), int(np.max(nz[2]))
        # Pad bbox by 1 voxel within bounds
        z0, z1 = max(0, zmin - 1), min(shp[0] - 1, zmax + 1)
        y0, y1 = max(0, ymin - 1), min(shp[1] - 1, ymax + 1)
        x0, x1 = max(0, xmin - 1), min(shp[2] - 1, xmax + 1)

        # Crop volumes to bbox
        fg_crop = (label_volume[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] > 0).astype(np.uint8)
        if fg_crop.sum() == 0:
            return out

        # Skeletonize cropped FG
        try:
            skel_crop = _skeletonize_3d_safe(fg_crop)
        except Exception:
            return out
        if skel_crop.sum() == 0:
            # No skeleton -> no landmarks
            return out

        # Bifurcation mask within crop
        kernel = np.ones((3, 3, 3), dtype=np.float32)
        nb = ndimage.convolve(skel_crop.astype(np.float32), kernel, mode="constant", cval=0.0)
        nb_minus_self = np.clip(nb - 1.0, 0.0, 26.0)
        bif_crop = (skel_crop > 0) & (nb_minus_self >= 3.0)
        bif_crop = ndimage.binary_dilation(bif_crop, iterations=1)

        # Scatter to full-size mask
        out["bifurcation_mask"][z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = bif_crop

        # Trachea/main bronchi proxy within superior portion of bbox
        # Determine superior extent from lung if available
        if lung_volume is not None and lung_volume.shape == label_volume.shape and (lung_volume > 0).any():
            z_inds = np.where(lung_volume.max(axis=(1, 2)) > 0)[0]
            if z_inds.size > 0:
                zmin_l, zmax_l = int(z_inds.min()), int(z_inds.max())
            else:
                zmin_l, zmax_l = z0, z1
        else:
            zmin_l, zmax_l = z0, z1
        z_extent = max(1, zmax_l - zmin_l + 1)
        # Expand superior region to capture main bronchi; was 20%, now 35%
        top_frac = 0.35
        top_end = zmin_l + max(1, int(top_frac * z_extent))

        # Restrict EDT to top region of the crop to save memory
        top0 = z0
        top1 = min(z1, top_end)
        top_slice = slice(max(z0, zmin_l), top1 + 1)
        fg_top = (label_volume[top_slice, y0:y1 + 1, x0:x1 + 1] > 0).astype(np.uint8)
        try:
            if spacing_zyx_mm is not None:
                edt_top = ndimage.distance_transform_edt(fg_top > 0, sampling=spacing_zyx_mm)
            else:
                edt_top = ndimage.distance_transform_edt(fg_top > 0)
        except Exception:
            edt_top = np.zeros_like(fg_top, dtype=float)
        # Large-radius threshold: be less strict than RAD_Q95; fallback if none found
        thr_primary = max(self.RAD_Q50, 0.8 * self.RAD_Q95)
        rel_start = max(0, top_slice.start - z0)
        rel_stop = max(rel_start, top_slice.stop - z0)
        skel_top = skel_crop[rel_start:rel_stop, :, :]
        trach_top = (skel_top > 0) & (edt_top >= thr_primary)
        if not trach_top.any():
            thr_fallback = max(self.RAD_Q50, 0.6 * self.RAD_Q95)
            trach_top = (skel_top > 0) & (edt_top >= thr_fallback)
        trach_top = ndimage.binary_dilation(trach_top, iterations=2)
        out["trachea_mask"][top_slice, y0:y1 + 1, x0:x1 + 1] = trach_top

        return out

    def compute_image_complexity(self, image_patch: np.ndarray, label_patch: np.ndarray) -> Dict[str, float]:
        # Image complexity metrics: contrast, SNR, boundary gradient, composite complexity score as described within the report
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
        # Composite sampling weight based on descriptors, bounded to [0.05, 5.0], with heuristic formula
        dz, dy, dx = self.patch_size
        # Avoid overweighting very small fg fractions
        fg_eps = 100.0 / float(dz * dy * dx)
        # Base weight: downweight very small fg fractions
        base_weight = 0.1 if fg_fraction < fg_eps else 1.0
        radius_weight = float(radius_class_weight)
        # Emphasize topology and complexity scores (0.0-2.0 and 0.0-3.0 respectively)
        topology_weight = 1.0 + 0.5 * float(topology_score)
        complexity_weight = 1.0 + 0.5 * float(complexity_score)
        w = base_weight * radius_weight * topology_weight * complexity_weight
        return float(max(0.05, min(w, 5.0)))
    
    def _radius_class_weight(self, mean_radius_mm: float) -> Tuple[str, float]:
        RAD_Q50, RAD_Q95, RAD_Q99 = 0.725, 2.791, 4.808  # mm
        if mean_radius_mm > RAD_Q99:
            return ("main", 0.6)
        if mean_radius_mm > RAD_Q95:
            return ("segmental", 0.8)
        if mean_radius_mm > RAD_Q50:
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
        # Precompute landmark masks once per case
        lm = self.compute_landmark_masks(label, lung, spacing_zyx_mm)
        lm_bif = lm.get("bifurcation_mask")
        lm_trach = lm.get("trachea_mask")
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
            # Landmark flags for this patch
            dz, dy, dx = self.patch_size
            has_bif = bool(lm_bif[z:z+dz, y:y+dy, x:x+dx].any()) if isinstance(lm_bif, np.ndarray) else False
            has_trach = bool(lm_trach[z:z+dz, y:y+dy, x:x+dx].any()) if isinstance(lm_trach, np.ndarray) else False
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
                "is_bifurcation": bool(has_bif),
                "is_trachea": bool(has_trach),
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


def audit_worker(
    stem: str,
    img_path_str: str,
    lbl_path_str: str,
    lung_path_str: Optional[str],
    meta_json_str: Optional[str],
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    min_lung_coverage: float,
    max_patches_per_case: Optional[int],
    use_skan: bool,
    out_dir_str: str,
) -> Tuple[str, int]:
    """Top-level worker for parallel execution"""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    img_p = Path(img_path_str)
    lbl_p = Path(lbl_path_str)
    lung_p = Path(lung_path_str) if lung_path_str else None
    meta_json = Path(meta_json_str) if meta_json_str else None
    out_dir = Path(out_dir_str)
    spacing_zyx = _read_spacing_zyx_from_meta(meta_json) if meta_json and meta_json.exists() else None
    auditor_local = PatchAuditSystem(
        patch_size=tuple(int(x) for x in patch_size),
        stride=tuple(int(x) for x in stride),
        min_lung_coverage=float(min_lung_coverage),
        max_patches_per_case=max_patches_per_case,
        use_skan=bool(use_skan),
    )
    patch_descs = auditor_local.audit_case(
        case_id=stem,
        image_path=img_p,
        label_path=lbl_p,
        lung_path=lung_p,
        spacing_zyx_mm=spacing_zyx,
    )
    bank_mgr = PatchBankManager()
    setattr(bank_mgr, "_patch_size", tuple(int(x) for x in patch_size))
    setattr(bank_mgr, "_stride", tuple(int(x) for x in stride))
    bank_mgr.save_patch_bank(patch_descs, out_dir / f"{stem}.pkl.gz")
    return (stem, len(patch_descs))


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
    parser.add_argument("--num_workers", type=int, default=0, help="Parallelize across cases with this many workers (0 = no parallelism)")
    parser.add_argument("--skip_global_stats", action="store_true", help="Skip writing global_stats.json to save time")
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
    processed_counts: List[int] = []
    if int(args.num_workers) and args.num_workers > 0:
        # 'spawn' to avoid fork-related crashes with NumPy/SciPy/ITK, and limit intra-op threads
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(args.num_workers),
            mp_context=ctx,
        ) as ex:
            futures = [
                ex.submit(
                    audit_worker,
                    stem,
                    str(img_p),
                    str(lbl_p),
                    str(lung_p) if lung_p else None,
                    str(meta_json) if meta_json else None,
                    tuple(int(x) for x in args.patch_size),
                    tuple(int(x) for x in args.stride),
                    float(args.min_lung_coverage),
                    args.max_patches_per_case,
                    bool(args.use_skan),
                    str(out_dir),
                )
                for (stem, img_p, lbl_p, lung_p, meta_json) in items
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Cases (parallel)"):
                try:
                    _, n = fut.result()
                    processed_counts.append(int(n))
                except Exception as e:
                    print(f"[patch_audit] ERROR processing case: {e}")
    else:
        # Fallback: sequential processing
        for (stem, img_p, lbl_p, lung_p, meta_json) in tqdm(items, desc="Cases"):
            _, n = audit_worker(
                stem,
                str(img_p),
                str(lbl_p),
                str(lung_p) if lung_p else None,
                str(meta_json) if meta_json else None,
                tuple(int(x) for x in args.patch_size),
                tuple(int(x) for x in args.stride),
                float(args.min_lung_coverage),
                args.max_patches_per_case,
                bool(args.use_skan),
                str(out_dir),
            )
            processed_counts.append(int(n))

    if not bool(args.skip_global_stats):
        all_descs: List[Dict] = []
        for bank_file in tqdm(sorted(out_dir.glob("*.pkl.gz")), desc="Aggregating stats from banks"):
            try:
                with gzip.open(str(bank_file), "rb") as f:
                    obj = pickle.load(f)
                patch_list = obj.get("patch_descriptors", obj) if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
                all_descs.extend(patch_list)
            except Exception as e:
                print(f"WARN: failed to read {bank_file.name} for stats: {e}")
        stats = _aggregate_global_stats(all_descs)
        with open(out_dir / "global_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved global stats to {out_dir / 'global_stats.json'}")

    print(f"Completed patch audit for {len(items)} cases. Wrote {len(list(out_dir.glob('*.pkl.gz')))} banks to: {out_dir}")


if __name__ == "__main__":
    main() 
