from __future__ import annotations

import gzip
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from monai.transforms.transform import MapTransform
from monai.transforms.croppad.dictionary import SpatialCropd
from monai.transforms.croppad.dictionary import CenterSpatialCropd

class SamplePatchFromBankd(MapTransform):
    """
    Sample patches deterministically from precomputed patch banks using static, audit-derived
    weights. At call-time, weights are normalized into sampling probabilities with an optional
    temperature to control sharpness.
    """

    def __init__(
        self,
        keys: Union[str, List[str]],
        patch_banks_dir: Union[str, Path],
        spatial_size: Tuple[int, int, int],
        num_samples: int = 1,
        pos_ratio: float = 0.9,
        temperature: float = 1.0,
        fg_threshold: float = 1e-4,
        # Optional landmark-aware sampling
        use_landmarks: bool = False,
        quota_bifurcation: float = 0.0,
        quota_trachea: float = 0.0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.patch_banks_dir = Path(patch_banks_dir)
        self.spatial_size = tuple(int(v) for v in spatial_size)
        self.num_samples = int(num_samples)
        self.pos_ratio = float(pos_ratio)
        self.temperature = max(1e-6, float(temperature))
        self.fg_threshold = float(fg_threshold)
        self.use_landmarks = bool(use_landmarks)
        self.quota_bifurcation = max(0.0, min(1.0, float(quota_bifurcation)))
        self.quota_trachea = max(0.0, min(1.0, float(quota_trachea)))

        self._banks: Dict[str, Dict[str, List[Dict]]] = {}
        self._bank_meta: Dict[str, Dict] = {}
        self._loaded = False
        self._warned_patchsize_missing = False

    # ---- helpers ----
    def _extract_case_id(self, data: Dict) -> str:
        meta = data.get("image_meta_dict")
        path_str = None
        if isinstance(meta, dict):
            v = meta.get("filename_or_obj")
            if isinstance(v, (list, tuple)) and v:
                path_str = str(v[0])
            elif isinstance(v, (str, Path)):
                path_str = str(v)
        if path_str is None and "image" in data:
            #try tensor meta if available
            img = data["image"]
            try:
                v = getattr(img, "meta", {}).get("filename_or_obj")
                if isinstance(v, (list, tuple)) and v:
                    path_str = str(v[0])
                elif isinstance(v, (str, Path)):
                    path_str = str(v)
            except Exception:
                pass
        if not path_str:
            return "unknown"
        return Path(path_str).stem

    def _load_bank_for_case(self, case_id: str) -> Dict[str, List[Dict]]:
        if case_id in self._banks:
            return self._banks[case_id]
        bank_path = self.patch_banks_dir / f"{case_id}.pkl.gz"
        if not bank_path.exists():
            self._banks[case_id] = {"positive": [], "background": []}
            self._bank_meta[case_id] = {}
            return self._banks[case_id]
        with gzip.open(str(bank_path), "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "patch_descriptors" in obj:
            patch_list = obj.get("patch_descriptors", [])
            meta = obj.get("metadata", {})
        else:
            patch_list = obj
            meta = {}
        self._bank_meta[case_id] = meta
        expected_ps = tuple(meta.get("patch_size") or ())
        if expected_ps and tuple(int(v) for v in expected_ps) != self.spatial_size:
            print(
                f"[SamplePatchFromBankd] Warning: spatial_size {self.spatial_size} != bank patch_size {expected_ps} for case {case_id}. Proceeding with current spatial_size."
            )
        if not expected_ps and not self._warned_patchsize_missing:
            print("[SamplePatchFromBankd] Warning: patch bank metadata lacks 'patch_size'; cannot verify size match.")
            self._warned_patchsize_missing = True
        pos = [p for p in patch_list if float(p.get("fg_fraction", 0.0)) > self.fg_threshold]
        bg = [p for p in patch_list if float(p.get("fg_fraction", 0.0)) <= self.fg_threshold]
        # Optional landmark pools (may overlap with pos/bg)
        if self.use_landmarks:
            bif = [p for p in patch_list if bool(p.get("is_bifurcation", False))]
            tra = [p for p in patch_list if bool(p.get("is_trachea", False))]
        else:
            bif, tra = [], []
        self._banks[case_id] = {"positive": pos, "background": bg, "bifurcation": bif, "trachea": tra}
        return self._banks[case_id]

    def _softmax_choice(self, patches: List[Dict]) -> Dict:
        if not patches:
            raise RuntimeError("No patches provided to _softmax_choice")
        # here, sampling_weight is the weight of the patch, where the higher the weight, the more likely it is to be sampled
        w = np.array([float(p.get("sampling_weight", 1.0)) for p in patches], dtype=np.float64)
        # temperature scaling in probability space: p_i ∝ w_i^(1/temperature)
        w = np.clip(w, 1e-8, 1e8)
        logits = np.log(w) / self.temperature
        logits -= np.max(logits)
        probs = np.exp(logits)
        s = probs.sum()
        if not np.isfinite(s) or s <= 0:
            idx = random.randrange(len(patches))
            return patches[idx]
        probs /= s
        idx = int(np.random.choice(len(patches), p=probs))
        return patches[idx]

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        case_id = self._extract_case_id(d)
        bank = self._load_bank_for_case(case_id)
        candidates_pos = bank.get("positive", [])
        candidates_bg = bank.get("background", [])
        # Fallback: if bank missing or empty, center-crop to requested spatial_size
        if not candidates_pos and not candidates_bg:
            print(f"[SamplePatchFromBankd] No patches found for case {case_id}; applying CenterSpatialCropd with roi_size={self.spatial_size}")
            crop = CenterSpatialCropd(keys=self.keys, roi_size=self.spatial_size, allow_missing_keys=True)
            return crop(d)

        selected: List[Dict] = []
        for _ in range(max(1, self.num_samples)):
            # Optional landmark quotas: attempt first if enabled
            chosen = None
            if self.use_landmarks and (self.quota_bifurcation > 0 or self.quota_trachea > 0):
                if (random.random() < self.quota_bifurcation) and bank.get("bifurcation"):
                    chosen = self._softmax_choice(bank["bifurcation"])  # type: ignore[arg-type]
                elif (random.random() < self.quota_trachea) and bank.get("trachea"):
                    chosen = self._softmax_choice(bank["trachea"])  # type: ignore[arg-type]
            if chosen is None:
                draw_pos = (random.random() < self.pos_ratio) and bool(candidates_pos)
                pool = candidates_pos if draw_pos else (candidates_bg if candidates_bg else candidates_pos)
                chosen = self._softmax_choice(pool)
             # Lightweight debug: infer the source without relying on draw_pos existing
            if random.random() < 0.05:  # sample ~5% for logging
                if chosen in candidates_pos:
                    src = "pos"
                elif chosen in candidates_bg:
                    src = "bg"
                else:
                    src = "lm"  # landmark (bifurcation/trachea)
                try:
                    fg = float(chosen.get("fg_fraction", -1.0))
                except Exception:
                    fg = -1.0
                print(f"[PB] case={case_id} src={src} fg={fg:.2e}")
            selected.append(chosen)
        # Apply SpatialCropd so meta is updated consistently
        for i, patch_desc in enumerate(selected):
            z, y, x = tuple(int(v) for v in patch_desc.get("position_zyx", patch_desc.get("position", (0, 0, 0))))
            dz, dy, dx = self.spatial_size
            roi_start = (z, y, x)
            roi_end = (z + dz, y + dy, x + dx)
            cropper = SpatialCropd(keys=self.keys, roi_start=roi_start, roi_end=roi_end, allow_missing_keys=True)
            cropped = cropper(d)
            if i == 0:
                d = cropped
            else:
                for key in self.key_iterator(cropped):
                    d[f"{key}_{i}"] = cropped[key]
        return d 
