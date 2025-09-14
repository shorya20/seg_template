import os
import sys
import gzip
import json
import pickle
from pathlib import Path
import numpy as np


def load_bank(bank_path: Path):
    if not bank_path.exists():
        return None, None
    with gzip.open(str(bank_path), "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "patch_descriptors" in obj:
        patch_list = obj.get("patch_descriptors", [])
        meta = obj.get("metadata", {})
    else:
        patch_list = obj
        meta = {}
    return patch_list, meta


def summarize_bank(patches, meta):
    n = len(patches) if isinstance(patches, (list, tuple)) else 0
    n_pos = 0
    n_bg = 0
    has_w = 0
    n_bif = 0
    n_trach = 0
    weights = []
    for p in (patches or []):
        fg = float(p.get("fg_fraction", 0.0))
        if fg > 0.0:
            n_pos += 1
        else:
            n_bg += 1
        if "sampling_weight" in p:
            has_w += 1
            try:
                weights.append(float(p.get("sampling_weight", 1.0)))
            except Exception:
                pass
        if bool(p.get("is_bifurcation", False)):
            n_bif += 1
        if bool(p.get("is_trachea", False)):
            n_trach += 1
    print(f"- Bank descriptors: total={n}, pos={n_pos}, bg={n_bg}")
    print(f"- Bank metadata: {meta if isinstance(meta, dict) else {}}")
    if isinstance(meta, dict) and "patch_size" in meta:
        print(f"- Bank patch_size: {tuple(meta.get('patch_size'))}")
    else:
        print("- Bank patch_size missing in metadata")
    if has_w:
        import numpy as np
        w = np.array(weights, dtype=float) if weights else np.array([1.0])
        print(f"- Weights: min={w.min():.3f} max={w.max():.3f} mean={w.mean():.3f}")
    if n_bif or n_trach:
        print(f"- Landmark tags: is_bifurcation={n_bif}, is_trachea={n_trach}")


def main(case_id: str = "ATM_045_0000"):
    root = Path("/vol/bitbucket/ss2424/project/seg_template/ATM22/npy_files")
    bank_path = root / "patch_banks" / f"{case_id}.pkl.gz"
    lungs_path = root / "lungsTr" / f"{case_id}.npy"
    image_meta = root / "imagesTr" / f"{case_id}_meta.json"

    print(f"Checking case: {case_id}")
    print(f"Bank path: {bank_path}")
    print(f"Lungs path: {lungs_path}")

    # Bank
    patches, meta = load_bank(bank_path)
    if patches is None:
        print("- Bank file not found.")
    else:
        summarize_bank(patches, meta)

    # Lungs
    if lungs_path.exists():
        try:
            lungs = np.load(str(lungs_path), mmap_mode="r")
            nz = int((lungs > 0).sum())
            print("lungs unique values:", np.unique(lungs), nz / lungs.size)
            shp = tuple(lungs.shape)
            print(f"- Lungs shape: {shp}, foreground voxels: {nz}")
            if nz == 0:
                print("  WARNING: lungs mask has zero foreground voxels.")
        except Exception as e:
            print(f"- Failed to load lungs npy: {e}")
    else:
        print("- Lungs numpy file not found.")

    # Meta sanity
    if image_meta.exists():
        try:
            meta_json = json.loads(Path(image_meta).read_text())
            proc_spacing = meta_json.get("processed_spacing")
            npy_shape = meta_json.get("numpy_array_shape_zyx")
            print(f"- Image meta processed_spacing: {proc_spacing}, npy_shape: {npy_shape}")
        except Exception as e:
            print(f"- Failed to read image meta: {e}")


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "ATM_617_0000"
    main(case)

