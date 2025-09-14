import os, sys, gzip, json, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

ROOT = Path("/vol/bitbucket/ss2424/project/seg_template/ATM22/npy_files")

def load_bank(case_id):
    bank_path = ROOT / "patch_banks" / f"{case_id}.pkl.gz"
    with gzip.open(str(bank_path), "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "patch_descriptors" in obj:
        patches = obj["patch_descriptors"]; meta = obj.get("metadata", {})
    else:
        patches, meta = obj, {}
    return patches, meta

def load_shapes(case_id):
    # shapes help allocate heatmap
    meta_json = ROOT / "imagesTr" / f"{case_id}_meta.json"
    if meta_json.exists():
        m = json.loads(meta_json.read_text())
        shp = tuple(int(v) for v in m.get("numpy_array_shape_zyx", []))
    else:
        img = np.load(str(ROOT / "imagesTr" / f"{case_id}.npy"), mmap_mode="r")
        shp = img.shape
    return shp

def make_footprint_heatmap(patches, vol_shape, ps=(128,128,128)):
    Z,Y,X = vol_shape; dz,dy,dx = ps
    heat = np.zeros((Z,Y,X), np.float32)
    for p in patches:
        z,y,x = map(int, p["position_zyx"])
        w = float(p.get("sampling_weight", 1.0))
        heat[z:z+dz, y:y+dy, x:x+dx] += w
    return heat


def mip_xyz(vol):
    # simple max intensity projections along each axis
    mip_z = vol.max(axis=0)
    mip_y = vol.max(axis=1)
    mip_x = vol.max(axis=2)
    return mip_z, mip_y, mip_x

def main(case_id="ATM_615_0000", topk=8):
    patches, meta = load_bank(case_id)
    print(f"{case_id} :: total={len(patches)}, meta={meta}")

    # basic stats
    w = np.array([float(p.get("sampling_weight", 1.0)) for p in patches], dtype=np.float32)
    fg = np.array([float(p.get("fg_fraction", 0.0)) for p in patches], dtype=np.float32)
    rc = [str(p.get("radius_class", "unknown")) for p in patches]
    lm_bif = sum(1 for p in patches if bool(p.get("is_bifurcation", False)))
    lm_tr = sum(1 for p in patches if bool(p.get("is_trachea", False)))
    print(f"weights: min={w.min():.3f} mean={w.mean():.3f} max={w.max():.3f}")
    print(f"fg>0: {(fg>0).sum()} / {len(fg)} | landmarks: bif={lm_bif}, trach={lm_tr}")
    print("radius class counts:", Counter(rc))

    # plots
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2,3,1); ax2 = fig.add_subplot(2,3,2); ax3 = fig.add_subplot(2,3,3)
    ax4 = fig.add_subplot(2,3,4); ax5 = fig.add_subplot(2,3,5); ax6 = fig.add_subplot(2,3,6)

    # (1) weight histogram (log x)
    ax1.hist(w, bins=40); ax1.set_xscale("log"); ax1.set_title("sampling_weight (log-x)")

    # (2) fg_fraction histogram (log x + eps)
    ax2.hist(fg + 1e-6, bins=40); ax2.set_xscale("log"); ax2.set_title("fg_fraction (log-x)")

    # (3) radius-class bar
    keys, vals = zip(*sorted(Counter(rc).items()))
    ax3.bar(keys, vals); ax3.set_title("radius class counts"); ax3.tick_params(axis='x', rotation=30)

    # (4-6) MIP heatmaps of sampling centers
    vol_shape = load_shapes(case_id)
    heat = make_footprint_heatmap(patches, vol_shape)
    for a, m, title in zip((ax4,ax5,ax6), mip_xyz(heat), ("MIP-Z (axial)", "MIP-Y (coronal)", "MIP-X (sagittal)")):
        im = a.imshow(m, interpolation="nearest"); a.set_title(title); a.axis("off")
    fig.tight_layout()
    out = ROOT / "patch_banks" / f"{case_id}_bank_viz.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

    # list top-K patches
    idx = np.argsort(-w)[:topk]
    print("Top-K patches by weight:")
    for i in idx:
        p = patches[i]
        print(f"  w={w[i]:.3f} fg={fg[i]:.2e} pos={p.get('position_zyx', p.get('position'))} radius={p.get('radius_mean_mm', None)} topo={p.get('topology_score', None)}")

if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "ATM_615_0000"
    main(case)
