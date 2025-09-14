import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
# Matplotlib style for publication-ready figures
plt.rcParams.update({
	"figure.dpi": 140,
	"savefig.dpi": 300,
	"font.size": 11,
	"axes.titlesize": 12,
	"axes.labelsize": 11,
	"legend.fontsize": 10,
	"xtick.labelsize": 10,
	"ytick.labelsize": 10,
	"axes.grid": True,
	"grid.alpha": 0.25,
})

# New imports for patch-bank loading
import gzip
import pickle
from collections import defaultdict
import math

QUANTILE_KEYS = ["q01", "q05", "q25", "q50", "q75", "q95", "q99"]

# Heuristics to choose axis scale
LOG_CANDIDATES = {
	"fg_fraction",
	"skel_len_mm",
	"radius_mean_mm",
	"radius_median_mm",
	"complexity_score",
}

SYML0_CANDIDATES = set()  # reserved for metrics that can cross zero

# Locked Phase-1 radius thresholds (in mm): distal <= t0, subsegmental <= t1, segmental <= t2, else main
LOCKED_RADIUS_THRESHOLDS = (0.725, 2.791, 4.808)
LOCKED_RADIUS_LABELS = ("distal", "subsegmental", "segmental", "main")

EPS = 1e-8  # epsilon to stabilize log plots for near-zero values


def _load_stats(stats_path: Path) -> Dict[str, Dict[str, float]]:
	with open(stats_path, "r") as f:
		stats = json.load(f)
	return stats

# ---- New: load raw patch descriptors from patch banks ----

def _load_patch_banks(
	banks_dir: Path,
	metrics: List[str],
	sample_limit: int = 250000,
	per_case_limit: int = 0,
) -> (Dict[str, List[float]], Dict[str, Dict[str, List[float]]], List[Dict[str, Any]]):
	"""Return overall values per metric, grouped-by-stored-radius, and patch rows.
	Each patch row at least contains: radius_mean_mm, fg_fraction and requested metrics.
	Sampling: cap total patches at sample_limit; optionally cap per-case.
	"""
	metrics_data: Dict[str, List[float]] = {m: [] for m in metrics}
	grouped_data: Dict[str, Dict[str, List[float]]] = {m: defaultdict(list) for m in metrics}
	patch_rows: List[Dict[str, Any]] = []
	loaded = 0
	pkls = sorted(banks_dir.glob("*.pkl.gz"))
	for pf in pkls:
		try:
			with gzip.open(str(pf), "rb") as f:
				obj = pickle.load(f)
			if isinstance(obj, dict) and "patch_descriptors" in obj:
				patches = obj["patch_descriptors"]
			else:
				patches = obj if isinstance(obj, list) else []
		except Exception as e:
			print(f"WARN: failed to load {pf.name}: {e}")
			continue
		if per_case_limit and len(patches) > per_case_limit:
			idx = np.random.choice(len(patches), size=per_case_limit, replace=False)
			patches = [patches[i] for i in idx]
		for p in patches:
			if sample_limit and loaded >= sample_limit:
				break
			# Gather common fields
			radius_mean = float(p.get("radius_mean_mm", np.nan))
			fg = float(p.get("fg_fraction", np.nan))
			stored_rc = str(p.get("radius_class", "unknown"))
			row: Dict[str, Any] = {
				"radius_mean_mm": radius_mean,
				"fg_fraction": fg,
				"radius_class_stored": stored_rc,
			}
			for m in metrics:
				if m in p:
					v = float(p.get(m, np.nan))
					row[m] = v
					if np.isfinite(v):
						metrics_data[m].append(v)
						grouped_data[m][stored_rc].append(v)
			patch_rows.append(row)
			loaded += 1
		if sample_limit and loaded >= sample_limit:
			break
	print(f"Loaded values for {len(metrics)} metrics from {len(pkls)} banks (patches used: {loaded})")
	return metrics_data, grouped_data, patch_rows


def _metric_list(stats: Dict[str, Dict[str, float]]) -> List[str]:
	return list(stats.keys())


def _choose_scale(metric: str, stats_m: Dict[str, float]) -> str:
	# If metric is known to be log-friendly and strictly positive, prefer log
	if metric in LOG_CANDIDATES:
		mn = float(stats_m.get("min", 0.0))
		mx = float(stats_m.get("max", 0.0))
		if mn > 0 and mx / max(mn, 1e-12) > 50:
			return "log"
	# Symmetric log if values cross 0 widely (not expected here)
	if metric in SYML0_CANDIDATES:
		return "symlog"
	return "linear"

# Convenience for banks mode (no precomputed min/max)
_DEF_LOG_METRICS = LOG_CANDIDATES


def _quantile_array(stats_m: Dict[str, float]) -> np.ndarray:
	return np.array([stats_m[k] for k in QUANTILE_KEYS], dtype=float)


def _box_from_quantiles(ax, quantiles: np.ndarray, title: str, xlabel: str, scale: str, min_val: float = None, max_val: float = None) -> None:
	# Standard box-and-whisker plot from quantiles
	q01, q05, q25, q50, q75, q95, q99 = quantiles
	
	# Debug: print values to verify they match JSON
	print(f"DEBUG {title}: q01={q01:.3f}, q05={q05:.3f}, q25={q25:.3f}, q50={q50:.3f}, q75={q75:.3f}, q95={q95:.3f}, q99={q99:.3f}")
	if min_val is not None and max_val is not None:
		print(f"DEBUG {title}: min={min_val:.3f}, max={max_val:.3f}")
	
	# Only draw if we have valid data
	if q75 > q25:  # Valid IQR box
		# Box for IQR (q25-q75)
		box_height = q75 - q25
		ax.add_patch(plt.Rectangle((0.8, q25), 0.4, box_height, fill=True, color="#4C72B0", alpha=0.35, zorder=1))
		
		# Median line (q50)
		ax.hlines([q50], xmin=0.8, xmax=1.2, colors="black", lw=2, zorder=3)
	
	# Whiskers from box edges to q05/q95
	if q25 > q05:
		ax.plot([1, 1], [q05, q25], color="black", lw=1, zorder=2)
		ax.hlines([q05], xmin=0.9, xmax=1.1, colors="black", lw=1, zorder=2)
	
	if q95 > q75:
		ax.plot([1, 1], [q75, q95], color="black", lw=1, zorder=2)
		ax.hlines([q95], xmin=0.9, xmax=1.1, colors="black", lw=1, zorder=2)

	# Outlier points for q01 and q99 (only if they're outside whiskers)
	outlier_x = []
	outlier_y = []
	if q01 < q05:
		outlier_x.append(1)
		outlier_y.append(q01)
	if q99 > q95:
		outlier_x.append(1)
		outlier_y.append(q99)
	
	# Add extreme outliers (min/max beyond quantiles) if provided
	if min_val is not None and min_val < q01:
		outlier_x.append(1)
		outlier_y.append(min_val)
	if max_val is not None and max_val > q99:
		outlier_x.append(1)
		outlier_y.append(max_val)
	
	if outlier_x:
		ax.scatter(outlier_x, outlier_y, marker='o', facecolors='none', edgecolors='red', s=25, zorder=4)
	
	ax.set_xlim(0.6, 1.4)
	ax.set_xticks([1])
	ax.set_xticklabels([xlabel])
	ax.set_title(title)
	ax.grid(True, axis="y", alpha=0.25)
	ax.set_yscale(scale)
	
	# Set y-limits to show all data including extreme outliers
	all_vals = [v for v in [q01, q05, q25, q50, q75, q95, q99] if v is not None and np.isfinite(v)]
	if min_val is not None:
		all_vals.append(min_val)
	if max_val is not None:
		all_vals.append(max_val)
		
	if all_vals:
		vmin = min(all_vals)
		vmax = max(all_vals)
		
		if scale == "log":
			vmin = max(vmin, 1e-12)
			if vmax <= vmin:
				vmax = vmin * 10
			ax.set_ylim(vmin * 0.5, vmax * 2.0)
		else:
			if vmax > vmin:
				margin = (vmax - vmin) * 0.1
				ax.set_ylim(vmin - margin, vmax + margin)
			else:
				ax.set_ylim(vmin - 1, vmax + 1)


def _quantile_curve(ax, quantiles: np.ndarray, title: str, xlabel: str, scale: str) -> None:
	# Plot quantile levels vs value
	levels = np.array([1, 5, 25, 50, 75, 95, 99], dtype=float)
	ax.plot(quantiles, levels, marker="o", lw=1.75, color="#4C72B0")
	ax.set_xlabel(xlabel)
	ax.set_ylabel("quantile (%)")
	ax.set_title(title)
	ax.grid(True, alpha=0.25)
	ax.set_xscale(scale)
	ax.set_ylim(0, 100)

# ---- New: histogram & violin plotting from patch banks ----

def _choose_value_scale_for_banks(metric: str, vals: np.ndarray) -> str:
	if metric in _DEF_LOG_METRICS:
		mn = float(np.nanmin(vals)) if vals.size else 0.0
		mx = float(np.nanmax(vals)) if vals.size else 0.0
		if mn >= 0 and mx / max(mn + EPS, EPS) > 50:
			return "log"
	return "linear"


def _plot_hist(ax, metric: str, vals: np.ndarray, median_val: float = None) -> None:
	vals = vals[np.isfinite(vals)]
	if vals.size == 0:
		ax.set_visible(False)
		return
	scale = _choose_value_scale_for_banks(metric, vals)
	if scale == "log":
		pos = vals[vals > 0]
		if pos.size == 0:
			ax.set_visible(False)
			return
		mn = max(pos.min(initial=EPS), EPS)
		mx = max(vals.max(), mn * 10)
		bins = np.logspace(math.log10(mn), math.log10(mx), 60)
		ax.hist(pos, bins=bins, color="#4C72B0", alpha=0.7)
		ax.set_xscale("log")
		if median_val and median_val > 0:
			ax.axvline(median_val, color="crimson", lw=1.25, ls="--", label=f"median={median_val:.3g}")
	else:
		ax.hist(vals, bins=60, color="#4C72B0", alpha=0.7)
		if median_val is not None:
			ax.axvline(median_val, color="crimson", lw=1.25, ls="--", label=f"median={median_val:.3g}")
	if median_val is not None:
		ax.legend(frameon=False, fontsize=9)
	ax.set_title(f"Histogram: {metric}")
	ax.set_xlabel(metric)
	ax.set_ylabel("count")
	ax.grid(True, alpha=0.25)
	ax.set_yscale("log")  # counts often heavy-tailed


def _plot_violin(ax, metric: str, grouped: Dict[str, List[float]], overall: np.ndarray, group_by_radius: bool, median_val: float = None) -> None:
	if group_by_radius and grouped:
		order = list(LOCKED_RADIUS_LABELS)
		present = [g for g in order if g in grouped and len(grouped[g]) > 0]
		data = [np.asarray(grouped[g], dtype=float) for g in present]
		if not data:
			ax.set_visible(False)
			return
		parts = ax.violinplot(dataset=data, showmeans=False, showextrema=True, showmedians=True)
		ax.set_xticks(np.arange(1, len(present) + 1))
		ax.set_xticklabels(present)
		ax.set_title(f"Violin by radius class: {metric}")
		ax.set_xlabel("radius_class")
		ax.set_ylabel(metric)
	else:
		vals = overall[np.isfinite(overall)]
		if vals.size == 0:
			ax.set_visible(False)
			return
		parts = ax.violinplot(dataset=[vals], showmeans=False, showextrema=True, showmedians=True)
		ax.set_xticks([1])
		ax.set_xticklabels([metric])
		ax.set_title(f"Violin: {metric}")
	# Scale
	scale = _choose_value_scale_for_banks(metric, overall)
	if scale == "log":
		ax.set_yscale("log")
	if median_val is not None:
		ax.axhline(median_val, color="crimson", lw=1.0, ls="--", alpha=0.9)
	ax.grid(True, alpha=0.25)
	# Subtle styling
	for pc in parts['bodies']:
		pc.set_facecolor('#4C72B0')
		pc.set_edgecolor('black')
		pc.set_alpha(0.35)


def _regroup_locked_bins(patch_rows: List[Dict[str, Any]], metric: str, fg_only: bool, thresholds: Tuple[float, float, float]) -> Dict[str, List[float]]:
	"""Group metric values by locked radius bins calculated from radius_mean_mm.
	thresholds = (t0, t1, t2) in mm.
	"""
	groups: Dict[str, List[float]] = {k: [] for k in LOCKED_RADIUS_LABELS}
	for row in patch_rows:
		if metric not in row:
			continue
		v = float(row.get(metric, np.nan))
		if not np.isfinite(v):
			continue
		fg = float(row.get("fg_fraction", np.nan))
		if fg_only and not (np.isfinite(fg) and fg > 0):
			continue
		r = float(row.get("radius_mean_mm", np.nan))
		if not np.isfinite(r):
			continue
		t0, t1, t2 = thresholds
		if r <= t0:
			lab = "distal"
		elif r <= t1:
			lab = "subsegmental"
		elif r <= t2:
			lab = "segmental"
		else:
			lab = "main"
		groups[lab].append(v)
	return groups


def _format_metric_label(m: str) -> str:
	label = m.replace("_", " ")
	return label


def plot_all(stats_path: Path, out_dir: Path) -> None:
	stats = _load_stats(stats_path)
	metrics = _metric_list(stats)
	out_dir.mkdir(parents=True, exist_ok=True)
	
	# Per-metric 1x2 panels: box (quantiles) + quantile curve
	for m in metrics:
		mstats = stats[m]
		if not mstats:
			continue
		q = _quantile_array(mstats)
		scale = _choose_scale(m, mstats)
		min_val = mstats.get("min")
		max_val = mstats.get("max")
		fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
		_box_from_quantiles(axes[0], q, title=_format_metric_label(m), xlabel=_format_metric_label(m), scale=scale, min_val=min_val, max_val=max_val)
		_quantile_curve(axes[1], q, title=f"Quantiles of {m}", xlabel=_format_metric_label(m), scale=scale)
		for ext in ("png", "pdf"):
			fig.savefig(out_dir / f"{m}_panel.{ext}")
		plt.close(fig)
	
	# Summary grid: box-style for a curated subset
	subset = [
		"fg_fraction",
		"n_components",
		"lcc_fraction",
		"size_entropy",
		"border_touch_fraction",
		"radius_mean_mm",
		"radius_median_mm",
		"skel_len_mm",
		"endpoint_count",
		"bifurcation_count",
		"branch_density",
		"topology_score",
		"complexity_score",
	]
	n = len(subset)
	cols = 4
	rows = int(np.ceil(n / cols))
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.6), constrained_layout=True)
	axes = np.array(axes).reshape(rows, cols)
	for idx, m in enumerate(subset):
		ax = axes[idx // cols, idx % cols]
		mstats = stats.get(m, {})
		if not mstats:
			ax.axis("off")
			continue
		q = _quantile_array(mstats)
		scale = _choose_scale(m, mstats)
		min_val = mstats.get("min")
		max_val = mstats.get("max")
		_box_from_quantiles(ax, q, title=_format_metric_label(m), xlabel=m, scale=scale, min_val=min_val, max_val=max_val)
	# Turn off any unused axes
	for j in range(n, rows * cols):
		axes[j // cols, j % cols].axis("off")
	for ext in ("png", "pdf"):
		fig.savefig(out_dir / f"summary_panels.{ext}")
	plt.close(fig)

# ---- New: banks plotting entrypoint ----

def plot_from_banks(
	banks_dir: Path,
	out_dir: Path,
	metrics: List[str],
	group_by_radius: bool = True,
	sample_limit: int = 250000,
	per_case_limit: int = 0,
	use_locked_bins: bool = True,
	fg_only: bool = False,
	locked_thresholds: Tuple[float, float, float] = LOCKED_RADIUS_THRESHOLDS,
	global_medians: Dict[str, float] = None,
) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	metrics_data, grouped_data_stored, patch_rows = _load_patch_banks(banks_dir, metrics, sample_limit, per_case_limit)
	# Histograms
	for m in metrics:
		vals = np.asarray(metrics_data.get(m, []), dtype=float)
		if fg_only:
			vals = np.asarray([row[m] for row in patch_rows if m in row and np.isfinite(row[m]) and float(row.get("fg_fraction", np.nan)) > 0], dtype=float)
		if vals.size == 0:
			continue
		median_val = None
		if global_medians and m in global_medians:
			median_val = float(global_medians[m])
		fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.2), constrained_layout=True)
		_plot_hist(ax, m, vals, median_val=median_val)
		for ext in ("png", "pdf"):
			suffix = "_fg" if fg_only else ""
			fig.savefig(out_dir / f"hist_{m}{suffix}.{ext}")
		plt.close(fig)
	# Violins
	for m in metrics:
		vals_all = np.asarray(metrics_data.get(m, []), dtype=float)
		if vals_all.size == 0:
			continue
		median_val = None
		if global_medians and m in global_medians:
			median_val = float(global_medians[m])
		fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.4), constrained_layout=True)
		if group_by_radius:
			if use_locked_bins:
				grp = _regroup_locked_bins(patch_rows, m, fg_only=fg_only, thresholds=locked_thresholds)
			else:
				# Use stored radius_class from banks; optionally FG-only filter
				grp = {k: [] for k in LOCKED_RADIUS_LABELS}
				for row in patch_rows:
					if m not in row:
						continue
					if fg_only and not (float(row.get("fg_fraction", np.nan)) > 0):
						continue
					rc = row.get("radius_class_stored", "unknown")
					if rc in grp:
						v = float(row[m])
						if np.isfinite(v):
							grp[rc].append(v)
				# Ensure consistent key order/labels
				grp = {lab: grp.get(lab, []) for lab in LOCKED_RADIUS_LABELS}
			_plot_violin(ax, m, grp, np.asarray([], dtype=float), group_by_radius=True, median_val=median_val)
		else:
			vals = np.asarray([row[m] for row in patch_rows if m in row and (not fg_only or float(row.get("fg_fraction", np.nan)) > 0)], dtype=float)
			_plot_violin(ax, m, {}, vals, group_by_radius=False, median_val=median_val)
		for ext in ("png", "pdf"):
			suffix = "_fg" if fg_only else ""
			prefix = "locked" if use_locked_bins else "stored"
			fig.savefig(out_dir / f"violin_{prefix}_{m}{suffix}.{ext}")
		plt.close(fig)


def main():
	import argparse
	parser = argparse.ArgumentParser(description="Plot patch audit figures: global stats (quantiles) and/or hist/violin from patch banks")
	parser.add_argument("--stats_path", type=str, required=False, help="Path to global_stats.json produced by patch_audit.py")
	parser.add_argument("--banks_dir", type=str, required=False, help="Directory containing per-case patch banks (*.pkl.gz)")
	parser.add_argument("--out_dir", type=str, required=False, default=None, help="Directory to save figures (default: alongside input)")
	parser.add_argument("--metrics", type=str, nargs="*", default=[
		"fg_fraction",
		"radius_mean_mm",
		"radius_median_mm",
		"skel_len_mm",
		"endpoint_count",
		"bifurcation_count",
		"branch_density",
		"lcc_fraction",
		"size_entropy",
		"border_touch_fraction",
		"topology_score",
		"complexity_score",
		"sampling_weight",
	], help="Which metrics to plot from banks; ignored for stats panels")
	parser.add_argument("--group_by_radius", action="store_true", help="Group violin plots by radius_class when available")
	parser.add_argument("--sample_limit", type=int, default=250000, help="Max total patches to load across all banks (0 = no cap)")
	parser.add_argument("--per_case_limit", type=int, default=0, help="Optional max patches per case (0 = no cap)")
	parser.add_argument("--use_locked_bins", action="store_true", help="Regroup by locked radius thresholds instead of stored classes")
	parser.add_argument("--locked_thresholds", type=float, nargs=3, default=list(LOCKED_RADIUS_THRESHOLDS), help="Locked radius thresholds (t0 t1 t2) in mm")
	parser.add_argument("--fg_only", action="store_true", help="Filter to foreground-only patches (fg_fraction>0)")
	parser.add_argument("--no_median_labels", action="store_true", help="Do not annotate global medians on plots")
	args = parser.parse_args()

	stats_path = Path(args.stats_path) if args.stats_path else None
	banks_dir = Path(args.banks_dir) if args.banks_dir else None
	if not stats_path and not banks_dir:
		raise SystemExit("Provide --stats_path and/or --banks_dir")

	# Decide output directory
	if args.out_dir:
		out_root = Path(args.out_dir)
		out_root.mkdir(parents=True, exist_ok=True)
	else:
		# Prefer banks_dir/figs if banks mode, else stats parent/figs
		if banks_dir:
			out_root = banks_dir / "figs"
		else:
			out_root = stats_path.parent / "figs"
		out_root.mkdir(parents=True, exist_ok=True)

	if stats_path and stats_path.exists():
		print(f"Plotting quantile panels from: {stats_path}")
		plot_all(stats_path, out_root / "stats")

	global_medians: Dict[str, float] = {}
	if stats_path and stats_path.exists() and not args.no_median_labels:
		stats = _load_stats(stats_path)
		for k, v in stats.items():
			if isinstance(v, dict) and ("q50" in v):
				global_medians[k] = float(v["q50"])  # expected medians for annotation

	if banks_dir and banks_dir.exists():
		print(f"Plotting histogram/violin from banks: {banks_dir}")
		plot_from_banks(
			banks_dir=banks_dir,
			out_dir=out_root / "banks",
			metrics=args.metrics,
			group_by_radius=bool(args.group_by_radius),
			sample_limit=int(args.sample_limit) if args.sample_limit is not None else 250000,
			per_case_limit=int(args.per_case_limit) if args.per_case_limit is not None else 0,
			use_locked_bins=bool(args.use_locked_bins),
			fg_only=bool(args.fg_only),
			locked_thresholds=tuple(float(x) for x in args.locked_thresholds),
			global_medians=global_medians if global_medians else None,
		)
	print(f"Saved figures to: {out_root}")


if __name__ == "__main__":
	main() 