import json
from pathlib import Path
from typing import Dict, List

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


def _load_stats(stats_path: Path) -> Dict[str, Dict[str, float]]:
	with open(stats_path, "r") as f:
		stats = json.load(f)
	return stats


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


def main():
	import argparse
	parser = argparse.ArgumentParser(description="Plot patch audit global stats into thesis-ready figures")
	parser.add_argument("--stats_path", type=str, required=True, help="Path to global_stats.json produced by patch_audit.py")
	parser.add_argument("--out_dir", type=str, required=False, default=None, help="Directory to save figures (default: alongside stats)")
	args = parser.parse_args()

	stats_path = Path(args.stats_path)
	assert stats_path.exists(), f"Stats file not found: {stats_path}"
	out_dir = Path(args.out_dir) if args.out_dir else stats_path.parent / "figs"
	plot_all(stats_path, out_dir)
	print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
	main() 