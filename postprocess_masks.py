"""Post-processing utilities for multi-class segmentation masks.

This script implements a configurable post-processing pipeline that operates on
RGB pseudo-colour segmentation masks where the colour palette is fixed to:

    0 -> background : black  (0,   0,   0)
    1 -> green      : green  (0, 255,   0)
    2 -> yellow     : yellow (255,255, 0)
    3 -> red        : red    (255,  0,  0)

The post-processing operations are inspired by common clean-up steps for
cellular ring segmentations and include majority filtering around boundaries,
per-class morphological clean-up, and topology constraints between classes.

The command line interface supports batched processing, optional metric
evaluation against ground-truth labels, configuration via JSON/YAML files, and
parallel execution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # Optional dependency for YAML configs.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml is optional.
    yaml = None


Palette = Dict[int, Tuple[int, int, int]]


DEFAULT_PALETTE: Palette = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (255, 255, 0),
    3: (255, 0, 0),
}


CLASS_NAMES = {0: "background", 1: "green", 2: "yellow", 3: "red"}


@dataclass
class RelationConfig:
    """Configuration block for inter-class constraints."""

    yellow_include_red: bool = True
    red_margin: int = 5
    add_green_band: bool = True


@dataclass
class ProcessingConfig:
    """Aggregated post-processing hyper-parameters."""

    r_open: Optional[int] = None
    r_close: Optional[int] = None
    r_band: Optional[int] = None
    area_fracs: Dict[int, float] = field(
        default_factory=lambda: {1: 0.001, 2: 0.001, 3: 0.004}
    )
    hole_fracs: Dict[int, float] = field(
        default_factory=lambda: {1: 0.004, 2: 0.006, 3: 0.003}
    )
    min_area_px: Dict[int, int] = field(default_factory=dict)
    max_hole_px: Dict[int, int] = field(default_factory=dict)
    class_priority: List[int] = field(default_factory=lambda: [3, 2, 1, 0])
    enable_mode_filter: bool = True
    mode_kernel: int = 5
    enforce_relations: RelationConfig = field(default_factory=RelationConfig)


# ---------------------------------------------------------------------------
# Utility helpers


def parse_size(size_str: str) -> Tuple[int, int]:
    """Parse size specification formatted as "<width>x<height>"."""

    try:
        width_str, height_str = size_str.lower().split("x")
        width, height = int(width_str), int(height_str)
    except ValueError as exc:  # pragma: no cover - user error path.
        raise argparse.ArgumentTypeError(
            f"--target_size must be formatted as <width>x<height>, got '{size_str}'."
        ) from exc
    if width <= 0 or height <= 0:  # pragma: no cover - user error path.
        raise argparse.ArgumentTypeError("Target width/height must be positive.")
    return width, height


def ensure_dir(path: Path) -> None:
    """Create directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Optional[Path]) -> Dict[str, object]:
    """Load configuration overrides from JSON or YAML file."""

    if path is None:
        return {}
    if not path.exists():  # pragma: no cover - user error path.
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() in {".json"}:
        return json.loads(path.read_text())
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:  # pragma: no cover - missing optional dependency.
            raise RuntimeError("PyYAML is required to parse YAML config files.")
        return yaml.safe_load(path.read_text())
    raise ValueError(
        f"Unsupported config extension '{path.suffix}'. Use JSON or YAML."
    )


def merge_config(overrides: Dict[str, object]) -> ProcessingConfig:
    """Merge user-provided overrides with the default configuration."""

    cfg = ProcessingConfig()
    if not overrides:
        return cfg

    for attr in ("r_open", "r_close", "r_band", "enable_mode_filter", "mode_kernel"):
        if attr in overrides:
            setattr(cfg, attr, overrides[attr])

    if "class_priority" in overrides:
        cfg.class_priority = list(overrides["class_priority"])  # type: ignore

    if "area_fracs" in overrides:
        cfg.area_fracs.update({int(k): float(v) for k, v in overrides["area_fracs"].items()})

    if "hole_fracs" in overrides:
        cfg.hole_fracs.update({int(k): float(v) for k, v in overrides["hole_fracs"].items()})

    if "min_area_px" in overrides:
        cfg.min_area_px.update({int(k): int(v) for k, v in overrides["min_area_px"].items()})

    if "max_hole_px" in overrides:
        cfg.max_hole_px.update({int(k): int(v) for k, v in overrides["max_hole_px"].items()})

    if "enforce_relations" in overrides:
        rel_overrides = overrides["enforce_relations"]
        if isinstance(rel_overrides, dict):
            rel = cfg.enforce_relations
            if "yellow_include_red" in rel_overrides:
                rel.yellow_include_red = bool(rel_overrides["yellow_include_red"])
            if "red_margin" in rel_overrides:
                rel.red_margin = int(rel_overrides["red_margin"])
            if "add_green_band" in rel_overrides:
                rel.add_green_band = bool(rel_overrides["add_green_band"])

    return cfg


def rgb_to_labels(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB mask to label indices."""

    mapping = {
        (0, 0, 0): 0,
        (0, 255, 0): 1,
        (255, 255, 0): 2,
        (255, 0, 0): 3,
    }
    flat_rgb = rgb.reshape(-1, 3)
    labels = np.zeros(flat_rgb.shape[0], dtype=np.uint8)
    for colour, label in mapping.items():
        mask = np.all(flat_rgb == colour, axis=1)
        labels[mask] = label
    return labels.reshape(rgb.shape[:2])


def labels_to_rgb(labels: np.ndarray, palette: Palette = DEFAULT_PALETTE) -> np.ndarray:
    """Convert label map back to RGB image."""

    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, colour in palette.items():
        rgb[labels == label] = colour
    return rgb


def resize_labels(labels: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize label map using nearest-neighbour interpolation."""

    width, height = size
    resized = cv2.resize(
        labels,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized


def compute_kernel(radius: int) -> Optional[np.ndarray]:
    """Create an elliptical structuring element for the given radius."""

    radius = int(radius)
    if radius <= 0:
        return None
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def morphological_gradient(label_map: np.ndarray) -> np.ndarray:
    """Compute boundary mask using morphological gradient."""

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(label_map, kernel)
    eroded = cv2.erode(label_map, kernel)
    gradient = dilated != eroded
    return gradient.astype(np.uint8)


def mode_filter_on_boundaries(
    labels: np.ndarray,
    kernel_size: int,
    priority: Sequence[int],
) -> np.ndarray:
    """Apply majority filter within boundary band defined by morphological gradient."""

    boundary = morphological_gradient(labels)
    boundary = cv2.dilate(boundary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if kernel_size <= 1:
        return labels

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    class_maps = []
    for cls in range(len(DEFAULT_PALETTE)):
        binary = (labels == cls).astype(np.uint8)
        counts = cv2.filter2D(binary, -1, kernel, borderType=cv2.BORDER_REFLECT)
        class_maps.append(counts)
    stacked = np.stack(class_maps, axis=0)

    # Tie-breaking by priority: higher priority classes come first.
    priority_indices = {cls: idx for idx, cls in enumerate(priority)}
    tie_breaker = np.zeros_like(stacked, dtype=np.float32)
    for cls in range(stacked.shape[0]):
        # Lower index (higher priority) should win; subtract small epsilon.
        tie_breaker[cls] = -priority_indices.get(cls, len(priority)) * 1e-3
    stacked = stacked.astype(np.float32) + tie_breaker

    filtered = stacked.argmax(axis=0).astype(labels.dtype)
    result = labels.copy()
    result[boundary.astype(bool)] = filtered[boundary.astype(bool)]
    return result


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components below the specified area threshold."""

    if min_area <= 0:
        return mask

    num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    to_remove = []
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < min_area:
            to_remove.append(idx)
    if not to_remove:
        return mask
    output = mask.copy()
    for idx in to_remove:
        output[label_map == idx] = False
    return output


def fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Fill holes within mask whose area is below threshold."""

    if max_area <= 0:
        return mask

    inv = (~mask).astype(np.uint8)
    num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
        inv, connectivity=8
    )
    h, w = mask.shape
    output = mask.copy()
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        width = stats[idx, cv2.CC_STAT_WIDTH]
        height = stats[idx, cv2.CC_STAT_HEIGHT]
        touches_border = x == 0 or y == 0 or (x + width) >= w or (y + height) >= h
        if not touches_border and area <= max_area:
            output[label_map == idx] = True
    return output


def apply_morphology(mask: np.ndarray, r_open: int, r_close: int) -> np.ndarray:
    """Apply opening followed by closing with given radii."""

    result = mask.astype(np.uint8)
    if r_open > 0:
        kernel_open = compute_kernel(r_open)
        if kernel_open is not None:
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_open)
    if r_close > 0:
        kernel_close = compute_kernel(r_close)
        if kernel_close is not None:
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_close)
    return result.astype(bool)


def enforce_relations(
    green: np.ndarray,
    yellow: np.ndarray,
    red: np.ndarray,
    cfg: ProcessingConfig,
    min_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply topology constraints between class layers."""

    rel = cfg.enforce_relations
    if rel.yellow_include_red:
        margin = max(1, int(rel.red_margin))
        kernel = compute_kernel(margin)
        if kernel is None:
            kernel = compute_kernel(1)
        if kernel is not None:
            red_dilated = cv2.dilate(red.astype(np.uint8), kernel).astype(bool)
            yellow = np.logical_or(yellow, red_dilated)
        red = np.logical_and(red, yellow)

    green = np.logical_and(green, np.logical_not(yellow))

    if rel.add_green_band:
        radius = cfg.r_band if cfg.r_band is not None else max(1, round(0.01 * min_dim))
        kernel = compute_kernel(radius)
        if kernel is not None:
            band = cv2.dilate(yellow.astype(np.uint8), kernel).astype(bool)
            band = np.logical_and(band, np.logical_not(yellow))
            green = np.logical_or(green, band)

    return green, yellow, red


def compose_labels(
    green: np.ndarray, yellow: np.ndarray, red: np.ndarray, priority: Sequence[int]
) -> np.ndarray:
    """Compose class layers into single label map according to priority."""

    h, w = green.shape
    labels = np.zeros((h, w), dtype=np.uint8)
    masks = {1: green, 2: yellow, 3: red}
    for cls in priority:
        if cls == 0:
            continue
        mask = masks.get(cls)
        if mask is None:
            continue
        labels[mask] = cls
    return labels


def auto_radius(base: Optional[int], scale: float, min_dim: int) -> int:
    """Return radius using explicit value or scale * min_dim."""

    if base is not None:
        return int(base)
    radius = int(round(scale * min_dim))
    return max(1, radius)


def process_label_map(
    label_map: np.ndarray,
    cfg: ProcessingConfig,
) -> np.ndarray:
    """Apply the full post-processing pipeline to a label map."""

    h, w = label_map.shape
    min_dim = min(h, w)

    labels = label_map.copy()
    if cfg.enable_mode_filter:
        kernel = max(1, int(cfg.mode_kernel))
        if kernel % 2 == 0:
            kernel += 1
        labels = mode_filter_on_boundaries(labels, kernel, cfg.class_priority)

    r_open = auto_radius(cfg.r_open, 0.004, min_dim)
    r_close = auto_radius(cfg.r_close, 0.012, min_dim)

    class_masks = {}
    for cls in (1, 2, 3):
        mask = labels == cls
        mask = apply_morphology(mask, r_open, r_close)
        min_area = cfg.min_area_px.get(cls)
        if min_area is None:
            min_area = int(round(cfg.area_fracs.get(cls, 0.0) * h * w))
        mask = remove_small_components(mask, min_area)

        max_hole = cfg.max_hole_px.get(cls)
        if max_hole is None:
            max_hole = int(round(cfg.hole_fracs.get(cls, 0.0) * h * w))
        mask = fill_small_holes(mask, max_hole)

        class_masks[cls] = mask

    green, yellow, red = (
        class_masks.get(1, np.zeros_like(labels, dtype=bool)),
        class_masks.get(2, np.zeros_like(labels, dtype=bool)),
        class_masks.get(3, np.zeros_like(labels, dtype=bool)),
    )

    green, yellow, red = enforce_relations(green, yellow, red, cfg, min_dim)
    labels = compose_labels(green, yellow, red, cfg.class_priority)

    return labels


# ---------------------------------------------------------------------------
# Evaluation helpers


def safe_read_image(path: Path) -> np.ndarray:
    """Read image from disk ensuring RGB channel order."""

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:  # pragma: no cover - user error path.
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_matching_label(pred_path: Path, label_dir: Path) -> Optional[Path]:
    """Locate ground-truth label file corresponding to a prediction."""

    candidates = [pred_path.name]
    stem = pred_path.stem
    if "Prediction" in stem:
        candidates.append(stem.replace("Prediction", "Label") + pred_path.suffix)
    if stem.endswith("_Pred"):
        candidates.append(stem[:-5] + "_Label" + pred_path.suffix)
    for name in candidates:
        candidate_path = label_dir / name
        if candidate_path.exists():
            return candidate_path
    return None


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """Compute IoU and Dice score per class."""

    metrics: Dict[int, Tuple[float, float]] = {}
    for cls in (1, 2, 3):
        pred_mask = pred == cls
        gt_mask = gt == cls
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        pred_area = pred_mask.sum()
        gt_area = gt_mask.sum()
        if union == 0:
            iou = 1.0 if pred_area == gt_area == 0 else 0.0
        else:
            iou = intersection / union
        denom = pred_area + gt_area
        if denom == 0:
            dice = 1.0
        else:
            dice = 2 * intersection / denom
        metrics[cls] = (iou, dice)
    return metrics


def summarize_metrics(metrics_list: List[Dict[int, Tuple[float, float]]]) -> None:
    """Print aggregated IoU and Dice scores."""

    if not metrics_list:
        return
    print("\nEvaluation metrics (IoU / Dice):")
    header = ["Class", "IoU", "Dice"]
    print("{:<12s} {:>10s} {:>10s}".format(*header))
    sums = {cls: np.zeros(2, dtype=float) for cls in (1, 2, 3)}
    for metrics in metrics_list:
        for cls, values in metrics.items():
            sums[cls] += values
    for cls in (1, 2, 3):
        mean_values = sums[cls] / len(metrics_list)
        print(
            f"{CLASS_NAMES[cls]:<12s} {mean_values[0]:>10.4f} {mean_values[1]:>10.4f}"
        )
    means = np.array(list(sums.values())) / len(metrics_list)
    foreground_mean_iou = means[:, 0].mean()
    foreground_mean_dice = means[:, 1].mean()
    print(f"Foreground mean IoU : {foreground_mean_iou:.4f}")
    print(f"Foreground mean Dice: {foreground_mean_dice:.4f}\n")


# ---------------------------------------------------------------------------
# Processing entry points


def process_file(
    path: Path,
    out_dir: Path,
    cfg: ProcessingConfig,
    target_size: Optional[Tuple[int, int]],
) -> Tuple[Path, Path, Dict[str, object]]:
    """Process a single RGB mask and save the cleaned result."""

    rgb = safe_read_image(path)
    labels = rgb_to_labels(rgb)
    if target_size is not None:
        labels = resize_labels(labels, target_size)

    processed = process_label_map(labels, cfg)
    rgb_out = labels_to_rgb(processed)

    ensure_dir(out_dir)
    out_path = out_dir / path.name
    cv2.imwrite(str(out_path), cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR))

    stats = {cls: int((processed == cls).sum()) for cls in (0, 1, 2, 3)}
    return path, out_path, stats


def run_self_test(cfg: ProcessingConfig) -> None:
    """Run a simple self-test on a synthetic label map."""

    print("Running self-test (no input directory provided)...")
    h, w = 128, 128
    labels = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(labels, (w // 2, h // 2), 50, 1, thickness=10)
    cv2.circle(labels, (w // 2, h // 2), 30, 2, thickness=8)
    cv2.circle(labels, (w // 2, h // 2), 20, 3, thickness=-1)

    processed = process_label_map(labels, cfg)
    unique, counts = np.unique(processed, return_counts=True)
    summary = dict(zip(unique.tolist(), counts.tolist()))
    print("Self-test completed. Class pixel counts:", summary)


def gather_files(in_dir: Path, pattern: str) -> List[Path]:
    """Collect input files matching the glob pattern."""

    return sorted(in_dir.rglob(pattern))


def postprocess_directory(
    in_dir: Path,
    out_dir: Path,
    pattern: str,
    cfg: ProcessingConfig,
    target_size: Optional[Tuple[int, int]],
    workers: int,
) -> List[Tuple[Path, Path, Dict[str, object]]]:
    """Process all matching files in a directory using a thread pool."""

    files = gather_files(in_dir, pattern)
    if not files:
        print(f"No files matched pattern '{pattern}' under {in_dir}.")
        return []

    start_time = time.time()
    results: List[Tuple[Path, Path, Dict[str, object]]] = []
    workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_file, path, out_dir, cfg, target_size): path
            for path in files
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:  # pragma: no cover - propagated to caller.
                print(f"Error processing {futures[future]}: {exc}")
                raise

    elapsed = time.time() - start_time
    print(
        f"Processed {len(results)} file(s) in {elapsed:.2f} s. Output saved to {out_dir}."
    )
    return results


def evaluate_predictions(
    results: List[Tuple[Path, Path, Dict[str, object]]],
    label_dir: Path,
) -> None:
    """Compute and print evaluation metrics against ground-truth labels."""

    metrics: List[Dict[int, Tuple[float, float]]] = []
    for pred_path, processed_path, _ in results:
        label_path = find_matching_label(pred_path, label_dir)
        if label_path is None:
            print(f"Ground-truth label not found for {pred_path.name}, skipping.")
            continue
        gt_rgb = safe_read_image(label_path)
        gt_labels = rgb_to_labels(gt_rgb)

        processed_rgb = safe_read_image(processed_path)
        processed_labels = rgb_to_labels(processed_rgb)

        if processed_labels.shape != gt_labels.shape:
            processed_labels = resize_labels(
                processed_labels, (gt_labels.shape[1], gt_labels.shape[0])
            )

        metrics.append(compute_metrics(processed_labels, gt_labels))

    summarize_metrics(metrics)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in_dir",
        type=Path,
        default=None,
        help="Directory with predicted RGB masks. If omitted, runs a self-test.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory to store processed masks.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_Prediction.png",
        help="Glob pattern to match prediction files (default: *_Prediction.png).",
    )
    parser.add_argument(
        "--label_dir",
        type=Path,
        default=None,
        help="Optional directory with ground-truth labels for evaluation.",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of worker threads (default: 8)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON or YAML configuration file to override hyper-parameters.",
    )
    parser.add_argument(
        "--target_size",
        type=parse_size,
        default=None,
        help="Force resize to WIDTHxHEIGHT using nearest-neighbour interpolation.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    overrides = load_config(args.config)
    cfg = merge_config(overrides)

    if args.in_dir is None or (args.in_dir.exists() and not any(args.in_dir.iterdir())):
        run_self_test(cfg)
        return 0

    if not args.in_dir.exists():  # pragma: no cover - user error path.
        raise FileNotFoundError(f"Input directory does not exist: {args.in_dir}")

    if args.out_dir is None:
        raise ValueError("--out_dir must be specified when processing real data.")

    results = postprocess_directory(
        args.in_dir,
        args.out_dir,
        args.pattern,
        cfg,
        args.target_size,
        args.workers,
    )

    if args.label_dir is not None and results:
        if not args.label_dir.exists():  # pragma: no cover - user error path.
            raise FileNotFoundError(f"Label directory does not exist: {args.label_dir}")
        evaluate_predictions(results, args.label_dir)

    total_pixels = sum(sum(stats.values()) for _, _, stats in results)
    print(f"Processed total pixels: {total_pixels}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    sys.exit(main())

