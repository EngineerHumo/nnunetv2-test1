"""compare_prediction.py
==========================

该脚本用于对指定目录下的多份 ONNX 分割模型进行批量评估。它会读取
`imagesTs` 与 `labelsTs` 中的测试图片和对应标签，将图片缩放到
(1, 3, 1024, 1024) 作为模型输入，获取模型输出的 (1, 4, 1024, 1024)
预测结果后，与标签的 one-hot 表达进行对比，计算 ACC、Dice、IoU、
Precision、Recall 等指标，对模型分割性能进行全面评估。同时脚本还会
在每个 ONNX 文件所在目录下新建文件夹，按“模型名+图像名+fold 信息”
命名保存彩色 PNG 预测图（0=黑、1=红、2=黄、3=绿），并将每个模型的
性能指标保存到 CSV 表格中。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

# 常量定义
NUM_CLASSES = 4
TARGET_SIZE = (1024, 1024)  # (width, height)
TARGET_WIDTH, TARGET_HEIGHT = TARGET_SIZE
COLOR_MAP = np.array(
    [
        [0, 0, 0],       # 0 -> black
        [255, 0, 0],     # 1 -> red
        [255, 255, 0],   # 2 -> yellow
        [0, 255, 0],     # 3 -> green
    ],
    dtype=np.uint8,
)


@dataclass
class Metrics:
    accuracy: float
    mean_dice: float
    mean_iou: float
    mean_precision: float
    mean_recall: float
    dice_per_class: Sequence[float]
    iou_per_class: Sequence[float]
    precision_per_class: Sequence[float]
    recall_per_class: Sequence[float]
    support_per_class: Sequence[int]
    num_samples: int


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量评估 ONNX 分割模型，对测试集生成预测并计算多种指标。"
    )
    parser.add_argument(
        "onnx_dir",
        type=Path,
        help="存放多个 ONNX 文件的目录。",
    )
    parser.add_argument(
        "images_dir",
        type=Path,
        help="测试图片所在的 imagesTs 目录。",
    )
    parser.add_argument(
        "labels_dir",
        type=Path,
        help="测试标签所在的 labelsTs 目录。",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path("onnx_metrics.csv"),
        help="保存评估结果的 CSV 文件路径（默认: onnx_metrics.csv）。",
    )
    parser.add_argument(
        "--image-extensions",
        nargs="*",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
        help="允许的图像扩展名（默认包含常见格式）。",
    )
    parser.add_argument(
        "--label-extensions",
        nargs="*",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"],
        help="允许的标签扩展名（默认包含常见格式与 .npy）。",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="是否将输入图像归一化到 [0, 1]（默认不归一化）。",
    )
    return parser.parse_args(argv)


def collect_pairs(
    images_dir: Path,
    labels_dir: Path,
    image_exts: Sequence[str],
    label_exts: Sequence[str],
) -> List[Tuple[Path, Path]]:
    image_map: Dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in image_exts:
            image_map[path.stem] = path

    label_map: Dict[str, Path] = {}
    for path in sorted(labels_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in label_exts:
            label_map[path.stem] = path

    pairs: List[Tuple[Path, Path]] = []
    missing_labels: List[str] = []
    for stem, img_path in image_map.items():
        label_path = label_map.get(stem)
        if label_path is None:
            missing_labels.append(stem)
            continue
        pairs.append((img_path, label_path))

    if missing_labels:
        missing_str = ", ".join(missing_labels)
        print(f"警告: 下列图像未找到对应标签，将被跳过: {missing_str}")

    if not pairs:
        raise RuntimeError("未能匹配到任何图像-标签对，请检查目录与文件名。")

    return pairs


def load_image(path: Path, normalize: bool) -> np.ndarray:
    """读取图像并调整为模型输入形状 (1, 3, 1024, 1024)。"""
    with Image.open(path) as img:
        image = img.convert("RGB")
        resized = image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    if normalize:
        array /= 255.0
    array = array.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    array = np.expand_dims(array, axis=0)
    return array


def load_label(path: Path) -> np.ndarray:
    """读取标签并调整为 (1024, 1024) 的整型数组。"""
    if path.suffix.lower() == ".npy":
        label = np.load(path)
        if label.ndim == 3:
            label = label.squeeze()
    else:
        with Image.open(path) as img:
            label_img = img.convert("L")
            label_img = label_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.NEAREST)
        label = np.asarray(label_img, dtype=np.int64)
    return label


def softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def evaluate_model(
    model_path: Path,
    pairs: Sequence[Tuple[Path, Path]],
    normalize: bool,
) -> Metrics:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    total_pixels = 0
    correct_pixels = 0
    tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    fp = np.zeros(NUM_CLASSES, dtype=np.int64)
    fn = np.zeros(NUM_CLASSES, dtype=np.int64)
    support = np.zeros(NUM_CLASSES, dtype=np.int64)

    for image_path, label_path in pairs:
        image_tensor = load_image(image_path, normalize)
        label_array = load_label(label_path)
        if label_array.shape != TARGET_SIZE:
            raise ValueError(
                f"标签 {label_path} 尺寸为 {label_array.shape}，无法调整到 {TARGET_SIZE}。"
            )
        unique_values = np.unique(label_array)
        if np.any((unique_values < 0) | (unique_values >= NUM_CLASSES)):
            raise ValueError(
                f"标签 {label_path} 包含非法类别值 {unique_values}，期望范围为 [0, {NUM_CLASSES - 1}]。"
            )

        ort_inputs = {input_name: image_tensor}
        outputs = session.run(None, ort_inputs)
        logits = outputs[0]
        if logits.shape != (1, NUM_CLASSES, TARGET_HEIGHT, TARGET_WIDTH):
            raise ValueError(
                f"模型输出形状 {logits.shape} 与期望的 (1, {NUM_CLASSES}, 1024, 1024) 不符。"
            )
        logits = logits[0]
        probabilities = softmax(logits, axis=0)
        pred_classes = np.argmax(probabilities, axis=0)

        total_pixels += label_array.size
        correct_pixels += (pred_classes == label_array).sum()

        for class_idx in range(NUM_CLASSES):
            pred_mask = pred_classes == class_idx
            true_mask = label_array == class_idx
            tp[class_idx] += np.logical_and(pred_mask, true_mask).sum()
            fp[class_idx] += np.logical_and(pred_mask, np.logical_not(true_mask)).sum()
            fn[class_idx] += np.logical_and(np.logical_not(pred_mask), true_mask).sum()
            support[class_idx] += true_mask.sum()

        save_prediction_image(
            model_path=model_path,
            image_path=image_path,
            prediction=pred_classes,
        )

    accuracy = safe_divide(correct_pixels, total_pixels)

    dice_per_class = []
    iou_per_class = []
    precision_per_class = []
    recall_per_class = []

    for class_idx in range(NUM_CLASSES):
        dice = safe_divide(2 * tp[class_idx], 2 * tp[class_idx] + fp[class_idx] + fn[class_idx])
        iou = safe_divide(tp[class_idx], tp[class_idx] + fp[class_idx] + fn[class_idx])
        precision = safe_divide(tp[class_idx], tp[class_idx] + fp[class_idx])
        recall = safe_divide(tp[class_idx], tp[class_idx] + fn[class_idx])

        dice_per_class.append(dice)
        iou_per_class.append(iou)
        precision_per_class.append(precision)
        recall_per_class.append(recall)

    def mean_with_support(values: Sequence[float], supports: Sequence[int]) -> float:
        filtered = [v for v, s in zip(values, supports) if s > 0]
        if not filtered:
            return 0.0
        return float(np.mean(filtered))

    mean_dice = mean_with_support(dice_per_class, support)
    mean_iou = mean_with_support(iou_per_class, support)
    mean_precision = mean_with_support(precision_per_class, support)
    mean_recall = mean_with_support(recall_per_class, support)

    return Metrics(
        accuracy=accuracy,
        mean_dice=mean_dice,
        mean_iou=mean_iou,
        mean_precision=mean_precision,
        mean_recall=mean_recall,
        dice_per_class=dice_per_class,
        iou_per_class=iou_per_class,
        precision_per_class=precision_per_class,
        recall_per_class=recall_per_class,
        support_per_class=support.tolist(),
        num_samples=len(pairs),
    )


def infer_fold_suffix(model_path: Path) -> str:
    match = re.search(r"fold[_-]?(\d+)", model_path.stem, re.IGNORECASE)
    if match:
        return f"fold{match.group(1)}"
    if "foldall" in model_path.stem.lower() or "all" in model_path.stem.lower():
        return "foldall"
    return "foldunknown"


def save_prediction_image(model_path: Path, image_path: Path, prediction: np.ndarray) -> None:
    if prediction.shape != (TARGET_HEIGHT, TARGET_WIDTH):
        raise ValueError("预测图尺寸不正确，无法保存彩色结果。")

    color_prediction = COLOR_MAP[prediction]

    fold_suffix = infer_fold_suffix(model_path)
    result_dir = model_path.parent / f"{model_path.stem}_predictions"
    result_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{model_path.stem}__{image_path.stem}__{fold_suffix}.png"
    save_path = result_dir / filename

    Image.fromarray(color_prediction).save(save_path)


def build_table_row(model_path: Path, metrics: Metrics) -> Dict[str, float]:
    row: Dict[str, float] = {
        "model_path": str(model_path),
        "num_samples": metrics.num_samples,
        "accuracy": metrics.accuracy,
        "mean_dice": metrics.mean_dice,
        "mean_iou": metrics.mean_iou,
        "mean_precision": metrics.mean_precision,
        "mean_recall": metrics.mean_recall,
    }

    for idx in range(NUM_CLASSES):
        row[f"dice_c{idx}"] = metrics.dice_per_class[idx]
        row[f"iou_c{idx}"] = metrics.iou_per_class[idx]
        row[f"precision_c{idx}"] = metrics.precision_per_class[idx]
        row[f"recall_c{idx}"] = metrics.recall_per_class[idx]
        row[f"support_c{idx}"] = metrics.support_per_class[idx]

    return row


def write_metrics_table(rows: Sequence[Dict[str, float]], output_path: Path) -> None:
    if not rows:
        print("没有可写入的评估结果。")
        return

    fieldnames: List[str] = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"指标表格已写入 {output_path}")


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if not args.onnx_dir.is_dir():
        raise NotADirectoryError(f"ONNX 目录不存在: {args.onnx_dir}")
    if not args.images_dir.is_dir():
        raise NotADirectoryError(f"图像目录不存在: {args.images_dir}")
    if not args.labels_dir.is_dir():
        raise NotADirectoryError(f"标签目录不存在: {args.labels_dir}")

    pairs = collect_pairs(args.images_dir, args.labels_dir, args.image_extensions, args.label_extensions)

    onnx_files = sorted(args.onnx_dir.glob("*.onnx"))
    if not onnx_files:
        raise RuntimeError(f"在 {args.onnx_dir} 中未找到任何 ONNX 文件。")

    rows: List[Dict[str, float]] = []

    for model_path in onnx_files:
        print(f"正在评估模型: {model_path}")
        metrics = evaluate_model(model_path, pairs, args.normalize)
        print(
            "  -> ACC: {acc:.4f}, mean Dice: {dice:.4f}, mean IoU: {iou:.4f}".format(
                acc=metrics.accuracy, dice=metrics.mean_dice, iou=metrics.mean_iou
            )
        )
        rows.append(build_table_row(model_path, metrics))

    write_metrics_table(rows, args.output_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
