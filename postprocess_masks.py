"""针对四分类掩膜的后处理脚本。

该脚本读取彩色 PNG 掩膜并转换为 0–3 的整型标签，按照用户给定的流程
执行多数平滑、连通域重赋值、形态学开运算和面积守恒的黄绿边界抹平，最
终输出经过尺寸规整的标签图。脚本既支持单张图片，也支持整个文件夹的批
量处理。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


# --------------------------- 基础配色与常量定义 ---------------------------
# 颜色到标签的映射，使用 BGR 顺序以匹配 OpenCV 的默认读取方式。
COLOR_TO_LABEL = {
    (0, 0, 0): 0,  # 背景（黑）
    (0, 255, 0): 1,  # 绿色
    (0, 255, 255): 2,  # 黄色（BGR）
    (0, 0, 255): 3,  # 红色
}
LABEL_PRIORITY = [0, 1, 2, 3]
TARGET_SIZE: Tuple[int, int] = (1240, 1240)


# --------------------------- 通用工具函数 ---------------------------

def load_label_image(path: Path) -> np.ndarray:
    """读取 PNG 掩膜并转换为单通道标签图。"""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {path}")

    h, w, _ = image.shape
    labels = np.zeros((h, w), dtype=np.uint8)
    matched = np.zeros((h, w), dtype=bool)
    # 逐像素匹配颜色到标签，若出现未知颜色则报错提示。
    for color, label in COLOR_TO_LABEL.items():
        mask = np.all(image == np.array(color, dtype=np.uint8), axis=-1)
        labels[mask] = label
        matched |= mask
    if not np.all(matched):
        raise ValueError(f"输入图片存在未知颜色: {path}")
    return labels


def save_label_image(path: Path, labels: np.ndarray) -> None:
    """按照标签优先级生成单通道图像并保存。"""

    h, w = labels.shape
    output = np.zeros((h, w), dtype=np.uint8)
    for label in LABEL_PRIORITY:
        output[labels == label] = label
    resized = cv2.resize(output, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), resized)


def connected_components(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    """对二值掩膜执行 8 邻域连通域分割。"""

    num, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return num, comp


def boundary_band(labels: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """利用形态学梯度计算所有类别边界组成的窄带。"""

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    band = np.zeros_like(labels, dtype=bool)
    for value in range(4):
        mask = (labels == value).astype(np.uint8)
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(mask, kernel)
        band |= dilated != eroded
    return band


def majority_filter_on_band(labels: np.ndarray, band: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """在边界窄带上执行多数平滑。"""

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    counts = []
    for value in range(4):
        mask = (labels == value).astype(np.float32)
        count = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        counts.append(count)
    stacked = np.stack(counts, axis=-1)
    modes = np.argmax(stacked, axis=-1).astype(np.uint8)
    smoothed = labels.copy()
    smoothed[band] = modes[band]
    return smoothed


def dilate_red(labels: np.ndarray) -> np.ndarray:
    """对红色区域执行 1 像素膨胀。"""

    red_mask = labels == 3
    if not np.any(red_mask):
        return labels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(red_mask.astype(np.uint8), kernel)
    result = labels.copy()
    result[dilated.astype(bool)] = 3
    return result


def replace_component_with_neighbors(labels: np.ndarray, component_mask: np.ndarray, value: int) -> None:
    """将指定连通域替换为其相邻像素中最常见的颜色。"""

    if not np.any(component_mask):
        return
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    border = cv2.dilate(component_mask.astype(np.uint8), kernel).astype(bool)
    border &= ~component_mask
    if not np.any(border):
        labels[component_mask] = 0
        return
    neighbors = labels[border]
    counts = np.bincount(neighbors, minlength=4)
    counts[value] = 0
    new_value = int(np.argmax(counts))
    labels[component_mask] = new_value


def clean_green_components(labels: np.ndarray) -> None:
    """删除面积不足的绿色连通域，并依邻域颜色填补。"""

    green_mask = labels == 1
    total = int(np.sum(green_mask))
    if total == 0:
        return
    threshold = max(int(total * 0.1), 1)
    num, comp = connected_components(green_mask)
    for idx in range(1, num):
        component_mask = comp == idx
        if int(np.sum(component_mask)) < threshold:
            replace_component_with_neighbors(labels, component_mask, 1)


def keep_largest_component(labels: np.ndarray, value: int) -> None:
    """保留指定颜色的最大连通域，其余区域依邻域颜色重赋值。"""

    mask = labels == value
    if not np.any(mask):
        return
    num, comp = connected_components(mask)
    areas = [np.sum(comp == idx) for idx in range(1, num)]
    if not areas:
        return
    largest_idx = int(np.argmax(areas)) + 1
    for idx in range(1, num):
        if idx == largest_idx:
            continue
        component_mask = comp == idx
        replace_component_with_neighbors(labels, component_mask, value)


def fill_removed_regions(labels: np.ndarray, removed_mask: np.ndarray, target_value: int) -> None:
    """将形态学开运算移除的像素填充为最近的其他颜色。"""

    if not np.any(removed_mask):
        return
    binary = (labels == target_value).astype(np.uint8)
    if np.all(binary == 1):
        labels[removed_mask] = target_value
        return
    dist, indices = cv2.distanceTransformWithLabels(
        binary, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    zero_coords = np.column_stack(np.where(binary == 0))
    target_indices = indices[removed_mask] - 1
    target_indices = np.clip(target_indices, 0, len(zero_coords) - 1)
    nearest_coords = zero_coords[target_indices]
    new_values = labels[nearest_coords[:, 0], nearest_coords[:, 1]]
    labels[removed_mask] = new_values


def opening_and_refill(labels: np.ndarray, value: int, radius: int = 4) -> None:
    """对指定颜色执行开运算并填补空缺。"""

    mask = labels == value
    if not np.any(mask):
        return
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    removed = mask & ~opened
    labels[mask] = value
    fill_removed_regions(labels, removed, value)


def area_preserving_rethreshold(labels: np.ndarray) -> None:
    """在黄绿窄带内执行面积守恒的高斯重阈值。"""

    yellow_mask = labels == 2
    green_mask = labels == 1
    movable = yellow_mask | green_mask
    if not np.any(movable):
        return

    kernel_band = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dilated_y = cv2.dilate(yellow_mask.astype(np.uint8), kernel_band).astype(bool)
    dilated_g = cv2.dilate(green_mask.astype(np.uint8), kernel_band).astype(bool)
    band = movable & (dilated_y & dilated_g)
    if not np.any(band):
        band = movable

    yellow_fixed = yellow_mask & ~band
    yellow_to_allocate = int(np.sum(yellow_mask)) - int(np.sum(yellow_fixed))
    if yellow_to_allocate <= 0:
        labels[band] = 1
        labels[yellow_fixed] = 2
        return
    if yellow_to_allocate >= int(np.sum(band)):
        labels[band] = 2
        return

    blur = cv2.GaussianBlur(
        yellow_mask.astype(np.float32),
        (0, 0),
        sigmaX=7.0,
        sigmaY=7.0,
        borderType=cv2.BORDER_REPLICATE,
    )
    values = blur[band]
    flat_band_indices = np.flatnonzero(band)
    selected = np.zeros(len(flat_band_indices), dtype=bool)
    partition_index = len(values) - yellow_to_allocate
    threshold_value = np.partition(values, partition_index)[partition_index]
    larger = values > threshold_value
    selected[larger] = True
    remaining = yellow_to_allocate - int(np.sum(larger))
    if remaining > 0:
        equals = np.where(values == threshold_value)[0]
        chosen = equals[:remaining]
        selected[chosen] = True
    new_yellow_mask = np.zeros_like(movable, dtype=bool)
    new_yellow_mask.flat[flat_band_indices[selected]] = True

    labels[band] = 1
    labels[new_yellow_mask] = 2
    labels[yellow_fixed] = 2


def process_label(labels: np.ndarray) -> np.ndarray:
    """对单张标签图执行完整的后处理流程。"""

    band = boundary_band(labels)
    labels = majority_filter_on_band(labels, band)
    labels = dilate_red(labels)
    clean_green_components(labels)
    keep_largest_component(labels, 3)
    keep_largest_component(labels, 0)
    opening_and_refill(labels, 1, radius=4)
    opening_and_refill(labels, 2, radius=4)
    area_preserving_rethreshold(labels)
    return labels


def process_file(input_path: Path, output_path: Path) -> None:
    """处理单个文件并写入结果。"""

    labels = load_label_image(input_path)
    processed = process_label(labels)
    save_label_image(output_path, processed)


def gather_images(path: Path) -> Iterable[Path]:
    """收集需要处理的 PNG 图片路径。"""

    if path.is_file():
        return [path]
    return sorted(p for p in path.glob("*.png"))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="对分割掩膜执行后处理")
    parser.add_argument("input", type=Path, help="输入 PNG 文件或包含 PNG 的文件夹")
    parser.add_argument("output", type=Path, help="输出文件或目录")
    return parser.parse_args()


def main() -> None:
    """脚本入口：根据输入类型批量或单张处理并保存。"""

    args = parse_args()
    inputs = list(gather_images(args.input))
    if not inputs:
        raise FileNotFoundError("未找到任何 PNG 输入文件")

    treat_as_dir = args.output.is_dir() or args.output.suffix == "" or len(inputs) > 1
    if treat_as_dir:
        args.output.mkdir(parents=True, exist_ok=True)
        for path in inputs:
            output_path = args.output / path.name
            process_file(path, output_path)
    else:
        if len(inputs) > 1:
            raise ValueError("当输入为多个文件时，输出应为目录")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        process_file(inputs[0], args.output)


if __name__ == "__main__":
    main()
