# `postprocess_masks.py` 报错分析

## 报错信息回顾
- `IndexError: index 544441 is out of bounds for axis 0 with size 1024`
- 触发位置：`area_preserving_rethreshold` 函数。

## 根本原因
在 `area_preserving_rethreshold` 中，使用 `np.flatnonzero(band)` 得到了黄绿可移动窄带的**平铺索引** (`flat_band_indices`)。【F:postprocess_masks.py†L223-L243】随后代码试图通过 `new_yellow_mask[flat_band_indices[selected]] = True` 把这些平铺索引直接用于二维布尔图 `new_yellow_mask` 的第一维索引。然而 NumPy 会把这一写法理解为“按行下标取子数组”，因此当索引值大于图像高度（例如 `544441`，远大于 1024）时就会发生越界访问，抛出 `IndexError`。【F:postprocess_masks.py†L242-L243】

## 为什么出现 544441
图像在脚本中以二维数组表示，假设其尺寸约为 `1024×1024`，则整个数组按行主序拉平成一维后共有约 `1024×1024 ≈ 1,048,576` 个元素。`np.flatnonzero` 返回的正是这种**拉平成一维后的线性索引**。例如图像中部某个像素的线性索引可能是 `544441`，这个数字合法地落在 `[0, 1,048,575]` 之间；但把它直接拿来当作“第 544442 行”的索引去访问二维数组时，就会超过数组第一维的上界 1024，于是触发上述越界错误。【F:postprocess_masks.py†L231-L243】

## 结论
要修复此问题，需要把 `flat_band_indices` 用于数组的平铺视图（例如 `new_yellow_mask.flat[...]` 或 `np.put`），或者先把线性索引转换回二维坐标后再赋值。这样才能在保持面积守恒重阈值逻辑的同时避免越界访问。
