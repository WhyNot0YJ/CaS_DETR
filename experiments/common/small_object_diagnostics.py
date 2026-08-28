"""Small-object 低样本类别排除的诊断规则（YOLO / DEIM / Faster R-CNN 共用）。

正式协议只有 ``dairv2x`` 与 ``uadetrac``：

- DAIR-V2X：bus / truck 的 small GT 少于 20 个，不参与 small-object 诊断均值；
- UA-DETRAC：bus 的 small GT 少于 20 个被排除；``others`` 没有 small GT，
  由 COCOeval 的 ``precision > -1`` 掩码自然忽略，无需显式排除。

官方 COCO mAP 与训练期 AP_small 计算不受本模块影响；这里只给出
「协议 → (排除类别, 诊断指标列名)」的映射，避免各评测入口各自硬编码。
"""

from __future__ import annotations

from typing import Tuple

# Listed categories have fewer than 20 small ground-truth instances in the
# respective test set. The official COCO AP_small remains unchanged.
DAIR_SMALL_EXCLUDED_CATEGORIES = ("bus", "truck")
DAIR_SMALL_DIAGNOSTIC_KEYS = (
    "AP_small_50_excl_bus_truck",
    "AP_small_5095_excl_bus_truck",
)
UADETRAC_SMALL_EXCLUDED_CATEGORIES = ("bus",)
UADETRAC_SMALL_DIAGNOSTIC_KEYS = (
    "AP_small_50_excl_bus",
    "AP_small_5095_excl_bus",
)


def _is_dair_dataset(dataset_name: str) -> bool:
    low = dataset_name.lower()
    return "dair" in low or "dairv2x" in low


def _is_uadetrac_dataset(dataset_name: str) -> bool:
    low = dataset_name.lower()
    return "uadetrac" in low or "ua-detrac" in low or "ua_detrac" in low


def small_object_diagnostic_spec(
    dataset_name: str,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return ``(excluded_categories, diagnostic_metric_keys)`` for a dataset."""
    if _is_dair_dataset(dataset_name):
        return DAIR_SMALL_EXCLUDED_CATEGORIES, DAIR_SMALL_DIAGNOSTIC_KEYS
    if _is_uadetrac_dataset(dataset_name):
        return UADETRAC_SMALL_EXCLUDED_CATEGORIES, UADETRAC_SMALL_DIAGNOSTIC_KEYS
    return (), ()
