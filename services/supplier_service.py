"""
services/supplier_service.py — 供应商档案服务(Step 9.2.2.b 重构)
================================================
重构说明:
- CRUD 函数(get_or_create / update / soft_delete / restore / rename)
  内部全部委托给 db.repositories.supplier_repo
- 老接口签名保留(包括 deprecated 的 suppliers: Dict 参数),
  以保证 app.py / pages 的现有调用不需要修改(Step 9.2 核心承诺)
- 纯计算函数(aggregate_kpi / aggregate_main_categories / classify_supplier /
  label_display / get_supplier_batches)保持不变,接收 batches 列表做无状态计算
- normalize_name 转发到 repo(顺手修复中文场景 bug)
- _matches_supplier 简化为只看 supplier_id(SQLite 时代所有批次都有 FK)
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Dict, Iterable

from models import Supplier, InboundBatch, BatchStatus
from db.repositories import supplier_repo as _repo


# ═══════════════════════════════════════════════════════════════
# 名称归一化(转发给 repo,保持向后兼容)
# ═══════════════════════════════════════════════════════════════
def normalize_name(name: str) -> str:
    """转发到 repo.normalize_name(已修复中文场景 bug)"""
    return _repo.normalize_name(name)


# ═══════════════════════════════════════════════════════════════
# CRUD(对外接口签名不变,内部走 repo)
# ═══════════════════════════════════════════════════════════════
def get_or_create_supplier(
    suppliers: Dict[str, Supplier],  # deprecated:保留参数兼容老调用,内部不使用
    name: str,
    resurrect: bool = True,
) -> Optional[Supplier]:
    """
    自动建档 + 复活(Step 7.4 hook / 9.2.2.b 切到 SQLite)
    - suppliers 参数已废弃,保留只为兼容 app.py 的现有调用;后续移除
    - resurrect 语义不变:True 时自动复活软删除档案,False 时只读匹配
    """
    return _repo.get_or_create(name, resurrect=resurrect)


def update_supplier(supplier: Supplier, **kwargs) -> Optional[Supplier]:
    """更新档案。允许:contact_phone / address / region / notes / is_active"""
    return _repo.update(supplier.supplier_id, **kwargs)


def soft_delete_supplier(supplier: Supplier) -> Optional[Supplier]:
    """软删除"""
    return _repo.soft_delete(supplier.supplier_id)


def restore_supplier(supplier: Supplier) -> Optional[Supplier]:
    """恢复软删除"""
    return _repo.restore(supplier.supplier_id)


def rename_supplier(
    suppliers: Dict[str, Supplier],  # deprecated:保留参数兼容老调用,内部不使用
    old_key: str,                     # deprecated:保留参数兼容,内部由 supplier_id 路由
    new_name: str,
    supplier_id: Optional[int] = None,  # 9.2.2.b 推荐用 supplier_id
) -> Optional[Supplier]:
    """
    重命名。优先用 supplier_id;老调用方传 old_key (normalized_name) 时回退到查找。
    """
    if supplier_id is None:
        # 老调用方式:从 normalized_key 反查 ID
        sup = _repo.find_by_normalized(old_key)
        if sup is None:
            return None
        supplier_id = sup.supplier_id
    return _repo.rename(supplier_id, new_name)


# ═══════════════════════════════════════════════════════════════
# 批次匹配 + KPI 聚合(纯函数,接收外部 batches 列表)
# ═══════════════════════════════════════════════════════════════
def _matches_supplier(batch: InboundBatch, supplier: Supplier) -> bool:
    """
    Step 9.2.2.b:简化为只看 supplier_id(SQLite 时代所有批次都通过 7.4 hook
    在创建时填了 supplier_id,不再需要 supplier_name fallback)。
    """
    return batch.supplier_id is not None and batch.supplier_id == supplier.supplier_id


def get_supplier_batches(
    supplier: Supplier,
    all_batches: Iterable[InboundBatch],
) -> List[InboundBatch]:
    """筛选出属于该供应商的所有批次"""
    return [b for b in all_batches if _matches_supplier(b, supplier)]


def aggregate_kpi(
    supplier: Supplier,
    all_batches: Iterable[InboundBatch],
) -> dict:
    """从 all_batches 动态聚合供应商的 KPI"""
    batches = get_supplier_batches(supplier, all_batches)

    total = len(batches)
    confirmed = sum(1 for b in batches if b.status == BatchStatus.CONFIRMED)
    rejected = sum(1 for b in batches if b.status == BatchStatus.REJECTED)
    pending_statuses = {
        BatchStatus.DRAFT,
        BatchStatus.READY_TO_CONFIRM,
        BatchStatus.PENDING_REVIEW,
    }
    pending = sum(1 for b in batches if b.status in pending_statuses)

    terminal_total = confirmed + rejected
    confirm_rate = (confirmed / terminal_total) if terminal_total > 0 else 0.0
    reject_rate = (rejected / terminal_total) if terminal_total > 0 else 0.0

    detected_batches = [b for b in batches if b.detected_total > 0]
    if detected_batches:
        avg_qualified = sum(b.qualified_rate for b in detected_batches) / len(detected_batches)
        avg_defect = sum(b.defect_rate for b in detected_batches) / len(detected_batches)
        avg_count_diff = sum(b.count_diff_pct for b in detected_batches) / len(detected_batches)
    else:
        avg_qualified = 0.0
        avg_defect = 0.0
        avg_count_diff = 0.0

    last_delivery = None
    last_batch_id = None
    for b in batches:
        t = b.confirmed_at or b.rejected_at or b.inbound_date
        if t is None:
            continue
        if last_delivery is None or t > last_delivery:
            last_delivery = t
            last_batch_id = b.batch_id

    return {
        "total_batches":  total,
        "confirmed":      confirmed,
        "rejected":       rejected,
        "pending":        pending,
        "confirm_rate":   confirm_rate,
        "reject_rate":    reject_rate,
        "avg_qualified":  avg_qualified,
        "avg_defect":     avg_defect,
        "avg_count_diff": avg_count_diff,
        "last_delivery":  last_delivery,
        "last_batch_id":  last_batch_id,
    }


def aggregate_main_categories(
    supplier: Supplier,
    all_batches: Iterable[InboundBatch],
    top_n: int = 3,
) -> List[str]:
    """从历史批次推断主营品类(出现次数最多的 top_n)"""
    from collections import Counter
    batches = get_supplier_batches(supplier, all_batches)
    cats = [b.fruit_category for b in batches if b.fruit_category]
    if not cats:
        return []
    return [c for c, _ in Counter(cats).most_common(top_n)]


# ═══════════════════════════════════════════════════════════════
# 分类标签
# ═══════════════════════════════════════════════════════════════
LABEL_PREMIUM   = "premium"    # 🏆 优质
LABEL_WARNING   = "warning"    # ⚠️ 警示
LABEL_NEW       = "new"        # ⚪ 新供应商
LABEL_NORMAL    = "normal"     # 普通(无徽章)


def classify_supplier(kpi: dict) -> str:
    """根据 KPI 给供应商打分类标签"""
    total = kpi["total_batches"]
    if total < 3:
        return LABEL_NEW

    confirm_rate = kpi["confirm_rate"]
    avg_qualified = kpi["avg_qualified"]
    reject_rate = kpi["reject_rate"]
    avg_defect = kpi["avg_defect"]

    if confirm_rate >= 0.90 and avg_qualified >= 0.95:
        return LABEL_PREMIUM

    if reject_rate >= 0.30 or avg_defect >= 0.10:
        return LABEL_WARNING

    return LABEL_NORMAL


def label_display(label: str):
    """根据标签返回 (徽章文字, 颜色, emoji)"""
    if label == LABEL_PREMIUM:
        return ("优质", "#22C55E", "🏆")
    if label == LABEL_WARNING:
        return ("警示", "#EF4444", "⚠️")
    if label == LABEL_NEW:
        return ("新供应商", "#94A3B8", "⚪")
    return ("普通", "#A07040", "")
