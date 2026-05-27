"""
db/repositories/batch_repo.py — BatchRepository (Step 9.2.2.c.1)
================================================
设计原则:
- 短 session:每个方法内部 with SessionLocal()
- ORM ↔ dataclass 双向转换,UI / services 层只见 dataclass(models.InboundBatch)
- save(dc):upsert 语义 — 不存在则 INSERT,存在则全字段 UPDATE
- apply_changes(batch_id, **fields):局部更新,供 batch_service 状态机函数用
- processed_files (set) <-> JSON list 透明处理
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Iterable

from sqlalchemy import select, delete as sa_delete

from db.session import SessionLocal
from db.orm_models import InboundBatchORM
from models import InboundBatch, BatchStatus


# ═══════════════════════════════════════════════════════════
# ORM ↔ dataclass 双向转换
# ═══════════════════════════════════════════════════════════
# 所有字段名(InboundBatch dataclass 与 InboundBatchORM 列一一对应)
# 用元组列出,转换时直接遍历 — 加字段时改一处即可
_PASS_THROUGH_FIELDS = (
    "batch_id", "operator_id", "operator_name", "inbound_date",
    "supplier_id", "supplier_name", "fruit_category", "declared_count",
    "warehouse", "storage_zone",
    "detected_total", "grade_a_count", "grade_b_count", "grade_c_count",
    "avg_confidence",
    "detection_rounds",
    "count_diff", "count_diff_pct", "qualified_rate", "defect_rate",
    "status", "status_reason",
    "reviewed_by", "reviewed_at", "reviewed_note", "reviewed_by_name",
    "confirmed_by", "confirmed_at", "confirmed_by_name",
    "rejected_by", "rejected_at", "rejected_by_name", "rejection_reason",
    "review_snapshot",
    "notes",
)
# processed_files 单独处理(set <-> list)


def _orm_to_dc(orm: InboundBatchORM) -> InboundBatch:
    """ORM 行 → dataclass"""
    kwargs = {f: getattr(orm, f) for f in _PASS_THROUGH_FIELDS}
    # processed_files: list -> set
    pf = orm.processed_files or []
    kwargs["processed_files"] = set(pf)
    return InboundBatch(**kwargs)


def _dc_to_orm_values(dc: InboundBatch) -> dict:
    """dataclass → ORM 字段字典(用于 INSERT / bulk update)"""
    values = {f: getattr(dc, f) for f in _PASS_THROUGH_FIELDS}
    # processed_files: set -> list
    values["processed_files"] = list(dc.processed_files) if dc.processed_files else []
    return values


# ═══════════════════════════════════════════════════════════
# 单条查找
# ═══════════════════════════════════════════════════════════
def find_by_id(batch_id: str) -> Optional[InboundBatch]:
    """按 batch_id 查找"""
    if not batch_id:
        return None
    with SessionLocal() as sess:
        orm = sess.get(InboundBatchORM, batch_id)
        return _orm_to_dc(orm) if orm else None


def exists(batch_id: str) -> bool:
    """是否存在该 batch_id"""
    if not batch_id:
        return False
    with SessionLocal() as sess:
        orm = sess.get(InboundBatchORM, batch_id)
        return orm is not None


# ═══════════════════════════════════════════════════════════
# 列表查询
# ═══════════════════════════════════════════════════════════
def list_all(
    include_draft: bool = True,
    status_in: Optional[Iterable[BatchStatus]] = None,
    supplier_id: Optional[int] = None,
) -> List[InboundBatch]:
    """
    列出批次。
    - include_draft=False:排除 DRAFT 状态(默认包含)
    - status_in:仅返回指定状态集合
    - supplier_id:筛选某供应商
    """
    with SessionLocal() as sess:
        stmt = select(InboundBatchORM)
        if not include_draft:
            stmt = stmt.where(InboundBatchORM.status != BatchStatus.DRAFT)
        if status_in is not None:
            statuses = list(status_in)
            if statuses:
                stmt = stmt.where(InboundBatchORM.status.in_(statuses))
        if supplier_id is not None:
            stmt = stmt.where(InboundBatchORM.supplier_id == supplier_id)
        rows = sess.execute(stmt).scalars().all()
        return [_orm_to_dc(r) for r in rows]


# ═══════════════════════════════════════════════════════════
# 写入
# ═══════════════════════════════════════════════════════════
def save(dc: InboundBatch) -> InboundBatch:
    """
    Upsert 语义:
    - batch_id 不存在 → INSERT
    - batch_id 已存在 → 全字段 UPDATE(覆盖式)
    返回最新 dataclass(从 DB 重读)
    """
    if not dc.batch_id:
        raise ValueError("save(): batch_id 不能为空")

    values = _dc_to_orm_values(dc)
    with SessionLocal() as sess:
        orm = sess.get(InboundBatchORM, dc.batch_id)
        if orm is None:
            orm = InboundBatchORM(**values)
            sess.add(orm)
        else:
            for k, v in values.items():
                if k == "batch_id":
                    continue  # 主键不改
                setattr(orm, k, v)
        sess.commit()
        sess.refresh(orm)
        return _orm_to_dc(orm)


def apply_changes(_batch_id: str, **fields) -> Optional[InboundBatch]:
    """
    局部更新(供 batch_service 状态机函数用)。
    位置参名 _batch_id 避免与 **fields 里偶尔出现的 batch_id 键冲突。
    自动处理 processed_files (set -> list)。
    返回更新后的 dataclass;batch 不存在则 None。
    """
    if not _batch_id:
        return None
    if not fields:
        return find_by_id(_batch_id)

    # set -> list 转换
    if "processed_files" in fields and isinstance(fields["processed_files"], set):
        fields["processed_files"] = list(fields["processed_files"])

    with SessionLocal() as sess:
        orm = sess.get(InboundBatchORM, _batch_id)
        if orm is None:
            return None
        for k, v in fields.items():
            if k == "batch_id":
                continue  # 防御性忽略,防止意外覆盖主键
            if hasattr(orm, k):
                setattr(orm, k, v)
        sess.commit()
        sess.refresh(orm)
        return _orm_to_dc(orm)


def delete(batch_id: str) -> bool:
    """物理删除(批次一般不会被删,留接口给测试)。返回是否真的删除"""
    if not batch_id:
        return False
    with SessionLocal() as sess:
        orm = sess.get(InboundBatchORM, batch_id)
        if orm is None:
            return False
        sess.delete(orm)
        sess.commit()
        return True


def delete_all() -> int:
    """清空所有批次。用于测试和数据重置。返回删除条数"""
    with SessionLocal() as sess:
        result = sess.execute(sa_delete(InboundBatchORM))
        sess.commit()
        return result.rowcount or 0


def count(supplier_id: Optional[int] = None) -> int:
    """统计批次总数,可按 supplier_id 过滤"""
    with SessionLocal() as sess:
        stmt = select(InboundBatchORM)
        if supplier_id is not None:
            stmt = stmt.where(InboundBatchORM.supplier_id == supplier_id)
        return len(sess.execute(stmt).scalars().all())
