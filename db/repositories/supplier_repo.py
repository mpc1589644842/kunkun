"""
db/repositories/supplier_repo.py — SupplierRepository (Step 9.2.2.a)
================================================
设计原则:
- 短 session:每个方法内部 with SessionLocal() as sess,自动回滚 / 关闭
- 接口签名与 services/supplier_service.py 的核心函数对齐,便于 9.2.2.b 切换
- ORM ↔ dataclass 双向转换:UI / services 层永远拿到 dataclass(models.Supplier)
- 归一化匹配:基于 normalized_name UNIQUE 字段,O(log n) 查找
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List

from sqlalchemy import select

from db.session import SessionLocal
from db.orm_models import SupplierORM
from models import Supplier


# ═══════════════════════════════════════════════════════════
# 归一化
# ═══════════════════════════════════════════════════════════
def normalize_name(name: str) -> str:
    """
    归一化名称,用于幂等匹配。
    - 中文之间不应有空格,英文情况下"ABC Apple Co"→"abcappleco"也无歧义
    - 因此统一:去掉所有空白字符 + 转小写
    """
    if not name:
        return ""
    return "".join(name.strip().split()).lower()


# ═══════════════════════════════════════════════════════════
# ORM ↔ dataclass 双向转换
# ═══════════════════════════════════════════════════════════
def _orm_to_dc(orm: SupplierORM) -> Supplier:
    """ORM 行 → dataclass(UI 层只见 dataclass)"""
    return Supplier(
        supplier_id   = orm.supplier_id,
        name          = orm.name,
        contact_phone = orm.contact_phone,
        address       = orm.address,
        is_active     = orm.is_active,
        created_at    = orm.created_at,
        region        = orm.region,
        notes         = orm.notes,
        updated_at    = orm.updated_at,
    )


def _apply_dc_to_orm(orm: SupplierORM, dc: Supplier) -> None:
    """dataclass 字段值 → ORM 行(用于 update;不动主键和 normalized_name)"""
    orm.name          = dc.name
    orm.contact_phone = dc.contact_phone
    orm.address       = dc.address
    orm.is_active     = dc.is_active
    orm.region        = dc.region
    orm.notes         = dc.notes
    orm.updated_at    = datetime.now()
    # 注意:created_at 不更新;normalized_name 由 rename 专门处理


# ═══════════════════════════════════════════════════════════
# CRUD
# ═══════════════════════════════════════════════════════════
def find_by_id(supplier_id: int) -> Optional[Supplier]:
    """按 supplier_id 查找(路由场景必备)"""
    with SessionLocal() as sess:
        orm = sess.get(SupplierORM, supplier_id)
        return _orm_to_dc(orm) if orm else None


def find_by_normalized(key: str) -> Optional[Supplier]:
    """按归一化名查找(命中即返回 dataclass,未命中返回 None)"""
    if not key:
        return None
    with SessionLocal() as sess:
        stmt = select(SupplierORM).where(SupplierORM.normalized_name == key)
        orm = sess.execute(stmt).scalar_one_or_none()
        return _orm_to_dc(orm) if orm else None


def get_or_create(name: str, resurrect: bool = True) -> Optional[Supplier]:
    """
    幂等建档(对齐 services.supplier_service.get_or_create_supplier 语义):
    - 名字为空 → None
    - 命中已有:若 is_active=False 且 resurrect=True,则复活
    - 未命中:新建空白档案(region/notes 为空)
    """
    if not name or not name.strip():
        return None

    key = normalize_name(name)
    with SessionLocal() as sess:
        stmt = select(SupplierORM).where(SupplierORM.normalized_name == key)
        orm = sess.execute(stmt).scalar_one_or_none()

        if orm is not None:
            if resurrect and not orm.is_active:
                orm.is_active = True
                orm.updated_at = datetime.now()
                sess.commit()
            return _orm_to_dc(orm)

        # 新建
        new_orm = SupplierORM(
            name            = name.strip(),
            normalized_name = key,
            is_active       = True,
            created_at      = datetime.now(),
            updated_at      = datetime.now(),
        )
        sess.add(new_orm)
        sess.commit()
        sess.refresh(new_orm)
        return _orm_to_dc(new_orm)


def list_all(include_inactive: bool = False) -> List[Supplier]:
    """列出所有供应商(默认只活跃,详情页传 True 看全部)"""
    with SessionLocal() as sess:
        stmt = select(SupplierORM)
        if not include_inactive:
            stmt = stmt.where(SupplierORM.is_active == True)
        rows = sess.execute(stmt).scalars().all()
        return [_orm_to_dc(r) for r in rows]


def update(supplier_id: int, **fields) -> Optional[Supplier]:
    """
    更新供应商档案。只允许改 contact_phone / address / region / notes / is_active
    (name 由 rename 专门处理,因为要同时更新 normalized_name 并检查冲突)
    """
    allowed = {"contact_phone", "address", "region", "notes", "is_active"}
    with SessionLocal() as sess:
        orm = sess.get(SupplierORM, supplier_id)
        if orm is None:
            return None
        for k, v in fields.items():
            if k in allowed:
                setattr(orm, k, v)
        orm.updated_at = datetime.now()
        sess.commit()
        sess.refresh(orm)
        return _orm_to_dc(orm)


def soft_delete(supplier_id: int) -> Optional[Supplier]:
    """软删除:is_active = False"""
    with SessionLocal() as sess:
        orm = sess.get(SupplierORM, supplier_id)
        if orm is None:
            return None
        orm.is_active = False
        orm.updated_at = datetime.now()
        sess.commit()
        sess.refresh(orm)
        return _orm_to_dc(orm)


def restore(supplier_id: int) -> Optional[Supplier]:
    """恢复软删除"""
    with SessionLocal() as sess:
        orm = sess.get(SupplierORM, supplier_id)
        if orm is None:
            return None
        orm.is_active = True
        orm.updated_at = datetime.now()
        sess.commit()
        sess.refresh(orm)
        return _orm_to_dc(orm)


def rename(supplier_id: int, new_name: str) -> Optional[Supplier]:
    """
    重命名(同时更新 name + normalized_name)。
    若新名归一化后与其他记录冲突,返回 None(调用方自己处理)。
    """
    new_name = new_name.strip()
    if not new_name:
        return None
    new_key = normalize_name(new_name)
    with SessionLocal() as sess:
        orm = sess.get(SupplierORM, supplier_id)
        if orm is None:
            return None
        # 检查冲突(自己除外)
        stmt = select(SupplierORM).where(
            SupplierORM.normalized_name == new_key,
            SupplierORM.supplier_id != supplier_id,
        )
        conflict = sess.execute(stmt).scalar_one_or_none()
        if conflict is not None:
            return None
        orm.name = new_name
        orm.normalized_name = new_key
        orm.updated_at = datetime.now()
        sess.commit()
        sess.refresh(orm)
        return _orm_to_dc(orm)
