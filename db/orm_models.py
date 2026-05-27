"""
db/orm_models.py — SQLAlchemy ORM 模型(Step 9.2.1)
================================================
设计原则:
- 字段名与 models.py 的 dataclass 严格一一对应,便于 Repository 层做双向转换。
- 直接复用 models.py 的 BatchStatus / UserRole 枚举,语义不重复定义。
- set / dict 类型字段用 JSON 列存储(SQLite 原生支持)。
- 当前阶段(Step 9.2.1)只建表,不接入业务逻辑。
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Integer, String, Float, Boolean, DateTime, Enum as SQLEnum,
    ForeignKey, JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from models import BatchStatus, UserRole


class Base(DeclarativeBase):
    """所有 ORM 模型的基类"""
    pass


# ═══════════════════════════════════════════════════════════
# User
# ═══════════════════════════════════════════════════════════
class UserORM(Base):
    __tablename__ = "users"

    user_id:       Mapped[int]    = mapped_column(Integer, primary_key=True, autoincrement=True)
    username:      Mapped[str]    = mapped_column(String(64), unique=True, nullable=False, index=True)
    password_hash: Mapped[str]    = mapped_column(String(255), nullable=False)
    full_name:     Mapped[str]    = mapped_column(String(64), nullable=False)
    role:          Mapped[UserRole] = mapped_column(SQLEnum(UserRole), nullable=False)
    phone:         Mapped[str]    = mapped_column(String(32), default="")
    is_active:     Mapped[bool]   = mapped_column(Boolean, default=True)
    created_at:    Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


# ═══════════════════════════════════════════════════════════
# Supplier
# ═══════════════════════════════════════════════════════════
class SupplierORM(Base):
    __tablename__ = "suppliers"

    supplier_id:     Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name:            Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    # Step 9.2.2.a: 归一化名称(去首尾空格、折叠中间空格、转小写),UNIQUE 保证幂等查找
    normalized_name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    contact_phone: Mapped[str]    = mapped_column(String(32), default="")
    address:       Mapped[str]    = mapped_column(String(255), default="")
    is_active:     Mapped[bool]   = mapped_column(Boolean, default=True)
    created_at:    Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    region:        Mapped[str]    = mapped_column(String(64), default="")
    notes:         Mapped[str]    = mapped_column(String(500), default="")
    updated_at:    Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


# ═══════════════════════════════════════════════════════════
# FruitCategory
# ═══════════════════════════════════════════════════════════
class FruitCategoryORM(Base):
    __tablename__ = "fruit_categories"

    category_id:  Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    name:         Mapped[str]  = mapped_column(String(32), unique=True, nullable=False)
    code:         Mapped[str]  = mapped_column(String(32), unique=True, nullable=False)
    typical_unit: Mapped[str]  = mapped_column(String(8), default="个")
    is_active:    Mapped[bool] = mapped_column(Boolean, default=True)


# ═══════════════════════════════════════════════════════════
# InboundBatch(最复杂的实体)
# ═══════════════════════════════════════════════════════════
class InboundBatchORM(Base):
    __tablename__ = "inbound_batches"

    # 基础信息(注意:batch_id 是业务主键,字符串)
    batch_id:       Mapped[str]      = mapped_column(String(64), primary_key=True)
    operator_id:    Mapped[int]      = mapped_column(Integer, ForeignKey("users.user_id"), nullable=False)
    operator_name:  Mapped[str]      = mapped_column(String(64), default="")
    inbound_date:   Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    # 货源
    supplier_id:    Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("suppliers.supplier_id"), nullable=True)
    supplier_name:  Mapped[str]  = mapped_column(String(128), default="")
    fruit_category: Mapped[str]  = mapped_column(String(32), default="")
    declared_count: Mapped[int]  = mapped_column(Integer, default=0)

    # 仓储
    warehouse:      Mapped[str]  = mapped_column(String(64), default="")
    storage_zone:   Mapped[str]  = mapped_column(String(64), default="")

    # 检测结果
    detected_total: Mapped[int]   = mapped_column(Integer, default=0)
    grade_a_count:  Mapped[int]   = mapped_column(Integer, default=0)
    grade_b_count:  Mapped[int]   = mapped_column(Integer, default=0)
    grade_c_count:  Mapped[int]   = mapped_column(Integer, default=0)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # 累积式检测(JSON 列存 list,Repository 层负责 set <-> list 转换)
    processed_files:  Mapped[list] = mapped_column(JSON, default=list)
    detection_rounds: Mapped[int]  = mapped_column(Integer, default=0)

    # 对账结论
    count_diff:     Mapped[int]   = mapped_column(Integer, default=0)
    count_diff_pct: Mapped[float] = mapped_column(Float, default=0.0)
    qualified_rate: Mapped[float] = mapped_column(Float, default=0.0)
    defect_rate:    Mapped[float] = mapped_column(Float, default=0.0)

    # 状态机
    status:        Mapped[BatchStatus] = mapped_column(SQLEnum(BatchStatus), default=BatchStatus.DRAFT)
    status_reason: Mapped[str] = mapped_column(String(255), default="")

    # 流转记录
    reviewed_by:      Mapped[Optional[int]]      = mapped_column(Integer, ForeignKey("users.user_id"), nullable=True)
    reviewed_at:      Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    reviewed_note:    Mapped[str] = mapped_column(String(500), default="")
    reviewed_by_name: Mapped[str] = mapped_column(String(64), default="")

    confirmed_by:      Mapped[Optional[int]]      = mapped_column(Integer, ForeignKey("users.user_id"), nullable=True)
    confirmed_at:      Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    confirmed_by_name: Mapped[str] = mapped_column(String(64), default="")

    rejected_by:      Mapped[Optional[int]]      = mapped_column(Integer, ForeignKey("users.user_id"), nullable=True)
    rejected_at:      Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    rejected_by_name: Mapped[str] = mapped_column(String(64), default="")
    rejection_reason: Mapped[str] = mapped_column(String(500), default="")

    # KPI 快照(JSON 列)
    review_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)

    # 备注
    notes: Mapped[str] = mapped_column(String(500), default="")


# ═══════════════════════════════════════════════════════════
# OperationLog(Step 8 审计日志,先建表)
# ═══════════════════════════════════════════════════════════
class OperationLogORM(Base):
    __tablename__ = "operation_logs"

    log_id:      Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id:     Mapped[int]      = mapped_column(Integer, ForeignKey("users.user_id"), nullable=False)
    action:      Mapped[str]      = mapped_column(String(64), nullable=False, index=True)
    target_type: Mapped[str]      = mapped_column(String(32), nullable=False)
    target_id:   Mapped[str]      = mapped_column(String(64), nullable=False, index=True)
    detail:      Mapped[str]      = mapped_column(String(2000), default="")
    created_at:  Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
