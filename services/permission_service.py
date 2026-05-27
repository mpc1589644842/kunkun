"""
services/permission_service.py — 角色权限检查

设计:简单的 RBAC(基于角色的权限控制)
  - operator: 创建批次、跑检测、提交快速入库
  - manager:  + 复核(通过/拒收)
  - admin:    + 管理供应商/品类/用户
"""
from models import User, UserRole, BatchStatus


# ═══════════════════════════════════════════════════════════════
# 批次操作权限
# ═══════════════════════════════════════════════════════════════

def can_create_batch(user: User) -> bool:
    """谁能创建入库批次"""
    return user.role in {UserRole.OPERATOR, UserRole.MANAGER, UserRole.ADMIN}


def can_run_detection(user: User) -> bool:
    """谁能触发模型检测"""
    return user.role in {UserRole.OPERATOR, UserRole.MANAGER, UserRole.ADMIN}


def can_confirm_batch(user: User, batch_status: BatchStatus) -> bool:
    """谁能在 READY_TO_CONFIRM 状态下确认入库(快速通道)"""
    if batch_status != BatchStatus.READY_TO_CONFIRM:
        return False
    return user.role in {UserRole.OPERATOR, UserRole.MANAGER, UserRole.ADMIN}


def can_review_batch(user: User, batch_status: BatchStatus) -> bool:
    """谁能复核 PENDING_REVIEW 的批次(通过或拒收)"""
    if batch_status != BatchStatus.PENDING_REVIEW:
        return False
    return user.role in {UserRole.MANAGER, UserRole.ADMIN}


def can_reject_batch(user: User, batch_status: BatchStatus) -> bool:
    """谁能拒收(包括 PENDING_REVIEW 和 READY_TO_CONFIRM)"""
    if batch_status not in {BatchStatus.PENDING_REVIEW, BatchStatus.READY_TO_CONFIRM}:
        return False
    return user.role in {UserRole.MANAGER, UserRole.ADMIN}


# ═══════════════════════════════════════════════════════════════
# 主数据管理权限(供应商/品类/用户)
# ═══════════════════════════════════════════════════════════════

def can_manage_suppliers(user: User) -> bool:
    """管理供应商档案"""
    return user.role == UserRole.ADMIN


def can_manage_categories(user: User) -> bool:
    """管理品类参考表"""
    return user.role == UserRole.ADMIN


def can_manage_users(user: User) -> bool:
    """管理用户账号"""
    return user.role == UserRole.ADMIN


def can_view_all_batches(user: User) -> bool:
    """查看所有批次(管理员/经理),其他人只能看自己的"""
    return user.role in {UserRole.MANAGER, UserRole.ADMIN}