"""
db/repositories/user_repo.py — UserRepository (Step 9.2.2.d)
================================================
- bcrypt 密码哈希(hash_password + verify_password)
- find_by_username / find_by_id
- 验证密码(verify):成功返回 User dataclass,失败返回 None
- list_all / create / update_password / soft_delete
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List

import bcrypt
from sqlalchemy import select

from db.session import SessionLocal
from db.orm_models import UserORM
from models import User, UserRole


# ═══════════════════════════════════════════════════════════
# 密码哈希
# ═══════════════════════════════════════════════════════════
def hash_password(plain: str) -> str:
    """用 bcrypt 把明文密码哈希成字符串"""
    if not plain:
        raise ValueError("密码不能为空")
    hashed = bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """验证明文密码是否匹配 bcrypt 哈希"""
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ═══════════════════════════════════════════════════════════
# ORM ↔ dataclass
# ═══════════════════════════════════════════════════════════
def _orm_to_dc(orm: UserORM) -> User:
    return User(
        user_id       = orm.user_id,
        username      = orm.username,
        password_hash = orm.password_hash,
        full_name     = orm.full_name,
        role          = orm.role,
        phone         = orm.phone,
        is_active     = orm.is_active,
        created_at    = orm.created_at,
    )


# ═══════════════════════════════════════════════════════════
# 查询
# ═══════════════════════════════════════════════════════════
def find_by_username(username: str) -> Optional[User]:
    if not username:
        return None
    key = username.strip().lower()
    with SessionLocal() as sess:
        stmt = select(UserORM).where(UserORM.username == key)
        orm = sess.execute(stmt).scalar_one_or_none()
        return _orm_to_dc(orm) if orm else None


def find_by_id(user_id: int) -> Optional[User]:
    with SessionLocal() as sess:
        orm = sess.get(UserORM, user_id)
        return _orm_to_dc(orm) if orm else None


def list_all(include_inactive: bool = False) -> List[User]:
    with SessionLocal() as sess:
        stmt = select(UserORM)
        if not include_inactive:
            stmt = stmt.where(UserORM.is_active == True)
        rows = sess.execute(stmt).scalars().all()
        return [_orm_to_dc(r) for r in rows]


# ═══════════════════════════════════════════════════════════
# 认证
# ═══════════════════════════════════════════════════════════
def authenticate(username: str, password: str) -> Optional[User]:
    """验证用户名密码。成功返回 User,失败返回 None。"""
    user = find_by_username(username)
    if user is None or not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    # 出去前抹掉哈希,避免被泄露到 UI 层
    user.password_hash = "[hidden]"
    return user


# ═══════════════════════════════════════════════════════════
# 写入
# ═══════════════════════════════════════════════════════════
def create(username: str, password: str, full_name: str,
           role: UserRole, phone: str = "",
           user_id: Optional[int] = None) -> Optional[User]:
    """创建用户。用户名冲突返回 None。"""
    if not username or not password or not full_name:
        return None
    key = username.strip().lower()
    if find_by_username(key) is not None:
        return None
    with SessionLocal() as sess:
        orm_kwargs = dict(
            username      = key,
            password_hash = hash_password(password),
            full_name     = full_name,
            role          = role,
            phone         = phone,
            is_active     = True,
            created_at    = datetime.now(),
        )
        if user_id is not None:
            orm_kwargs["user_id"] = user_id
        orm = UserORM(**orm_kwargs)
        sess.add(orm)
        sess.commit()
        sess.refresh(orm)
        return _orm_to_dc(orm)


def update_password(user_id: int, new_password: str) -> Optional[User]:
    if not new_password:
        return None
    with SessionLocal() as sess:
        orm = sess.get(UserORM, user_id)
        if orm is None:
            return None
        orm.password_hash = hash_password(new_password)
        sess.commit()
        sess.refresh(orm)
        return _orm_to_dc(orm)


def soft_delete(user_id: int) -> Optional[User]:
    with SessionLocal() as sess:
        orm = sess.get(UserORM, user_id)
        if orm is None:
            return None
        orm.is_active = False
        sess.commit()
        sess.refresh(orm)
        return _orm_to_dc(orm)
