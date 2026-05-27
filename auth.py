"""
auth.py — 用户认证(Step 9.2.2.d 切到 DB + bcrypt)
================================================
- authenticate:用 user_repo.authenticate(走 DB + bcrypt)
- get_demo_accounts:展示演示账号信息(密码从 DB 反推不可行,只展示已知组合)
"""
from typing import Optional, List

from models import User
from db.repositories import user_repo as _ur


# 演示账号清单(仅用于登录页提示,不用于实际认证;实际认证走 DB)
_DEMO_ACCOUNTS_HINT = [
    {"username": "operator", "password": "op123",    "role_label": "操作员"},
    {"username": "manager",  "password": "mgr123",   "role_label": "经理"},
    {"username": "admin",    "password": "admin123", "role_label": "管理员"},
]


def authenticate(username: str, password: str) -> Optional[User]:
    """验证用户名密码。委托给 user_repo,内部走 bcrypt。"""
    return _ur.authenticate(username, password)


def get_demo_accounts() -> List[dict]:
    """返回演示账号列表(登录页提示用)。"""
    out = []
    for rec in _DEMO_ACCOUNTS_HINT:
        u = _ur.find_by_username(rec["username"])
        if u is None:
            continue
        out.append({
            "username": rec["username"],
            "password": rec["password"],
            "role": u.role.display_name,
        })
    return out
