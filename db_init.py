"""
db_init.py — 初始化 SQLite 数据库(Step 9.2.1 + 9.2.2.d)
================================================
用法:
    python db_init.py             # 首次:建表 + 注入演示用户(bcrypt) + 4 个水果品类
    python db_init.py --reset     # 危险:删库重建(用于调试)
"""
from __future__ import annotations
import sys
from datetime import datetime

from db.session import engine, SessionLocal, init_db, DB_PATH
from db.orm_models import UserORM, FruitCategoryORM
from db.repositories import user_repo as _ur
from models import UserRole


# 演示账号(明文密码仅在初始化时使用,落库前 bcrypt 哈希)
_DEMO_USERS = [
    {"user_id": 1, "username": "operator", "password": "op123",
     "full_name": "操作员·小张", "role": UserRole.OPERATOR, "phone": "138-0000-0001"},
    {"user_id": 2, "username": "manager",  "password": "mgr123",
     "full_name": "经理·老李",  "role": UserRole.MANAGER,  "phone": "138-0000-0002"},
    {"user_id": 3, "username": "admin",    "password": "admin123",
     "full_name": "系统管理员",  "role": UserRole.ADMIN,    "phone": "138-0000-0003"},
]


def reset_db() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"[已删除] {DB_PATH}")
    init_db()
    print(f"[已重建] {DB_PATH}")


def seed_users() -> int:
    """注入演示用户。密码经 bcrypt 哈希。返回插入条数"""
    inserted = 0
    for rec in _DEMO_USERS:
        existing = _ur.find_by_username(rec["username"])
        if existing is not None:
            continue
        u = _ur.create(
            username  = rec["username"],
            password  = rec["password"],
            full_name = rec["full_name"],
            role      = rec["role"],
            phone     = rec["phone"],
            user_id   = rec["user_id"],
        )
        if u is not None:
            inserted += 1
    return inserted


def seed_categories() -> int:
    cats = [
        {"name": "苹果", "code": "apple",  "typical_unit": "个"},
        {"name": "香蕉", "code": "banana", "typical_unit": "串"},
        {"name": "葡萄", "code": "grape",  "typical_unit": "串"},
        {"name": "橙子", "code": "orange", "typical_unit": "个"},
    ]
    inserted = 0
    with SessionLocal() as sess:
        for c in cats:
            existing = sess.query(FruitCategoryORM).filter(
                FruitCategoryORM.code == c["code"]
            ).first()
            if existing:
                continue
            sess.add(FruitCategoryORM(**c, is_active=True))
            inserted += 1
        sess.commit()
    return inserted


def main():
    do_reset = "--reset" in sys.argv

    print("=" * 60)
    print(f"  fruit_app.db 初始化")
    print(f"  路径: {DB_PATH}")
    print("=" * 60)

    if do_reset:
        reset_db()
    else:
        init_db()
        print("[建表完成] (幂等)")

    n_users = seed_users()
    print(f"[注入用户] {n_users} 条 (bcrypt 哈希)")

    n_cats = seed_categories()
    print(f"[注入品类] {n_cats} 条")

    with SessionLocal() as sess:
        users = sess.query(UserORM).all()
        cats = sess.query(FruitCategoryORM).all()
        print()
        print("当前数据库状态:")
        print(f"  用户:  {len(users)} 条")
        for u in users:
            print(f"    - [{u.user_id}] {u.username}  ·  {u.full_name}  ·  {u.role.value}")
        print(f"  品类:  {len(cats)} 条")
        for c in cats:
            print(f"    - [{c.category_id}] {c.name} ({c.code})")
    print()
    print("✅ 初始化完成。Step 9.2.2.d done.")


if __name__ == "__main__":
    main()
