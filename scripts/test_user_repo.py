"""
scripts/test_user_repo.py — UserRepository 单元测试 (Step 9.2.2.d)
"""
import sys
import tempfile
from pathlib import Path

_TMP_DIR = tempfile.mkdtemp(prefix="user_repo_test_")
_TMP_DB = Path(_TMP_DIR) / "test_fruit.db"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import db.session as _sess_mod
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_sess_mod.DB_PATH = _TMP_DB
_sess_mod.DB_URL = f"sqlite:///{_TMP_DB}"
_sess_mod.engine = create_engine(
    _sess_mod.DB_URL, echo=False, future=True,
    connect_args={"check_same_thread": False},
)
_sess_mod.SessionLocal = sessionmaker(
    bind=_sess_mod.engine,
    autoflush=False, autocommit=False,
    expire_on_commit=False, future=True,
)

from db.orm_models import Base
Base.metadata.create_all(_sess_mod.engine)

from db.repositories import user_repo as repo
from models import UserRole


_passed = 0
_failed = 0
def assert_eq(a, e, msg=""):
    global _passed, _failed
    if a == e:
        _passed += 1; print(f"  ✅ {msg}")
    else:
        _failed += 1
        print(f"  ❌ {msg}\n     期望:{e!r}\n     实际:{a!r}")

def assert_truthy(a, msg=""):
    global _passed, _failed
    if a:
        _passed += 1; print(f"  ✅ {msg}")
    else:
        _failed += 1; print(f"  ❌ {msg}")

def assert_none(a, msg=""):
    global _passed, _failed
    if a is None:
        _passed += 1; print(f"  ✅ {msg}")
    else:
        _failed += 1; print(f"  ❌ {msg}  (实际:{a!r})")


print("=" * 70)
print("  UserRepository 单元测试")
print("=" * 70)

# ─── 1. 密码哈希 ───
print("\n[1] hash_password + verify_password")
h1 = repo.hash_password("op123")
assert_truthy(h1.startswith("$2b$") or h1.startswith("$2a$"), "bcrypt 哈希格式")
assert_eq(repo.verify_password("op123", h1), True, "正确密码验证通过")
assert_eq(repo.verify_password("wrong", h1), False, "错误密码验证失败")
assert_eq(repo.verify_password("", h1), False, "空密码验证失败")
assert_eq(repo.verify_password("op123", ""), False, "空哈希验证失败")
# 同一密码两次哈希结果不同(salt 不同),但都能验证通过
h2 = repo.hash_password("op123")
assert_truthy(h1 != h2, "相同密码两次哈希(因 salt)不同")
assert_eq(repo.verify_password("op123", h2), True, "第二次哈希也能验证")

# ─── 2. create ───
print("\n[2] create")
u1 = repo.create("operator", "op123", "操作员·小张",
                 UserRole.OPERATOR, phone="138-0000-0001", user_id=1)
assert_truthy(u1, "create 返回用户")
assert_eq(u1.username, "operator", "username 保存")
assert_eq(u1.role, UserRole.OPERATOR, "role 保存")

# 用户名冲突
dup = repo.create("operator", "different", "Dup",
                  UserRole.MANAGER, user_id=999)
assert_none(dup, "重复 username 返回 None")

# 缺失字段
assert_none(repo.create("", "pw", "X", UserRole.ADMIN), "空 username 返回 None")
assert_none(repo.create("u", "", "X", UserRole.ADMIN), "空 password 返回 None")

# ─── 3. find_by_username / find_by_id ───
print("\n[3] find_by_username / find_by_id")
found = repo.find_by_username("operator")
assert_truthy(found, "find_by_username 命中")
assert_eq(found.user_id, 1, "user_id 一致")

found2 = repo.find_by_username("OPERATOR")
assert_truthy(found2, "find_by_username 不区分大小写")

assert_none(repo.find_by_username("not_exist"), "不存在用户名返回 None")
assert_none(repo.find_by_username(""), "空字符返回 None")

assert_truthy(repo.find_by_id(1), "find_by_id 命中")
assert_none(repo.find_by_id(999), "find_by_id 不存在返回 None")

# ─── 4. authenticate ───
print("\n[4] authenticate")
auth_ok = repo.authenticate("operator", "op123")
assert_truthy(auth_ok, "正确账密验证成功")
assert_eq(auth_ok.password_hash, "[hidden]", "返回前抹掉密码哈希")
assert_eq(auth_ok.username, "operator", "username 正确")

assert_none(repo.authenticate("operator", "wrong"), "错密码返回 None")
assert_none(repo.authenticate("not_exist", "op123"), "不存在用户返回 None")
assert_none(repo.authenticate("", ""), "空账密返回 None")

# 大小写不敏感
assert_truthy(repo.authenticate("OPERATOR", "op123"), "用户名大小写不敏感")

# ─── 5. list_all ───
print("\n[5] list_all")
repo.create("manager", "mgr123", "经理·老李", UserRole.MANAGER, user_id=2)
repo.create("admin",   "admin123", "系统管理员", UserRole.ADMIN, user_id=3)
all_users = repo.list_all()
assert_eq(len(all_users), 3, "活跃用户共 3 个")

# ─── 6. update_password ───
print("\n[6] update_password")
updated = repo.update_password(1, "newpw")
assert_truthy(updated, "update_password 返回用户")
assert_eq(repo.authenticate("operator", "newpw"), updated.__class__(
    user_id=1, username="operator", password_hash="[hidden]",
    full_name="操作员·小张", role=UserRole.OPERATOR,
    phone="138-0000-0001", is_active=True,
    created_at=updated.created_at,
) if False else None, "占位") if False else assert_truthy(
    repo.authenticate("operator", "newpw"), "新密码可以验证"
)
assert_none(repo.authenticate("operator", "op123"), "旧密码不再有效")

assert_none(repo.update_password(999, "x"), "不存在用户返回 None")
assert_none(repo.update_password(1, ""), "空密码返回 None")

# ─── 7. soft_delete ───
print("\n[7] soft_delete")
deleted = repo.soft_delete(1)
assert_truthy(deleted, "soft_delete 返回用户")
assert_eq(deleted.is_active, False, "is_active 已置 False")
assert_none(repo.authenticate("operator", "newpw"), "软删除后 authenticate 失败")
assert_eq(len(repo.list_all(include_inactive=False)), 2, "默认列表少 1")
assert_eq(len(repo.list_all(include_inactive=True)), 3, "include_inactive 仍 3")

assert_none(repo.soft_delete(999), "soft_delete 不存在返回 None")

print()
print("=" * 70)
total = _passed + _failed
print(f"通过 {_passed}/{total},失败 {_failed}")
print("=" * 70)

import shutil
try:
    shutil.rmtree(_TMP_DIR)
except Exception:
    pass

sys.exit(0 if _failed == 0 else 1)
