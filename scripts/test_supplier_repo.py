"""
scripts/test_supplier_repo.py — SupplierRepository 单元测试(Step 9.2.2.a)
================================================
独立运行,不依赖 Streamlit。用临时数据库文件避免污染 fruit_app.db。
"""
import os
import sys
import tempfile
from pathlib import Path

# 临时数据库:测试开始前指定一个临时文件
_TMP_DIR = tempfile.mkdtemp(prefix="supplier_repo_test_")
_TMP_DB = Path(_TMP_DIR) / "test_fruit.db"

# 项目根加入 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Monkey-patch DB_PATH 之前 import db.session
import db.session as _sess_mod
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 用临时库替换默认引擎
_sess_mod.DB_PATH = _TMP_DB
_sess_mod.DB_URL = f"sqlite:///{_TMP_DB}"
_sess_mod.engine = create_engine(
    _sess_mod.DB_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},
)
_sess_mod.SessionLocal = sessionmaker(
    bind=_sess_mod.engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)

# 建表
from db.orm_models import Base
Base.metadata.create_all(_sess_mod.engine)

# 现在可以 import repository 了
from db.repositories import supplier_repo as repo


# ─── 测试工具 ───
_passed = 0
_failed = 0

def assert_eq(actual, expected, msg=""):
    global _passed, _failed
    if actual == expected:
        _passed += 1
        print(f"  ✅ {msg}")
    else:
        _failed += 1
        print(f"  ❌ {msg}")
        print(f"     期望: {expected!r}")
        print(f"     实际: {actual!r}")

def assert_truthy(actual, msg=""):
    global _passed, _failed
    if actual:
        _passed += 1
        print(f"  ✅ {msg}")
    else:
        _failed += 1
        print(f"  ❌ {msg}  (期望 truthy,实际:{actual!r})")

def assert_none(actual, msg=""):
    global _passed, _failed
    if actual is None:
        _passed += 1
        print(f"  ✅ {msg}")
    else:
        _failed += 1
        print(f"  ❌ {msg}  (期望 None,实际:{actual!r})")


# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("  SupplierRepository 单元测试")
print("=" * 70)

# ─── 1. 归一化 ───
print("\n[1] normalize_name")
assert_eq(repo.normalize_name("  山东烟台果园  "), "山东烟台果园", "去首尾空格")
assert_eq(repo.normalize_name("ABC  Apple   Co"), "abcappleco", "英文+多空格完全去除")
assert_eq(repo.normalize_name("烟台 果园"), "烟台果园", "中文+空格完全去除(关键)")
assert_eq(repo.normalize_name(""), "", "空字符串")
assert_eq(repo.normalize_name("   "), "", "纯空格")

# ─── 2. get_or_create:新建 ───
print("\n[2] get_or_create 新建")
s1 = repo.get_or_create("烟台果园")
assert_truthy(s1, "新建返回非 None")
assert_eq(s1.name, "烟台果园", "名字保留原样")
assert_eq(s1.is_active, True, "默认活跃")
assert_truthy(s1.supplier_id, "有 supplier_id")

# ─── 3. get_or_create:命中已有(归一化匹配)───
print("\n[3] get_or_create 命中(归一化匹配)")
s2 = repo.get_or_create("  烟台果园  ")
assert_eq(s2.supplier_id, s1.supplier_id, "同归一化 key,返回同一条")

s3 = repo.get_or_create("烟台 果园")  # 中间多空格
assert_eq(s3.supplier_id, s1.supplier_id, "中间多空格仍归一到同一条")

# ─── 4. 空名 ───
print("\n[4] get_or_create 空名")
assert_none(repo.get_or_create(""), "空字符串返回 None")
assert_none(repo.get_or_create("   "), "纯空格返回 None")

# ─── 5. find_by_normalized ───
print("\n[5] find_by_normalized")
found = repo.find_by_normalized("烟台果园")
assert_truthy(found, "find 命中")
assert_eq(found.supplier_id, s1.supplier_id, "ID 一致")
assert_none(repo.find_by_normalized("不存在的"), "未命中返回 None")

# ─── 5b. find_by_id ───
print("\n[5b] find_by_id")
found_id = repo.find_by_id(s1.supplier_id)
assert_truthy(found_id, "find_by_id 命中")
assert_eq(found_id.name, s1.name, "find_by_id 返回的对象 name 一致")
assert_none(repo.find_by_id(999999), "find_by_id 不存在 ID 返回 None")

# ─── 6. update ───
print("\n[6] update 字段")
updated = repo.update(s1.supplier_id, region="山东", notes="优质合作伙伴", contact_phone="13800000000")
assert_truthy(updated, "update 返回对象")
assert_eq(updated.region, "山东", "region 已更新")
assert_eq(updated.notes, "优质合作伙伴", "notes 已更新")
assert_eq(updated.contact_phone, "13800000000", "phone 已更新")

# update 不允许的字段应被忽略(supplier_id / name / created_at 都不在白名单)
ignored = repo.update(s1.supplier_id, name="试图改名", created_at="hack")
assert_eq(ignored.supplier_id, s1.supplier_id, "supplier_id 不可改")
assert_eq(ignored.name, "烟台果园", "name 通过 update 不可改(只能 rename)")

# ─── 7. soft_delete + 复活 ───
print("\n[7] soft_delete + 自动复活")
deleted = repo.soft_delete(s1.supplier_id)
assert_eq(deleted.is_active, False, "软删除后 is_active = False")

# 反推场景:resurrect=False
quiet = repo.get_or_create("烟台果园", resurrect=False)
assert_eq(quiet.is_active, False, "resurrect=False 不复活")

# operator 建新批次场景:resurrect=True(默认)
revived = repo.get_or_create("烟台果园")
assert_eq(revived.is_active, True, "默认 resurrect=True 自动复活")

# ─── 8. restore ───
print("\n[8] restore")
repo.soft_delete(s1.supplier_id)
restored = repo.restore(s1.supplier_id)
assert_eq(restored.is_active, True, "restore 后 is_active = True")

# ─── 9. list_all ───
print("\n[9] list_all")
repo.get_or_create("供应商 B")
repo.get_or_create("供应商 C")
all_active = repo.list_all()
assert_eq(len(all_active), 3, "活跃供应商共 3 个")

repo.soft_delete(s1.supplier_id)
only_active = repo.list_all(include_inactive=False)
assert_eq(len(only_active), 2, "排除软删除后剩 2")
with_inactive = repo.list_all(include_inactive=True)
assert_eq(len(with_inactive), 3, "include_inactive 共 3")
repo.restore(s1.supplier_id)  # 恢复

# ─── 10. rename ───
print("\n[10] rename")
renamed = repo.rename(s1.supplier_id, "烟台果园(总部)")
assert_truthy(renamed, "rename 返回对象")
assert_eq(renamed.name, "烟台果园(总部)", "名字已改")

# 用新名查得到,旧归一化键查不到
assert_truthy(repo.find_by_normalized(repo.normalize_name("烟台果园(总部)")), "新名可查到")
assert_none(repo.find_by_normalized("烟台果园"), "旧归一化键已失效")

# rename 冲突
conflict = repo.rename(s1.supplier_id, "供应商 B")
assert_none(conflict, "rename 到已存在名字应返回 None")

# ─── 11. 找不到的 ID ───
print("\n[11] 不存在的 supplier_id")
assert_none(repo.update(999999, region="X"), "update 不存在 ID 返回 None")
assert_none(repo.soft_delete(999999), "soft_delete 不存在 ID 返回 None")
assert_none(repo.rename(999999, "X"), "rename 不存在 ID 返回 None")

# ─── 收尾 ───
print()
print("=" * 70)
total = _passed + _failed
print(f"通过 {_passed}/{total},失败 {_failed}")
print("=" * 70)

# 清理临时库
try:
    import shutil
    shutil.rmtree(_TMP_DIR)
except Exception:
    pass

sys.exit(0 if _failed == 0 else 1)
