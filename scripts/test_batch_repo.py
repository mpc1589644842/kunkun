"""
scripts/test_batch_repo.py — BatchRepository 单元测试 (Step 9.2.2.c.1)
================================================
独立运行,临时数据库,不污染 fruit_app.db
"""
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# 临时数据库
_TMP_DIR = tempfile.mkdtemp(prefix="batch_repo_test_")
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

from db.repositories import batch_repo as repo
from models import InboundBatch, BatchStatus


# ─── 测试工具 ───
_passed = 0
_failed = 0
def assert_eq(a, e, msg=""):
    global _passed, _failed
    if a == e:
        _passed += 1; print(f"  ✅ {msg}")
    else:
        _failed += 1
        print(f"  ❌ {msg}\n     期望: {e!r}\n     实际: {a!r}")

def assert_truthy(a, msg=""):
    global _passed, _failed
    if a:
        _passed += 1; print(f"  ✅ {msg}")
    else:
        _failed += 1; print(f"  ❌ {msg}  (期望 truthy,实际:{a!r})")

def assert_none(a, msg=""):
    global _passed, _failed
    if a is None:
        _passed += 1; print(f"  ✅ {msg}")
    else:
        _failed += 1; print(f"  ❌ {msg}  (期望 None,实际:{a!r})")


# ─── 构造工具:最小有效 dataclass ───
def make_batch(batch_id="B001", **overrides):
    defaults = dict(
        batch_id        = batch_id,
        operator_id     = 1,
        operator_name   = "测试操作员",
        inbound_date    = datetime(2026, 5, 11, 10, 0, 0),
        supplier_name   = "测试供应商",
        fruit_category  = "苹果",
        declared_count  = 100,
        warehouse       = "一号仓",
        storage_zone    = "A区",
    )
    defaults.update(overrides)
    return InboundBatch(**defaults)


# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("  BatchRepository 单元测试")
print("=" * 70)

# ─── 1. save 新建 ───
print("\n[1] save 新建")
b1 = make_batch("B001")
saved = repo.save(b1)
assert_truthy(saved, "save 返回 dataclass")
assert_eq(saved.batch_id, "B001", "batch_id 保留")
assert_eq(saved.declared_count, 100, "declared_count 保留")
assert_eq(saved.status, BatchStatus.DRAFT, "默认状态 DRAFT")
assert_eq(saved.processed_files, set(), "processed_files 默认空 set")
assert_eq(saved.review_snapshot, {}, "review_snapshot 默认空 dict")

# ─── 2. find_by_id ───
print("\n[2] find_by_id")
found = repo.find_by_id("B001")
assert_truthy(found, "find_by_id 命中")
assert_eq(found.declared_count, 100, "字段值一致")
assert_none(repo.find_by_id("NOT_EXIST"), "不存在返回 None")
assert_none(repo.find_by_id(""), "空 batch_id 返回 None")

# ─── 3. exists ───
print("\n[3] exists")
assert_eq(repo.exists("B001"), True, "存在返回 True")
assert_eq(repo.exists("NOT_EXIST"), False, "不存在返回 False")
assert_eq(repo.exists(""), False, "空字符返回 False")

# ─── 4. save upsert ───
print("\n[4] save upsert(同 ID 第二次 save 是 update)")
b1_v2 = make_batch("B001", declared_count=200, supplier_name="供应商已改")
updated = repo.save(b1_v2)
assert_eq(updated.declared_count, 200, "declared_count 已更新")
assert_eq(updated.supplier_name, "供应商已改", "supplier_name 已更新")
# 验证只剩 1 条而不是 2 条
assert_eq(repo.count(), 1, "upsert 后总数仍为 1")

# ─── 5. processed_files set <-> list ───
print("\n[5] processed_files set ↔ list")
b2 = make_batch("B002", processed_files={"a.jpg", "b.jpg", "c.jpg"}, detection_rounds=2)
repo.save(b2)
fetched = repo.find_by_id("B002")
assert_eq(fetched.processed_files, {"a.jpg", "b.jpg", "c.jpg"}, "set 内容保留")
assert_truthy(isinstance(fetched.processed_files, set), "类型仍是 set")
assert_eq(fetched.detection_rounds, 2, "detection_rounds 保留")

# 空 set
b3 = make_batch("B003", processed_files=set())
repo.save(b3)
assert_eq(repo.find_by_id("B003").processed_files, set(), "空 set 也保留")

# ─── 6. review_snapshot dict ───
print("\n[6] review_snapshot JSON dict")
snap = {
    "action": "approved",
    "declared_total": 100,
    "detected_total": 98,
    "count_diff_pct": 0.02,
    "defect_rate": 0.03,
    "detection_rounds": 1,
}
b4 = make_batch("B004", review_snapshot=snap, status=BatchStatus.CONFIRMED)
repo.save(b4)
fetched = repo.find_by_id("B004")
assert_eq(fetched.review_snapshot, snap, "snapshot dict 内容一致")
assert_eq(fetched.status, BatchStatus.CONFIRMED, "status 枚举正确还原")

# ─── 7. apply_changes 局部更新 ───
print("\n[7] apply_changes 局部更新")
result = repo.apply_changes("B001",
                            status=BatchStatus.READY_TO_CONFIRM,
                            detected_total=95,
                            grade_a_count=80,
                            grade_b_count=10,
                            grade_c_count=5)
assert_truthy(result, "apply_changes 返回对象")
assert_eq(result.status, BatchStatus.READY_TO_CONFIRM, "status 已更新")
assert_eq(result.detected_total, 95, "detected_total 已更新")
# 其他字段不变
assert_eq(result.declared_count, 200, "未涉及字段保持不变")

# apply_changes 自动转 set -> list
result2 = repo.apply_changes("B001", processed_files={"x.jpg", "y.jpg"})
assert_eq(result2.processed_files, {"x.jpg", "y.jpg"}, "set 参数被正确接受")

# apply_changes 不存在的 batch
assert_none(repo.apply_changes("NOT_EXIST", status=BatchStatus.CONFIRMED), "不存在返回 None")

# apply_changes 防御性忽略 batch_id 键(防止意外覆盖主键)
result3 = repo.apply_changes("B001", batch_id="HACK")
assert_eq(result3.batch_id, "B001", "batch_id 字段会被防御性忽略")
assert_none(repo.find_by_id("HACK"), "HACK ID 没有被创建")

# ─── 8. list_all 基础 ───
print("\n[8] list_all 基础")
all_batches = repo.list_all()
assert_eq(len(all_batches), 4, "共 4 个批次(B001~B004)")

# 排除 DRAFT
non_draft = repo.list_all(include_draft=False)
# B001 status=READY_TO_CONFIRM(改过)、B002/B003 DRAFT(默认)、B004 CONFIRMED
# 所以排除 DRAFT 后应该剩 2(B001 + B004)
assert_eq(len(non_draft), 2, "排除 DRAFT 后剩 2(READY + CONFIRMED)")

# ─── 9. list_all 按状态筛选 ───
print("\n[9] list_all 按状态筛选")
confirmed = repo.list_all(status_in=[BatchStatus.CONFIRMED])
assert_eq(len(confirmed), 1, "仅 CONFIRMED 1 条")
assert_eq(confirmed[0].batch_id, "B004", "是 B004")

multi = repo.list_all(status_in=[BatchStatus.DRAFT, BatchStatus.CONFIRMED])
assert_eq(len(multi), 3, "DRAFT + CONFIRMED 共 3 条")

empty = repo.list_all(status_in=[])
assert_eq(len(empty), 4, "空 status_in 视为不过滤(行为符合 list_all 设计)")

# ─── 10. list_all 按 supplier 筛选 ───
print("\n[10] list_all 按 supplier 筛选")
b5 = make_batch("B005", supplier_id=42)
b6 = make_batch("B006", supplier_id=42)
b7 = make_batch("B007", supplier_id=99)
repo.save(b5); repo.save(b6); repo.save(b7)
sup42 = repo.list_all(supplier_id=42)
assert_eq(len(sup42), 2, "supplier_id=42 共 2 条")
sup99 = repo.list_all(supplier_id=99)
assert_eq(len(sup99), 1, "supplier_id=99 共 1 条")
sup_none = repo.list_all(supplier_id=999)
assert_eq(len(sup_none), 0, "不存在的 supplier_id 返回空")

# ─── 11. count ───
print("\n[11] count")
assert_eq(repo.count(), 7, "总数 7")
assert_eq(repo.count(supplier_id=42), 2, "supplier=42 共 2")

# ─── 12. delete ───
print("\n[12] delete")
assert_eq(repo.delete("B007"), True, "删除存在的 batch 返回 True")
assert_eq(repo.delete("B007"), False, "再次删除返回 False")
assert_eq(repo.delete("NOT_EXIST"), False, "删除不存在返回 False")
assert_eq(repo.count(), 6, "删除后剩 6")

# ─── 13. delete_all ───
print("\n[13] delete_all")
n = repo.delete_all()
assert_eq(n, 6, "delete_all 返回删除条数")
assert_eq(repo.count(), 0, "清空后 0 条")

# ─── 14. 字段完整性:31 字段全部往返 ───
print("\n[14] 字段完整性:复杂 batch 往返(往复保真)")
now = datetime(2026, 5, 11, 12, 0, 0)
full = InboundBatch(
    batch_id          = "FULL001",
    operator_id       = 1,
    operator_name     = "Alice",
    inbound_date      = now,
    supplier_id       = 7,
    supplier_name     = "供应商七",
    fruit_category    = "葡萄",
    declared_count    = 500,
    warehouse         = "三号仓",
    storage_zone      = "C区",
    detected_total    = 487,
    grade_a_count     = 400,
    grade_b_count     = 70,
    grade_c_count     = 17,
    avg_confidence    = 0.923,
    processed_files   = {"img1.jpg", "img2.jpg"},
    detection_rounds  = 3,
    count_diff        = -13,
    count_diff_pct    = 0.026,
    qualified_rate    = 0.965,
    defect_rate       = 0.035,
    status            = BatchStatus.PENDING_REVIEW,
    status_reason     = "次品率偏高需复核",
    reviewed_by       = 2,
    reviewed_at       = now + timedelta(hours=1),
    reviewed_note     = "已检查图片,可入库",
    reviewed_by_name  = "Bob",
    confirmed_by      = 1,
    confirmed_at      = now + timedelta(hours=2),
    confirmed_by_name = "Alice",
    rejected_by       = None,
    rejected_at       = None,
    rejected_by_name  = "",
    rejection_reason  = "",
    review_snapshot   = {"action": "approved", "key": "value"},
    notes             = "测试备注",
)
saved = repo.save(full)
back = repo.find_by_id("FULL001")
assert_eq(back.operator_id, 1, "operator_id")
assert_eq(back.supplier_id, 7, "supplier_id")
assert_eq(back.declared_count, 500, "declared_count")
assert_eq(back.avg_confidence, 0.923, "avg_confidence(float)")
assert_eq(back.processed_files, {"img1.jpg", "img2.jpg"}, "processed_files (set)")
assert_eq(back.detection_rounds, 3, "detection_rounds")
assert_eq(back.qualified_rate, 0.965, "qualified_rate")
assert_eq(back.status, BatchStatus.PENDING_REVIEW, "status 枚举")
assert_eq(back.reviewed_at, now + timedelta(hours=1), "reviewed_at datetime")
assert_eq(back.review_snapshot, {"action": "approved", "key": "value"}, "review_snapshot")
assert_eq(back.notes, "测试备注", "中文 notes")

# ─── 收尾 ───
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
