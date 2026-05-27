"""
scripts/test_state_machine.py — 状态机单元测试

不依赖 Streamlit、不依赖数据库,纯逻辑测试。
跑这个脚本验证业务规则是否正确实现。

用法:python scripts/test_state_machine.py
"""
import sys
from pathlib import Path

# 把项目根目录加到 sys.path,这样能 import models / services
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models import (
    InboundBatch, BatchStatus, User, UserRole, Supplier, FruitCategory
)
from services.batch_service import (
    apply_detection_result, confirm_batch, review_approve, review_reject,
    StateTransitionError
)
from services.permission_service import (
    can_create_batch, can_review_batch, can_manage_suppliers
)


# ── 测试用户 ──
operator = User(user_id=1, username="op1", password_hash="x",
                full_name="操作员小张", role=UserRole.OPERATOR)
manager  = User(user_id=2, username="mgr1", password_hash="x",
                full_name="经理老李",   role=UserRole.MANAGER)
admin    = User(user_id=3, username="admin", password_hash="x",
                full_name="管理员",     role=UserRole.ADMIN)


def make_batch(declared=100):
    """工厂方法:创建一个空批次"""
    return InboundBatch(
        batch_id="2026-04-29-001",
        operator_id=operator.user_id,
        supplier_name="山东果园",
        fruit_category="苹果",
        declared_count=declared,
        warehouse="一号仓",
        storage_zone="A区",
    )


def test(name, fn):
    """简易测试运行器"""
    try:
        fn()
        print(f"  ✅ {name}")
    except AssertionError as e:
        print(f"  ❌ {name}  断言失败: {e}")
    except Exception as e:
        print(f"  💥 {name}  异常: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════
# 测试 1:快速通道(完美场景)
# ═══════════════════════════════════════════════════════════════
def test_快速通道_完美场景():
    """申报 100,实际 99 (1% 差异),0 次品 → READY_TO_CONFIRM"""
    b = make_batch(declared=100)
    apply_detection_result(b, detected_total=99, grade_a=95, grade_b=4, grade_c=0,
                           avg_confidence=0.92)
    assert b.status == BatchStatus.READY_TO_CONFIRM, f"期望 READY,实际 {b.status}"
    assert b.count_diff == -1
    assert abs(b.count_diff_pct - 0.01) < 0.001
    assert b.defect_rate == 0.0


# ═══════════════════════════════════════════════════════════════
# 测试 2:数量差异过大触发复核
# ═══════════════════════════════════════════════════════════════
def test_数量差异触发复核():
    """申报 100,实际 90 (10% 差异) → PENDING_REVIEW"""
    b = make_batch(declared=100)
    apply_detection_result(b, detected_total=90, grade_a=88, grade_b=2, grade_c=0,
                           avg_confidence=0.90)
    assert b.status == BatchStatus.PENDING_REVIEW
    assert "差异" in b.status_reason


# ═══════════════════════════════════════════════════════════════
# 测试 3:次品率过高触发复核
# ═══════════════════════════════════════════════════════════════
def test_次品率触发复核():
    """申报 100,实际 100,但 C 级 10 个 (10% 次品率) → PENDING_REVIEW"""
    b = make_batch(declared=100)
    apply_detection_result(b, detected_total=100, grade_a=70, grade_b=20, grade_c=10,
                           avg_confidence=0.85)
    assert b.status == BatchStatus.PENDING_REVIEW
    assert "次品率" in b.status_reason


# ═══════════════════════════════════════════════════════════════
# 测试 4:小样本兜底
# ═══════════════════════════════════════════════════════════════
def test_小样本触发复核():
    """检测出 5 个 → 强制 PENDING_REVIEW(无论比例)"""
    b = make_batch(declared=5)
    apply_detection_result(b, detected_total=5, grade_a=5, grade_b=0, grade_c=0,
                           avg_confidence=0.95)
    assert b.status == BatchStatus.PENDING_REVIEW
    assert "样本量" in b.status_reason


# ═══════════════════════════════════════════════════════════════
# 测试 5:零目标保持草稿
# ═══════════════════════════════════════════════════════════════
def test_零目标保持草稿():
    """检测 0 个 → 保持 DRAFT"""
    b = make_batch(declared=100)
    apply_detection_result(b, detected_total=0, grade_a=0, grade_b=0, grade_c=0,
                           avg_confidence=0.0)
    assert b.status == BatchStatus.DRAFT
    assert "未检测到" in b.status_reason


# ═══════════════════════════════════════════════════════════════
# 测试 6:operator 在 READY 状态下可以快速确认
# ═══════════════════════════════════════════════════════════════
def test_operator_快速确认():
    b = make_batch(declared=100)
    apply_detection_result(b, 99, 95, 4, 0, 0.92)
    assert b.status == BatchStatus.READY_TO_CONFIRM

    confirm_batch(b, operator)
    assert b.status == BatchStatus.CONFIRMED
    assert b.confirmed_by == operator.user_id


# ═══════════════════════════════════════════════════════════════
# 测试 7:operator 在 PENDING_REVIEW 不能确认(必须 manager 复核)
# ═══════════════════════════════════════════════════════════════
def test_operator_不能跳过复核():
    b = make_batch(declared=100)
    apply_detection_result(b, 90, 88, 2, 0, 0.85)
    assert b.status == BatchStatus.PENDING_REVIEW

    try:
        confirm_batch(b, operator)
        raise AssertionError("应该抛 StateTransitionError")
    except StateTransitionError:
        pass  # 期望的


# ═══════════════════════════════════════════════════════════════
# 测试 8:manager 复核通过
# ═══════════════════════════════════════════════════════════════
def test_manager_复核通过():
    b = make_batch(declared=100)
    apply_detection_result(b, 90, 88, 2, 0, 0.85)
    assert b.status == BatchStatus.PENDING_REVIEW

    review_approve(b, manager, note="实物核验通过")
    assert b.status == BatchStatus.CONFIRMED
    assert b.reviewed_by  == manager.user_id
    assert b.confirmed_by == manager.user_id


# ═══════════════════════════════════════════════════════════════
# 测试 9:manager 拒收
# ═══════════════════════════════════════════════════════════════
def test_manager_拒收():
    b = make_batch(declared=100)
    apply_detection_result(b, 90, 80, 0, 10, 0.80)
    assert b.status == BatchStatus.PENDING_REVIEW

    review_reject(b, manager, reason="次品率过高,且数量短少明显")
    assert b.status == BatchStatus.REJECTED
    assert b.rejected_by == manager.user_id


# ═══════════════════════════════════════════════════════════════
# 测试 10:operator 不能复核(权限不足)
# ═══════════════════════════════════════════════════════════════
def test_operator_无复核权限():
    b = make_batch(declared=100)
    apply_detection_result(b, 90, 88, 2, 0, 0.85)

    try:
        review_approve(b, operator, note="x")
        raise AssertionError("应该抛 StateTransitionError")
    except StateTransitionError:
        pass


# ═══════════════════════════════════════════════════════════════
# 测试 11:权限服务 — 谁能管理供应商
# ═══════════════════════════════════════════════════════════════
def test_权限_供应商管理():
    assert can_manage_suppliers(admin)    is True
    assert can_manage_suppliers(manager)  is False
    assert can_manage_suppliers(operator) is False


# ═══════════════════════════════════════════════════════════════
# 运行全部测试
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 入库批次状态机单元测试")
    print("=" * 70)

    test("快速通道-完美场景",       test_快速通道_完美场景)
    test("数量差异触发复核",        test_数量差异触发复核)
    test("次品率触发复核",          test_次品率触发复核)
    test("小样本触发复核",          test_小样本触发复核)
    test("零目标保持草稿",          test_零目标保持草稿)
    test("operator 快速确认",       test_operator_快速确认)
    test("operator 不能跳过复核",   test_operator_不能跳过复核)
    test("manager 复核通过",        test_manager_复核通过)
    test("manager 拒收",            test_manager_拒收)
    test("operator 无复核权限",     test_operator_无复核权限)
    test("权限-供应商管理",         test_权限_供应商管理)

    print()
    print("=" * 70)
    print("✅ 测试全部完成")
    print("=" * 70)