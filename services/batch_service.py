"""
services/batch_service.py — 入库批次状态机

核心规则(三阈值快速通道判定):
  - count_diff_pct ≤ 3% 且
  - defect_rate ≤ 5% 且
  - detected_total ≥ 10 (小样本兜底)
  → READY_TO_CONFIRM (符合快速通道)

  否则 → PENDING_REVIEW (强制复核)

特殊情况:
  - detected_total == 0 → 保持 DRAFT,提示"未检测到任何目标"
"""
from datetime import datetime
from typing import Tuple

from models import InboundBatch, BatchStatus, User, UserRole


# ═══════════════════════════════════════════════════════════════
# 阈值配置(集中管理,方便论文引用)
# ═══════════════════════════════════════════════════════════════

THRESHOLD_COUNT_DIFF_PCT  = 0.03   # 数量差异 ≤ 3% 视为可信
THRESHOLD_DEFECT_RATE     = 0.05   # 次品率 ≤ 5% 视为合格
THRESHOLD_MIN_SAMPLE_SIZE = 10     # 检测目标数 ≥ 10 才信任统计


# ═══════════════════════════════════════════════════════════════
# 状态自动判定(检测完成后调用)
# ═══════════════════════════════════════════════════════════════

def determine_status_after_detection(batch: InboundBatch) -> Tuple[BatchStatus, str]:
    """
    检测完成后,根据三阈值规则判定批次应进入哪个状态

    Args:
        batch: 已填充检测结果的批次(detected_total 等字段已就绪)

    Returns:
        (新状态, 判定原因)
    """
    # 先确保指标都是最新的
    batch.recalculate_metrics()

    # 0 个目标:保留 DRAFT,提示用户
    if batch.detected_total == 0:
        return BatchStatus.DRAFT, "未检测到任何目标,请人工补检后重试"

    # 小样本兜底
    if batch.detected_total < THRESHOLD_MIN_SAMPLE_SIZE:
        return (
            BatchStatus.PENDING_REVIEW,
            f"样本量过小(检测到 {batch.detected_total} 个,< {THRESHOLD_MIN_SAMPLE_SIZE}),需人工复核"
        )

    # 数量差异检查
    if batch.count_diff_pct > THRESHOLD_COUNT_DIFF_PCT:
        return (
            BatchStatus.PENDING_REVIEW,
            f"申报与实际数量差异 {batch.count_diff_pct*100:.1f}% "
            f"(申报 {batch.declared_count} / 实际 {batch.detected_total}),"
            f"超过 {THRESHOLD_COUNT_DIFF_PCT*100:.0f}% 阈值,需复核"
        )

    # 次品率检查
    if batch.defect_rate > THRESHOLD_DEFECT_RATE:
        return (
            BatchStatus.PENDING_REVIEW,
            f"次品率 {batch.defect_rate*100:.1f}% (C 级 {batch.grade_c_count}/{batch.detected_total}) "
            f"超过 {THRESHOLD_DEFECT_RATE*100:.0f}% 阈值,需复核"
        )

    # 通过快速通道
    return (
        BatchStatus.READY_TO_CONFIRM,
        f"符合快速入库条件:数量差异 {batch.count_diff_pct*100:.1f}%、"
        f"次品率 {batch.defect_rate*100:.1f}%、检测样本 {batch.detected_total} 个"
    )


def apply_detection_result(batch: InboundBatch,
                           detected_total: int,
                           grade_a: int, grade_b: int, grade_c: int,
                           avg_confidence: float) -> Tuple[BatchStatus, str]:
    """
    把模型检测结果填入批次,并自动判定状态

    这是 Streamlit 中"跑完检测"后调用的入口
    """
    batch.detected_total = detected_total
    batch.grade_a_count  = grade_a
    batch.grade_b_count  = grade_b
    batch.grade_c_count  = grade_c
    batch.avg_confidence = avg_confidence

    # 调用判定逻辑(内部会重算 metrics)
    new_status, reason = determine_status_after_detection(batch)
    batch.status        = new_status
    batch.status_reason = reason
    return new_status, reason


# ═══════════════════════════════════════════════════════════════
# Step 5.2: 累积式检测支持
# ═══════════════════════════════════════════════════════════════

def apply_supplementary_detection(batch: InboundBatch,
                                  new_count: int,
                                  new_a: int, new_b: int, new_c: int,
                                  new_avg_confidence: float,
                                  newly_processed_files: set) -> Tuple[BatchStatus, str]:
    """
    补充检测(累加式)— 同一批次第 2/3/N 次检测时调用

    与 apply_detection_result 的核心区别:
      - apply_detection_result 是"覆盖式":新结果直接替换旧结果(适合首次检测)
      - apply_supplementary_detection 是"累加式":新结果加到旧结果上

    Args:
        batch: 已经被首检过的批次,detected_total/grade_x_count 都有值
        new_count: 本轮新检测到的目标数
        new_a/b/c: 本轮各等级目标数
        new_avg_confidence: 本轮新检测的平均置信度
        newly_processed_files: 本轮处理的文件名集合(用于追加到 batch.processed_files)

    Returns:
        (新状态, 判定原因)
    """
    # 累加计数
    old_total = batch.detected_total
    old_avg   = batch.avg_confidence

    batch.detected_total += new_count
    batch.grade_a_count  += new_a
    batch.grade_b_count  += new_b
    batch.grade_c_count  += new_c

    # 加权平均置信度:(老平均 × 老总数 + 新平均 × 新总数) / 累计总数
    if batch.detected_total > 0:
        batch.avg_confidence = (
            old_avg * old_total + new_avg_confidence * new_count
        ) / batch.detected_total
    else:
        batch.avg_confidence = 0.0

    # 标记本轮处理过的文件
    batch.processed_files.update(newly_processed_files)
    batch.detection_rounds += 1

    # 状态重判定(累加后总数变了,可能从 PENDING 升级到 READY,也可能反过来)
    new_status, reason = determine_status_after_detection(batch)
    batch.status        = new_status
    batch.status_reason = reason
    return new_status, reason


def reset_detection(batch: InboundBatch) -> None:
    """
    重新检测 — 清空所有累积的检测结果,但保留批次身份和审计字段

    适用场景:operator 对当前检测结果不满意,想从零重新检测
    限制:仅在非终态(DRAFT/READY_TO_CONFIRM/PENDING_REVIEW)时可调用

    保留:batch_id, declared_count, supplier_name, 等批次身份字段
          detection_rounds(继续累加,审计要看 operator 改了几次主意)
    清空:detected_total, grade_x_count, avg_confidence, processed_files
          status → DRAFT, status_reason 清空
    """
    if batch.status.is_terminal:
        raise StateTransitionError(
            f"批次已处于终态({batch.status.display_name}),无法重新检测"
        )

    batch.detected_total  = 0
    batch.grade_a_count   = 0
    batch.grade_b_count   = 0
    batch.grade_c_count   = 0
    batch.avg_confidence  = 0.0
    batch.processed_files = set()
    # detection_rounds 不清,保留审计痕迹
    batch.status          = BatchStatus.DRAFT
    batch.status_reason   = ""

    # 重算对账指标(全部归零,但要保持字段一致)
    batch.recalculate_metrics()


# ═══════════════════════════════════════════════════════════════
# 状态流转(operator/manager 操作触发)
# ═══════════════════════════════════════════════════════════════

class StateTransitionError(Exception):
    """状态流转非法(权限不足或当前状态不允许此操作)"""


def confirm_batch(batch: InboundBatch, operator: User) -> None:
    """
    操作员确认入库(快速通道)
    要求:状态为 READY_TO_CONFIRM,角色为 operator/manager/admin
    """
    if batch.status != BatchStatus.READY_TO_CONFIRM:
        raise StateTransitionError(
            f"当前状态 {batch.status.display_name},无法直接确认入库,需先复核"
        )
    if operator.role not in {UserRole.OPERATOR, UserRole.MANAGER, UserRole.ADMIN}:
        raise StateTransitionError("权限不足,无法执行确认操作")

    batch.status             = BatchStatus.CONFIRMED
    batch.confirmed_by       = operator.user_id
    batch.confirmed_at       = datetime.now()
    batch.confirmed_by_name  = operator.full_name

def review_approve(batch: InboundBatch, manager: User, note: str = "") -> None:
    """
    经理复核通过(进入已入库)
    要求:状态为 PENDING_REVIEW 或 READY_TO_CONFIRM,角色为 manager/admin

    Step 6.2-fix-1 起放宽:READY_TO_CONFIRM 状态也允许 manager 直接复核通过,
    用于兜底处理 operator 长时间未确认的快速通道批次。
    """
    if batch.status not in {BatchStatus.PENDING_REVIEW, BatchStatus.READY_TO_CONFIRM}:
        raise StateTransitionError(
            f"当前状态 {batch.status.display_name},无法复核"
        )
    if manager.role not in {UserRole.MANAGER, UserRole.ADMIN}:
        raise StateTransitionError("权限不足,仅经理可复核")


    # Step 6.1b: 复核 KPI 快照(后续重检不影响该快照)
    batch.review_snapshot = {
        "action":           "approved",
        "declared_total":   batch.declared_count,
        "detected_total":   batch.detected_total,
        "count_diff_pct":   batch.count_diff_pct,
        "defect_rate":      batch.defect_rate,
        "detection_rounds": batch.detection_rounds,
    }

    batch.reviewed_by   = manager.user_id
    batch.reviewed_at   = datetime.now()
    batch.reviewed_note     = note
    batch.reviewed_by_name  = manager.full_name
    batch.confirmed_by       = manager.user_id
    batch.confirmed_at       = datetime.now()
    batch.confirmed_by_name  = manager.full_name
    batch.status             = BatchStatus.CONFIRMED


def review_reject(batch: InboundBatch, manager: User, reason: str) -> None:
    """
    经理拒收(进入已拒收)
    要求:状态为 PENDING_REVIEW 或 READY_TO_CONFIRM,角色为 manager/admin
    """
    if batch.status not in {BatchStatus.PENDING_REVIEW, BatchStatus.READY_TO_CONFIRM}:
        raise StateTransitionError(
            f"当前状态 {batch.status.display_name},无法拒收"
        )
    if manager.role not in {UserRole.MANAGER, UserRole.ADMIN}:
        raise StateTransitionError("权限不足,仅经理可拒收")
    if not reason.strip():
        raise StateTransitionError("拒收必须填写原因")


    # Step 6.1b: 复核 KPI 快照(后续重检不影响该快照)
    batch.review_snapshot = {
        "action":           "rejected",
        "declared_total":   batch.declared_count,
        "detected_total":   batch.detected_total,
        "count_diff_pct":   batch.count_diff_pct,
        "defect_rate":      batch.defect_rate,
        "detection_rounds": batch.detection_rounds,
    }

    batch.rejected_by       = manager.user_id
    batch.rejected_at       = datetime.now()
    batch.rejected_by_name  = manager.full_name
    batch.rejection_reason  = reason
    batch.reviewed_by_name  = manager.full_name
    batch.status            = BatchStatus.REJECTED