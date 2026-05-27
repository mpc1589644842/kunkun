"""
models.py — 业务实体模型定义

包含:
  - 状态枚举(BatchStatus, UserRole)
  - 实体 dataclass(User, Supplier, FruitCategory, InboundBatch, OperationLog)

设计原则:
  - 纯数据类,不含业务逻辑(逻辑在 services/ 中)
  - 不绑定数据库,后续接入 SQLite 时可平滑替换为 SQLAlchemy ORM
  - 所有时间字段统一用 datetime,序列化时再转字符串
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# 枚举定义
# ═══════════════════════════════════════════════════════════════

class BatchStatus(str, Enum):
    """入库批次状态机"""
    DRAFT             = "DRAFT"              # 草稿(已创建,未检测)
    READY_TO_CONFIRM  = "READY_TO_CONFIRM"   # 就绪(检测完成,符合快速通道)
    PENDING_REVIEW    = "PENDING_REVIEW"     # 待复核(检测完成,需 manager 复核)
    CONFIRMED         = "CONFIRMED"          # 已入库(终态)
    REJECTED          = "REJECTED"           # 已拒收(终态)

    @property
    def display_name(self) -> str:
        """中文显示名"""
        return {
            "DRAFT":            "草稿",
            "READY_TO_CONFIRM": "待确认",
            "PENDING_REVIEW":   "待复核",
            "CONFIRMED":        "已入库",
            "REJECTED":         "已拒收",
        }[self.value]

    @property
    def color(self) -> str:
        """UI 显示颜色(配合 Streamlit)"""
        return {
            "DRAFT":            "#9CA3AF",  # 灰
            "READY_TO_CONFIRM": "#22C55E",  # 绿
            "PENDING_REVIEW":   "#F59E0B",  # 黄
            "CONFIRMED":        "#16A34A",  # 深绿
            "REJECTED":         "#EF4444",  # 红
        }[self.value]

    @property
    def is_terminal(self) -> bool:
        """是否终态(不可再流转)"""
        return self in {BatchStatus.CONFIRMED, BatchStatus.REJECTED}


class UserRole(str, Enum):
    """用户角色"""
    OPERATOR = "operator"   # 操作员:创建批次、跑检测、提交快速入库
    MANAGER  = "manager"    # 经理:复核(通过/拒收)
    ADMIN    = "admin"      # 管理员:维护供应商、品类、用户

    @property
    def display_name(self) -> str:
        return {
            "operator": "操作员",
            "manager":  "经理",
            "admin":    "管理员",
        }[self.value]


# ═══════════════════════════════════════════════════════════════
# 实体定义
# ═══════════════════════════════════════════════════════════════

@dataclass
class User:
    """用户"""
    user_id:       int
    username:      str             # 登录名(唯一)
    password_hash: str             # bcrypt 哈希(暂时存明文,接 SQLite 后改)
    full_name:     str
    role:          UserRole
    phone:         str = ""
    is_active:     bool = True
    created_at:    datetime = field(default_factory=datetime.now)


@dataclass
class Supplier:
    """供应商"""
    supplier_id:   int
    name:          str             # 供应商名称
    contact_phone: str = ""
    address:       str = ""
    is_active:     bool = True
    created_at:    datetime = field(default_factory=datetime.now)
    # ── Step 7.1: 档案扩展字段 ──
    region:        str = ""           # 地域(如 "山东烟台" / "广西南宁")
    notes:         str = ""           # 备注/资质说明
    updated_at:    datetime = field(default_factory=datetime.now)  # 最近更新时间


@dataclass
class FruitCategory:
    """水果品类参考表"""
    category_id:  int
    name:         str              # "苹果" / "香蕉" / "葡萄" / "橙子"
    code:         str              # "apple" / "banana" / ...(英文,数据库友好)
    typical_unit: str = "个"        # 盘点单位:"个" / "串"
    is_active:    bool = True


@dataclass
class InboundBatch:
    """
    入库批次 — 系统核心实体

    生命周期:
      DRAFT → (跑检测,触发自动判定) → READY_TO_CONFIRM 或 PENDING_REVIEW
        → (operator 确认 或 manager 复核) → CONFIRMED 或 REJECTED
    """

    # ── 基础信息 ──
    batch_id:       str                          # 批次号(主键,自动生成或手填)
    operator_id:    int                          # 操作员 user_id
    operator_name:  str = ""                     # 冗余字段,显示用
    inbound_date:   datetime = field(default_factory=datetime.now)

    # ── 货源(对账三要素)──
    supplier_id:    Optional[int] = None         # 关联 Supplier(可空,允许临时录入)
    supplier_name:  str = ""                     # 冗余字段,显示用
    fruit_category: str = ""                     # "苹果" / "香蕉" / ...
    declared_count: int = 0                      # 🎯 申报数量

    # ── 仓储 ──
    warehouse:      str = ""
    storage_zone:   str = ""

    # ── 检测结果 ──
    detected_total: int   = 0
    grade_a_count:  int   = 0
    grade_b_count:  int   = 0
    grade_c_count:  int   = 0
    avg_confidence: float = 0.0

    # ── Step 5.1: 累积式检测支持 ──
    # 累积式批次:同一批次允许多次"补充检测",每次只跑新增的图片。
    # processed_files 记录已检测过的文件名(以 file.name 为键),避免重复推理。
    # detection_rounds 记录该批次累计被检测了几轮,用于审计与论文素材。
    processed_files:  set = field(default_factory=set)
    detection_rounds: int = 0

    # ── 对账结论(系统自动算)──
    count_diff:      int   = 0       # detected_total - declared_count
    count_diff_pct:  float = 0.0     # |count_diff| / declared_count
    qualified_rate:  float = 0.0     # (A+B) / total
    defect_rate:     float = 0.0     # C / total

    # ── 状态机 ──
    status:         BatchStatus = BatchStatus.DRAFT
    status_reason:  str = ""         # 系统自动判定时记录原因

    # ── 流转记录(留下决策痕迹)──
    reviewed_by:      Optional[int]      = None
    reviewed_at:      Optional[datetime] = None
    reviewed_note:    str = ""
    reviewed_by_name: str = ""         # 冗余字段,显示用(同 supplier_name 设计)

    confirmed_by:      Optional[int]      = None
    confirmed_at:      Optional[datetime] = None
    confirmed_by_name: str = ""         # 冗余字段,显示用

    rejected_by:      Optional[int]      = None
    rejected_at:      Optional[datetime] = None
    rejected_by_name: str = ""         # 冗余字段,显示用
    rejection_reason: str = ""

    # ── Step 6.1a: 复核 KPI 快照 ──
    # 复核当下的关键指标(申报/检测/差异率/不良率/已检测轮数)。
    # 后续若 operator 重检,batch 当前数据会变,但快照保留 manager 决策依据。
    # 字段结构见 services/batch_service.py: record_review_action()
    review_snapshot: dict = field(default_factory=dict)

    # ── 备注 ──
    notes: str = ""

    def recalculate_metrics(self):
        """根据检测结果,重新计算对账与质检指标"""
        # 数量差额
        self.count_diff = self.detected_total - self.declared_count
        if self.declared_count > 0:
            self.count_diff_pct = abs(self.count_diff) / self.declared_count
        else:
            self.count_diff_pct = 0.0

        # 质检率
        if self.detected_total > 0:
            self.qualified_rate = (self.grade_a_count + self.grade_b_count) / self.detected_total
            self.defect_rate    = self.grade_c_count / self.detected_total
        else:
            self.qualified_rate = 0.0
            self.defect_rate    = 0.0


@dataclass
class OperationLog:
    """操作审计日志(只插入,不修改,不删除)"""
    log_id:      int
    user_id:     int
    action:      str              # "create_batch" / "run_detection" / "approve" / "reject" / ...
    target_type: str              # "batch" / "supplier" / "user" / ...
    target_id:   str              # 字符串以兼容批次号
    detail:      str = ""         # JSON 字符串,记录变更
    created_at:  datetime = field(default_factory=datetime.now)