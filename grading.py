"""
grading.py — 水果三级分级逻辑模块 v2.0
类别映射 (16类):
  0:ripe_apple  1:unripe_apple  2:rotten_apple
  3:ripe_banana 4:unripe_banana 5:rotten_banana
  6:ripe_grape  7:unripe_grape
  8:ripe_strawberry 9:unripe_strawberry
  10:ripe_persimmon 11:unripe_persimmon
  12:ripe_orange 13:unripe_orange 14:rotten_orange
  15:rotten_fruit
"""

from dataclasses import dataclass

CLASS_META = {
    0:  {"fruit": "苹果",  "ripeness": "ripe"},
    1:  {"fruit": "苹果",  "ripeness": "unripe"},
    2:  {"fruit": "苹果",  "ripeness": "rotten"},
    3:  {"fruit": "香蕉",  "ripeness": "ripe"},
    4:  {"fruit": "香蕉",  "ripeness": "unripe"},
    5:  {"fruit": "香蕉",  "ripeness": "rotten"},
    6:  {"fruit": "葡萄",  "ripeness": "ripe"},
    7:  {"fruit": "葡萄",  "ripeness": "unripe"},
    8: {"fruit": "草莓", "ripeness": "ripe"},  # 保留，但后续过滤
    9: {"fruit": "草莓", "ripeness": "unripe"},
    10: {"fruit": "橙子", "ripeness": "ripe"},  # 原"柿子"→"橙子"
    11: {"fruit": "橙子", "ripeness": "unripe"},  # 原"柿子"→"橙子"
    12: {"fruit": "橙子",  "ripeness": "ripe"},
    13: {"fruit": "橙子",  "ripeness": "unripe"},
    14: {"fruit": "橙子",  "ripeness": "rotten"},
    15: {"fruit": "水果",  "ripeness": "rotten"},
}

RIPENESS_LABEL = {
    "ripe":   "成熟",
    "unripe": "未成熟",
    "rotten": "腐烂",
}

GRADE_CONFIG = {
    "A": {
        "label":       "一级果 (Grade A)",
        "color":       "#22c55e",
        "emoji":       "🟢",
        "description": "品质优良，外观完整，成熟度最佳",
        "action":      "✅ 建议：直接上市 / 精品礼盒装",
    },
    "B": {
        "label":       "二级果 (Grade B)",
        "color":       "#f59e0b",
        "emoji":       "🟡",
        "description": "品质一般，成熟度不足或置信度偏低",
        "action":      "⚠️ 建议：超市散装 / 批发渠道销售",
    },
    "C": {
        "label":       "三级果 (Grade C)",
        "color":       "#ef4444",
        "emoji":       "🔴",
        "description": "存在腐烂迹象或检测置信度过低",
        "action":      "❌ 建议：降级加工处理 / 废弃",
    },
}

CLASS_NAMES = [
    "ripe_apple", "unripe_apple", "rotten_apple",
    "ripe_banana", "unripe_banana", "rotten_banana",
    "ripe_grape", "unripe_grape",
    "ripe_strawberry", "unripe_strawberry",
    "ripe_persimmon", "unripe_persimmon",
    "ripe_orange", "unripe_orange", "rotten_orange",
    "rotten_fruit",
]


@dataclass
class GradeResult:
    class_id:    int
    class_name:  str
    fruit:       str
    ripeness:    str
    confidence:  float
    grade:       str
    grade_label: str
    grade_color: str
    grade_emoji: str
    description: str
    action:      str


def classify_grade(class_id: int, confidence: float) -> GradeResult:
    # ── 草莓重定向：业务场景中入库不涉及草莓，
    # ── 模型误判为草莓的目标几乎都是红苹果，强制改判
    REDIRECT_MAP = {
        8: 0,   # ripe_strawberry   → ripe_apple
        9: 1,   # unripe_strawberry → unripe_apple
    }
    if class_id in REDIRECT_MAP:
        class_id = REDIRECT_MAP[class_id]

    meta     = CLASS_META.get(class_id, {"fruit": "未知", "ripeness": "rotten"})
    ripeness = meta["ripeness"]

    # ── 业务规则:rotten_fruit 通用腐烂类弃用 ──
    # 数据依据:验证集混淆矩阵显示 rotten_fruit (class 15)
    # mAP@0.5:0.95 = 0.639,为 16 类中最低;且对条纹富士苹果存在
    # 高置信度误报(实测 conf 可达 0.95+),业务侧不可信。
    # 因此该类无论置信度如何,统一降为 B 级人工复核,不进入 C 级。
    if class_id == 15:
        grade = "B"
    # ── rotten_apple 高门槛保护(class 2) ──
    # 富士条纹苹果易被误判为腐烂苹果,要求 conf ≥ 0.85 才认定
    elif ripeness == "rotten" and class_id == 2 and confidence < 0.85:
        grade = "B"
    # ── persimmon 高门槛保护(class 10/11) ──
    # 显示为"橙子"的柿子类,在红苹果场景下易误判,要求 conf ≥ 0.80
    elif class_id in {10, 11} and confidence < 0.80:
        grade = "B"
    # ── 标准分级规则 ──
    elif ripeness == "rotten" or confidence < 0.30:
        grade = "C"
    elif ripeness == "ripe" and confidence >= 0.70:
        grade = "A"
    else:
        grade = "B"

    cfg        = GRADE_CONFIG[grade]
    class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else "unknown"

    return GradeResult(
        class_id=class_id,
        class_name=class_name,
        fruit=meta["fruit"],
        ripeness=RIPENESS_LABEL.get(ripeness, ripeness),
        confidence=confidence,
        grade=grade,
        grade_label=cfg["label"],
        grade_color=cfg["color"],
        grade_emoji=cfg["emoji"],
        description=cfg["description"],
        action=cfg["action"],
    )

def grade_batch(detections: list[dict]) -> list[GradeResult]:
    return [classify_grade(d["class_id"], d["confidence"]) for d in detections]


def summarize_grades(results: list[GradeResult]) -> dict:
    summary  = {"total": len(results), "A": 0, "B": 0, "C": 0, "by_fruit": {}}
    conf_sum = 0.0
    for r in results:
        summary[r.grade] += 1
        conf_sum         += r.confidence
        if r.fruit not in summary["by_fruit"]:
            summary["by_fruit"][r.fruit] = {"A": 0, "B": 0, "C": 0}
        summary["by_fruit"][r.fruit][r.grade] += 1
    summary["avg_confidence"] = round(conf_sum / len(results), 4) if results else 0.0
    summary["dominant_grade"] = max(["A", "B", "C"], key=lambda g: summary[g])
    return summary