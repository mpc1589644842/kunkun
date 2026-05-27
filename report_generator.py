"""
report_generator.py — 水果检测 PDF 报告生成模块
依赖: pip install reportlab pillow
使用: from report_generator import generate_pdf_report
"""

import io
from datetime import datetime

import numpy as np
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF

import os, sys

# ─────────────────────────────────────────────
# 字体注册（自动查找系统中文字体）
# ─────────────────────────────────────────────
def _register_fonts():
    """尝试注册中文字体，找不到则使用内置字体"""
    candidates = [
        # Windows
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("ChineseFont", path))
                pdfmetrics.registerFont(TTFont("ChineseFont-Bold", path))
                return "ChineseFont"
            except Exception:
                continue
    return "Helvetica"  # 降级：英文字体

FONT_NAME = _register_fonts()
FONT_BOLD = FONT_NAME  # reportlab ttc 同一字体含粗体

# ─────────────────────────────────────────────
# 品牌颜色
# ─────────────────────────────────────────────
C_ORANGE  = colors.HexColor("#FF8C00")
C_ORANGE2 = colors.HexColor("#FFA500")
C_CREAM   = colors.HexColor("#FFF8ED")
C_BORDER  = colors.HexColor("#F5D99A")
C_TEXT    = colors.HexColor("#3D2B1F")
C_MUTED   = colors.HexColor("#A07040")
C_GREEN   = colors.HexColor("#22C55E")
C_YELLOW  = colors.HexColor("#F59E0B")
C_RED     = colors.HexColor("#EF4444")
C_WHITE   = colors.white

GRADE_COLORS = {"A": C_GREEN, "B": C_YELLOW, "C": C_RED}
GRADE_NAMES  = {"A": "一级果", "B": "二级果", "C": "三级果"}

# ─────────────────────────────────────────────
# 样式
# ─────────────────────────────────────────────
def _styles():
    return {
        "title": ParagraphStyle("title", fontName=FONT_BOLD, fontSize=22,
                                textColor=C_ORANGE, alignment=TA_CENTER,
                                spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", fontName=FONT_NAME, fontSize=10,
                                   textColor=C_MUTED, alignment=TA_CENTER,
                                   spaceAfter=2),
        "h1": ParagraphStyle("h1", fontName=FONT_BOLD, fontSize=13,
                              textColor=C_ORANGE, spaceBefore=10, spaceAfter=5),
        "h2": ParagraphStyle("h2", fontName=FONT_BOLD, fontSize=11,
                              textColor=C_TEXT, spaceBefore=6, spaceAfter=3),
        "body": ParagraphStyle("body", fontName=FONT_NAME, fontSize=9,
                               textColor=C_TEXT, leading=15),
        "small": ParagraphStyle("small", fontName=FONT_NAME, fontSize=8,
                                textColor=C_MUTED, leading=12),
        "center": ParagraphStyle("center", fontName=FONT_NAME, fontSize=9,
                                 textColor=C_TEXT, alignment=TA_CENTER),
        "right": ParagraphStyle("right", fontName=FONT_NAME, fontSize=8,
                                textColor=C_MUTED, alignment=TA_RIGHT),
        "conclusion": ParagraphStyle("conclusion", fontName=FONT_NAME, fontSize=9,
                                     textColor=colors.HexColor("#7A4A00"),
                                     backColor=C_CREAM, leading=15,
                                     borderPad=8),
    }

# ─────────────────────────────────────────────
# 图形绘制工具
# ─────────────────────────────────────────────
def _pie_drawing(summary: dict, w=150, h=150) -> Drawing:
    d   = Drawing(w, h)
    pie = Pie()
    pie.x, pie.y   = w//2 - 50, h//2 - 50
    pie.width = pie.height = 100

    vals  = [summary.get(g, 0) for g in ["A", "B", "C"]]
    total = sum(vals) or 1
    pie.data = [max(v, 0.001) for v in vals]

    pie.labels = [
        f"Grade A: {vals[0]} ({vals[0]/total*100:.0f}%)",
        f"Grade B: {vals[1]} ({vals[1]/total*100:.0f}%)",
        f"Grade C: {vals[2]} ({vals[2]/total*100:.0f}%)",
    ]
    pie.slices[0].fillColor = C_GREEN
    pie.slices[1].fillColor = C_YELLOW
    pie.slices[2].fillColor = C_RED
    pie.slices.strokeColor  = C_WHITE
    pie.slices.strokeWidth  = 2
    for i in range(3):
        pie.slices[i].labelRadius = 1.35
        pie.slices[i].fontSize    = 7
        pie.slices[i].fontName    = "Helvetica"
    pie.sideLabels   = 0
    pie.simpleLabels = 1
    d.add(pie)
    return d


def _bar_drawing(by_fruit: dict, w=280, h=140) -> Drawing:
    if not by_fruit:
        return Drawing(w, h)
    d      = Drawing(w, h)
    fruits = list(by_fruit.keys())
    nf     = len(fruits)
    if nf == 0:
        return d

    fruit_en = {"苹果": "Apple", "香蕉": "Banana", "橙子": "Orange"}
    fruits_label = [fruit_en.get(f, f) for f in fruits]

    bc = VerticalBarChart()
    bc.x, bc.y = 40, 20
    bc.width   = w - 60
    bc.height  = h - 40

    data = []
    for g in ["A", "B", "C"]:
        data.append([by_fruit[f].get(g, 0) for f in fruits])
    bc.data = data

    bc.bars[0].fillColor = C_GREEN
    bc.bars[1].fillColor = C_YELLOW
    bc.bars[2].fillColor = C_RED
    bc.groupSpacing      = 5

    bc.categoryAxis.categoryNames    = fruits_label
    bc.categoryAxis.labels.fontName  = "Helvetica"
    bc.categoryAxis.labels.fontSize  = 7
    bc.categoryAxis.labels.angle     = 15 if nf > 3 else 0

    bc.valueAxis.valueMin        = 0
    bc.valueAxis.labels.fontName = "Helvetica"
    bc.valueAxis.labels.fontSize = 7

    d.add(bc)
    return d

    bc = VerticalBarChart()
    bc.x, bc.y = 40, 20
    bc.width   = w - 60
    bc.height  = h - 40

    data = []
    for g in ["A", "B", "C"]:
        data.append([by_fruit[f].get(g, 0) for f in fruits])
    bc.data = data

    bc.bars[0].fillColor = C_GREEN
    bc.bars[1].fillColor = C_YELLOW
    bc.bars[2].fillColor = C_RED
    bc.groupSpacing      = 5

    bc.categoryAxis.categoryNames = fruits
    bc.categoryAxis.labels.fontName  = FONT_NAME
    bc.categoryAxis.labels.fontSize  = 7
    bc.categoryAxis.labels.angle     = 15 if nf > 3 else 0

    bc.valueAxis.valueMin = 0
    bc.valueAxis.labels.fontName = FONT_NAME
    bc.valueAxis.labels.fontSize = 7

    d.add(bc)
    return d


def _kpi_table(summary: dict, latency_ms: float, model_name: str, S):
    total   = summary.get("total", 0)
    avg_c   = summary.get("avg_confidence", 0)
    dom     = summary.get("dominant_grade", "A")
    dom_col = GRADE_COLORS.get(dom, C_GREEN)

    def kpi_cell(num, label, hex_color):
        return Paragraph(
            f'<font name="{FONT_BOLD}" size="18" color="{hex_color}"><b>{num}</b></font>'
            f'<br/><font name="{FONT_NAME}" size="8" color="#A07040">{label}</font>',
            S["center"]
        )

    data = [[
        kpi_cell(str(total),           "检测目标数",  "#FF8C00"),
        kpi_cell(f"{latency_ms:.0f}ms","推理时延",    "#7C3AED"),
        kpi_cell(f"{avg_c:.3f}",       "平均置信度",  "#0EA5E9"),
        kpi_cell(f"Grade {dom}",       "主导等级",    dom_col.hexval()),
    ]]

    t = Table(data, colWidths=[38*mm]*4, rowHeights=[18*mm])
    t.setStyle(TableStyle([
        ("BOX",           (0,0),(-1,-1), 0.5, C_BORDER),
        ("INNERGRID",     (0,0),(-1,-1), 0.5, C_BORDER),
        ("BACKGROUND",    (0,0),(-1,-1), C_CREAM),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ]))
    return t

    def kpi_cell(num, label, color=C_ORANGE):
        return [Paragraph(f'<font color="{color.hexval()}" size="16"><b>{num}</b></font>', S["center"]),
                Paragraph(label, S["small"])]

    data = [
        [kpi_cell(str(total), "检测目标数"),
         kpi_cell(f"{latency_ms:.0f}ms", "推理时延", colors.HexColor("#7C3AED")),
         kpi_cell(f"{avg_c:.3f}", "平均置信度", colors.HexColor("#0EA5E9")),
         kpi_cell(f"Grade {dom}", "主导等级", dom_col)],
    ]
    t = Table(data, colWidths=[38*mm]*4)
    t.setStyle(TableStyle([
        ("BOX",        (0,0),(-1,-1), 0.5, C_BORDER),
        ("INNERGRID",  (0,0),(-1,-1), 0.5, C_BORDER),
        ("BACKGROUND", (0,0),(-1,-1), C_CREAM),
        ("ALIGN",      (0,0),(-1,-1), "CENTER"),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[C_CREAM]),
    ]))
    return t


def _grade_detail_table(grades: list, S):
    if not grades:
        return Paragraph("本次检测无目标", S["body"])

    header = ["#", "水果", "成熟度", "置信度", "等级", "分拣建议"]
    rows   = [header]
    for i, g in enumerate(grades):
        rows.append([
            str(i+1), g.fruit, g.ripeness,
            f"{g.confidence:.3f}",
            g.grade_label,
            g.action.replace("建议：","").replace("✅","").replace("⚠️","").replace("❌","").strip(),
        ])

    col_w = [8*mm, 18*mm, 20*mm, 18*mm, 32*mm, 52*mm]
    t = Table(rows, colWidths=col_w, repeatRows=1)

    style = [
        ("BACKGROUND",   (0,0),(-1,0),  C_ORANGE),
        ("TEXTCOLOR",    (0,0),(-1,0),  C_WHITE),
        ("FONTNAME",     (0,0),(-1,0),  FONT_BOLD),
        ("FONTSIZE",     (0,0),(-1,-1), 8),
        ("FONTNAME",     (0,1),(-1,-1), FONT_NAME),
        ("ALIGN",        (0,0),(-1,-1), "CENTER"),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_WHITE, C_CREAM]),
        ("BOX",          (0,0),(-1,-1), 0.5, C_BORDER),
        ("INNERGRID",    (0,0),(-1,-1), 0.3, C_BORDER),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]
    # 等级列上色
    for i, g in enumerate(grades, start=1):
        col = GRADE_COLORS.get(g.grade, C_TEXT)
        style.append(("TEXTCOLOR", (4,i),(4,i), col))
        style.append(("FONTNAME",  (4,i),(4,i), FONT_BOLD))

    t.setStyle(TableStyle(style))
    return t


def _model_compare_table(model_name: str, S):
    MODEL_DATA = [
        ("YOLOv11n PT原始",    "5.2",  "64.5", "15.5", "0.995", "2.6M", "6.3"),
        ("ONNX FP32 (本文)",   "10.1", "31.6", "31.6", "0.995", "2.6M", "6.3"),
        ("ONNX INT8 (量化)",   "3.0",  "56.9", "17.6", "—",     "2.6M", "—"),
        ("TorchScript",        "10.0", "75.7", "13.2", "0.995", "2.6M", "6.3"),
    ]
    header = ["模型", "大小(MB)", "时延(ms)", "FPS", "mAP@0.5", "参数量", "GFLOPs"]
    rows   = [header] + [list(r) for r in MODEL_DATA]

    col_w = [40*mm, 18*mm, 18*mm, 14*mm, 20*mm, 16*mm, 18*mm]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    style = [
        ("BACKGROUND",    (0,0),(-1,0), C_ORANGE),
        ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
        ("FONTNAME",      (0,0),(-1,0), FONT_BOLD),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("FONTNAME",      (0,1),(-1,-1), FONT_NAME),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_CREAM]),
        ("BOX",           (0,0),(-1,-1), 0.5, C_BORDER),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, C_BORDER),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        # 高亮 ONNX FP32 行
        ("BACKGROUND",    (0,2),(-1,2), colors.HexColor("#FFF3E0")),
        ("FONTNAME",      (0,2),(-1,2), FONT_BOLD),
        ("TEXTCOLOR",     (2,2),(3,2),  C_GREEN),
    ]
    t.setStyle(TableStyle(style))
    return t


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def generate_pdf_report(
    grades:      list,
    summary:     dict,
    annotated_img: np.ndarray | None,
    latency_ms:  float,
    model_name:  str,
    source_name: str = "未知来源",
    history:     list = None,
) -> bytes:
    """
    生成完整PDF检测报告

    Parameters
    ----------
    grades        : list[GradeResult]  本次检测的分级结果列表
    summary       : dict               summarize_grades() 返回值
    annotated_img : np.ndarray | None  标注后的BGR图像（可为None）
    latency_ms    : float              推理时延
    model_name    : str                使用的模型名称
    source_name   : str                检测来源（文件名）
    history       : list               历史记录列表（可选）

    Returns
    -------
    bytes  PDF 文件的字节内容，可直接用于 st.download_button
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm,  bottomMargin=16*mm,
    )
    W = A4[0] - 36*mm   # 可用宽度
    S = _styles()
    story = []

    # ── 封面头部 ──────────────────────────────
    story.append(Paragraph("🍎 水果成熟度智能检测报告", S["title"]))
    story.append(Paragraph("基于改进轻量化 YOLOv11 · 三级分拣决策系统", S["subtitle"]))
    story.append(Spacer(1, 2*mm))
    story.append(HRFlowable(width="100%", thickness=2, color=C_ORANGE))
    story.append(Spacer(1, 3*mm))

    # 基本信息行
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    info_data = [
        [Paragraph(f"检测时间：{now}", S["body"]),
         Paragraph(f"检测来源：{source_name}", S["body"]),
         Paragraph(f"使用模型：{model_name}", S["body"])],
    ]
    info_t = Table(info_data, colWidths=[W/3]*3)
    info_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), C_CREAM),
        ("BOX",        (0,0),(-1,-1), 0.5, C_BORDER),
        ("INNERGRID",  (0,0),(-1,-1), 0.5, C_BORDER),
        ("TOPPADDING", (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    story.append(info_t)
    story.append(Spacer(1, 5*mm))

    # ── 一、检测概览 ──────────────────────────
    story.append(Paragraph("一、检测概览", S["h1"]))
    story.append(_kpi_table(summary, latency_ms, model_name, S))
    story.append(Spacer(1, 4*mm))

    # ── 二、检测结果图 + 分级饼图 ────────────
    story.append(Paragraph("二、检测结果可视化", S["h1"]))

    row_items = []
    # 标注图
    if annotated_img is not None:
        try:
            from PIL import Image as PILImg
            import cv2
            rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            pil = PILImg.fromarray(rgb)
            img_buf = io.BytesIO()
            pil.save(img_buf, format="PNG")
            img_buf.seek(0)
            rl_img = RLImage(img_buf, width=90*mm, height=68*mm)
            row_items.append(rl_img)
        except Exception:
            row_items.append(Paragraph("（图像加载失败）", S["small"]))
    else:
        row_items.append(Paragraph("（无检测图像）", S["small"]))

    # 饼图
    pie_d = _pie_drawing(summary, w=180, h=160)
    row_items.append(pie_d)

    img_row = Table([row_items], colWidths=[95*mm, W-95*mm])
    img_row.setStyle(TableStyle([
        ("VALIGN",  (0,0),(-1,-1), "MIDDLE"),
        ("ALIGN",   (0,0),(-1,-1), "CENTER"),
        ("TOPPADDING",(0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
    ]))
    story.append(img_row)
    story.append(Spacer(1, 3*mm))

    # 按品类柱状图
    if summary.get("by_fruit"):
        story.append(Paragraph("各品类分级数量", S["h2"]))
        story.append(_bar_drawing(summary["by_fruit"], w=int(W*2.83), h=130))
        story.append(Spacer(1, 3*mm))

    # ── 三、逐目标分拣建议 ────────────────────
    story.append(Paragraph("三、逐目标分拣建议", S["h1"]))
    story.append(_grade_detail_table(grades, S))
    story.append(Spacer(1, 3*mm))

    # 分级汇总
    total = summary.get("total", 0)
    sum_rows = [
        ["等级", "数量", "占比", "处置建议"],
        ["一级果 (Grade A)", str(summary.get("A",0)),
         f"{summary.get('A',0)/total*100:.1f}%" if total else "0%",
         "直接上市 / 礼盒装"],
        ["二级果 (Grade B)", str(summary.get("B",0)),
         f"{summary.get('B',0)/total*100:.1f}%" if total else "0%",
         "超市散装 / 批发渠道"],
        ["三级果 (Grade C)", str(summary.get("C",0)),
         f"{summary.get('C',0)/total*100:.1f}%" if total else "0%",
         "降级加工 / 废弃"],
        ["合计", str(total), "100%", "—"],
    ]
    sum_t = Table(sum_rows, colWidths=[35*mm, 20*mm, 20*mm, W-75*mm])
    sum_style = [
        ("BACKGROUND",    (0,0),(-1,0), C_ORANGE),
        ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
        ("FONTNAME",      (0,0),(-1,0), FONT_BOLD),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("FONTNAME",      (0,1),(-1,-1), FONT_NAME),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_CREAM]),
        ("BOX",           (0,0),(-1,-1), 0.5, C_BORDER),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, C_BORDER),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("BACKGROUND",    (0,4),(-1,4), C_CREAM),
        ("FONTNAME",      (0,4),(-1,4), FONT_BOLD),
        # 等级列颜色
        ("TEXTCOLOR",     (0,1),(0,1), C_GREEN),
        ("TEXTCOLOR",     (0,2),(0,2), C_YELLOW),
        ("TEXTCOLOR",     (0,3),(0,3), C_RED),
    ]
    sum_t.setStyle(TableStyle(sum_style))
    story.append(Spacer(1, 3*mm))
    story.append(sum_t)
    story.append(Spacer(1, 4*mm))

    # ── 四、模型性能对比 ──────────────────────
    story.append(Paragraph("四、模型性能对比", S["h1"]))
    story.append(_model_compare_table(model_name, S))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "结论：ONNX FP32 部署方案在保持 mAP@0.5=0.995 精度完全不损失的前提下，"
        "推理时延由 64.5ms 降低至 31.6ms，加速比 2.04x，FPS 由 15.5 提升至 31.6，"
        "是精度与速度最优平衡的部署选择。",
        S["conclusion"],
    ))
    story.append(Spacer(1, 4*mm))

    # ── 五、历史检测汇总（可选）──────────────
    if history and len(history) > 1:
        story.append(Paragraph("五、本次会话检测汇总", S["h1"]))
        hist_header = ["时间", "来源", "目标数", "一级(A)", "二级(B)", "三级(C)", "时延(ms)"]
        hist_rows   = [hist_header]
        for r in history[:15]:  # 最多15条
            hist_rows.append([
                r.get("time","—"), r.get("filename","—"),
                str(r.get("total",0)), str(r.get("A",0)),
                str(r.get("B",0)), str(r.get("C",0)),
                f"{r.get('latency',0):.0f}",
            ])
        col_w_h = [18*mm, 45*mm, 18*mm, 18*mm, 18*mm, 18*mm, 18*mm]
        ht = Table(hist_rows, colWidths=col_w_h, repeatRows=1)
        ht.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), C_ORANGE),
            ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
            ("FONTNAME",      (0,0),(-1,0), FONT_BOLD),
            ("FONTSIZE",      (0,0),(-1,-1), 8),
            ("FONTNAME",      (0,1),(-1,-1), FONT_NAME),
            ("ALIGN",         (0,0),(-1,-1), "CENTER"),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_CREAM]),
            ("BOX",           (0,0),(-1,-1), 0.5, C_BORDER),
            ("INNERGRID",     (0,0),(-1,-1), 0.3, C_BORDER),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
        ]))
        story.append(ht)
        story.append(Spacer(1, 3*mm))

    # ── 页脚说明 ──────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=C_BORDER))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f"本报告由「水果成熟度智能检测与分级系统」自动生成 · 生成时间：{now} · "
        f"模型：YOLOv11n · mAP@0.5=0.995",
        S["small"],
    ))

    doc.build(story)
    return buf.getvalue()
