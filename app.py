"""
app.py — 水果成熟度智能检测与分级系统 v2.0
新增功能：
  ① 检测历史记录（Session内保存每次结果）
  ② 数据统计报告（可导出CSV）
  ③ 模型性能对比面板
  ④ 货架分拣看板（模拟流水线）
运行: python -m streamlit run app.py --browser.gatherUsageStats false
"""

import io
import time
import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from ultralytics import YOLO

from grading import classify_grade, summarize_grades, GRADE_CONFIG
from report_generator import generate_pdf_report

# 业务模型
from models import User, UserRole, InboundBatch, BatchStatus
from services.batch_service import apply_detection_result
from db.repositories import batch_repo as _batch_repo   # Step 9.2.2.c.2
from pages.login_page import render_login_page

# ─────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="水果入库盘点与品质分级系统",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 登录守卫:未登录则只显示登录页
# ─────────────────────────────────────────────
if "current_user" not in st.session_state:
    render_login_page()
    st.stop()   # 关键:阻止后续 UI 渲染

# 当前用户(后续逻辑可直接使用)
current_user: User = st.session_state.current_user

# ─────────────────────────────────────────────
# Session State 初始化（历史记录 & 看板数据）
# ─────────────────────────────────────────────
if "history"       not in st.session_state: st.session_state.history       = []
if "board_total"   not in st.session_state: st.session_state.board_total   = {"A":0,"B":0,"C":0}
if "board_fruits"  not in st.session_state: st.session_state.board_fruits  = {}
if "board_records" not in st.session_state: st.session_state.board_records = []
# ── Step 4.0: 批次池(支持复核中心列出待复核批次)──
if "all_batches"           not in st.session_state: st.session_state.all_batches           = []
if "suppliers"             not in st.session_state: st.session_state.suppliers             = {}
if "selected_review_batch" not in st.session_state: st.session_state.selected_review_batch = None
# ── Step 5.3: file_uploader 版本号(用于一键清空)──
if "uploader_version" not in st.session_state: st.session_state.uploader_version = 0

# ── Step 7.3 → 9.2.2.b: 反推函数已停用 ──
# SQLite 持久化后,suppliers 数据本身就持久存在,不需要从 batch 反推。
# session_state.suppliers 仍保留初始化(避免现有引用报错),但变成"无用 dict"。
# 业务层从 db.repositories.supplier_repo 读取真实数据。
# (保留原函数定义仅作 git diff 参考,运行时不调用)
def _backfill_suppliers_from_batches():
    pass  # no-op


# ── Step 9.2.2.c.3: 启动时从 SQLite 反推批次到 session_state ──
# 设计:每次 rerun 都从 DB 重新加载活跃批次,让 session_state 充当
# "DB 的内存缓存层"。开销:SQLite 单文件查询毫秒级,毕设量级完全可接受。
# 也同时刷新 current_batch — 让其他用户(manager 复核)的修改能被 operator 看到。
def _load_batches_from_db():
    _all = _batch_repo.list_all(include_draft=False)
    st.session_state.all_batches = _all

    # 同步刷新 current_batch:如果选中的批次在 DB 里有新版本,用新版本替换
    _cur = st.session_state.get("current_batch")
    if _cur is not None and _cur.batch_id:
        _latest = _batch_repo.find_by_id(_cur.batch_id)
        if _latest is not None:
            st.session_state.current_batch = _latest
        # 如果 DB 里找不到(被删了等情况),保留原 current_batch 不动

_load_batches_from_db()
# ─────────────────────────────────────────────
# 侧边栏:当前用户 + 登出
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="background:#FFF8ED;border-radius:12px;padding:0.8rem 1rem;
                border:1.5px solid #F5D99A;margin-bottom:0.8rem">
        <div style="font-size:0.7rem;color:#A07040;font-weight:600">当前用户</div>
        <div style="font-size:1rem;font-weight:800;color:#C05C00;margin-top:2px">
            {current_user.full_name}
        </div>
        <div style="font-size:0.75rem;color:#6B4F2F;margin-top:3px">
            <span style="background:#FF8C00;color:white;padding:1px 8px;
                         border-radius:50px;font-weight:700;font-size:0.7rem">
                {current_user.role.display_name}
            </span>
            &nbsp;@{current_user.username}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 登出", use_container_width=True):
        del st.session_state.current_user
        st.rerun()

    st.markdown("---")
    st.caption(f"📞 {current_user.phone}")

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Noto+Sans+SC:wght@400;500;700&display=swap');

.stApp { background:#FDFAF5; font-family:'Noto Sans SC','Nunito',sans-serif; }

/* ── UI 大修阶段 1: 主容器限宽 + 居中(Streamlit 1.55+ 选择器)── */
.stMainBlockContainer, .main .block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1400px !important;
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    margin: 0 auto !important;
}

/* ── UI 大修阶段 2 收尾: 右列实时预览面板 sticky 粘顶 ── */
/* Streamlit 1.55+ 列容器是 [data-testid="stColumn"],第二个列是右列 */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) {
    position: sticky !important;
    top: 1rem !important;
    align-self: flex-start !important;
}
/* 让两列容器允许溢出可见,否则 sticky 失效 */
[data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
}

/* ── UI 大修阶段 1: 顶栏紧凑化 + 元素间距收紧 ── */
.element-container { margin-bottom: 0.4rem !important; }
[data-testid="stHorizontalBlock"] { gap: 0.6rem !important; }
[data-testid="stVerticalBlock"] { gap: 0.55rem !important; }

/* ── UI 大修阶段 1: Tab 视觉强化(active 加下划线)── */
.stTabs [data-baseweb="tab-list"] {
    border-radius: 50px !important;
    padding: 5px !important;
    border: 1.5px solid #F5D99A !important;
    background: linear-gradient(135deg, #FFF3E0 0%, #FFEFD5 100%) !important;
    box-shadow: 0 2px 8px rgba(200,130,0,.06) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 50px !important;
    padding: 0.45rem 1.2rem !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    transition: all 0.18s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 140, 0, 0.08) !important;
    color: #C05C00 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #FF8C00, #FFA500) !important;
    color: white !important;
    box-shadow: 0 3px 10px rgba(255, 140, 0, 0.35) !important;
}

section[data-testid="stSidebar"] {
    background:#FFFFFF; border-right:2px solid #F0EAD6;
}
section[data-testid="stSidebar"] * { color:#3D2B1F !important; }
.stApp, .stApp p, .stApp span, .stApp label { color:#3D2B1F; }
#MainMenu,footer,header{ visibility:hidden; }
.stDeployButton{ display:none; }

/* 英雄区 */
.hero-wrap {
    background:linear-gradient(135deg,#FFF8ED 0%,#FFF0D6 50%,#FFE8C2 100%);
    border-radius:24px; padding:1.8rem 2.5rem; margin-bottom:1.2rem;
    border:1.5px solid #F5D99A; position:relative; overflow:hidden;
}
.hero-wrap::before {
    content:'🍎🍌🍊'; position:absolute; right:2rem; top:50%;
    transform:translateY(-50%); font-size:3.2rem;
    letter-spacing:.5rem; opacity:.2;
}
.hero-title {
    font-family:'Nunito','Noto Sans SC',sans-serif;
    font-size:1.9rem; font-weight:900; color:#C05C00; margin:0 0 .25rem; line-height:1.2;
}
.hero-sub { font-size:.88rem; color:#A07040; font-weight:500; margin:0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background:#FFF3E0; border-radius:50px; padding:4px; gap:2px;
    border:1.5px solid #F5D99A;
}
.stTabs [data-baseweb="tab"] {
    border-radius:50px !important; padding:.4rem 1.1rem !important;
    font-weight:700 !important; color:#A07040 !important;
    background:transparent !important; font-size:.85rem !important;
}
.stTabs [aria-selected="true"] { background:#FF8C00 !important; color:white !important; }

/* 卡片 */
.kpi-row { display:flex; gap:10px; margin:.8rem 0; }
.kpi-card {
    flex:1; background:#FFFFFF; border-radius:16px; padding:.9rem 1rem;
    text-align:center; border:1.5px solid #F0EAD6;
    box-shadow:0 2px 10px rgba(200,130,0,.07);
    transition:transform .15s,box-shadow .15s;
}
.kpi-card:hover { transform:translateY(-2px); box-shadow:0 5px 18px rgba(200,130,0,.13); }
.kpi-num  { font-size:1.8rem; font-weight:900; line-height:1.1; }
.kpi-lbl  { font-size:.72rem; color:#A07040; margin-top:3px; font-weight:600; }

/* 分拣建议 */
.action-card {
    background:#FFFFFF; border-radius:12px; padding:.8rem 1rem; margin:.4rem 0;
    border:1.5px solid #F0EAD6; border-left-width:5px;
    box-shadow:0 2px 8px rgba(0,0,0,.04); font-size:.86rem; color:#3D2B1F;
}
.badge {
    display:inline-block; padding:2px 11px; border-radius:50px;
    font-weight:800; font-size:.78rem; color:white; vertical-align:middle;
}

/* 历史记录行 */
.hist-row {
    background:#FFFFFF; border-radius:12px; padding:.7rem 1rem; margin:.35rem 0;
    border:1.5px solid #F0EAD6; font-size:.84rem; color:#3D2B1F;
    display:flex; justify-content:space-between; align-items:center;
}
.hist-time { color:#A07040; font-size:.76rem; }

/* 看板大卡 */
.board-card {
    background:#FFFFFF; border-radius:20px; padding:1.5rem;
    text-align:center; border:2px solid; box-shadow:0 4px 20px rgba(0,0,0,.07);
}
.board-num  { font-size:3.5rem; font-weight:900; line-height:1; }
.board-lbl  { font-size:.9rem; font-weight:700; margin-top:.4rem; }
.board-pct  { font-size:.8rem; color:#A07040; margin-top:.2rem; }

/* 模型对比表格 */
.cmp-table {
    width:100%; border-collapse:collapse; font-size:.86rem;
    border-radius:12px; overflow:hidden;
}
.cmp-table th {
    background:#FFF3E0; color:#C05C00; font-weight:800;
    padding:.6rem .9rem; text-align:center; border-bottom:2px solid #F5D99A;
}
.cmp-table td {
    padding:.55rem .9rem; text-align:center;
    border-bottom:1px solid #F5EDD8; color:#3D2B1F;
}
.cmp-table tr:hover td { background:#FFFBF0; }
.cmp-table .highlight td { background:#FFF8ED; font-weight:700; }
.cmp-best { color:#22C55E; font-weight:900; }
.cmp-tag {
    display:inline-block; background:#FF8C00; color:white;
    border-radius:50px; padding:1px 8px; font-size:.72rem; font-weight:700;
}

/* 上传区 */
[data-testid="stFileUploader"] {
    border:2.5px dashed #F5B800 !important; border-radius:16px !important;
    background:#FFFBF0 !important; padding:1rem !important;
}

/* 按钮 */
.stButton>button {
    background:linear-gradient(135deg,#FF8C00,#FFA500) !important;
    color:white !important; border:none !important; border-radius:50px !important;
    font-weight:700 !important; padding:.45rem 1.6rem !important;
    box-shadow:0 3px 12px rgba(255,140,0,.3) !important;
    transition:all .2s !important;
}
.stButton>button:hover {
    transform:translateY(-1px) !important;
    box-shadow:0 6px 18px rgba(255,140,0,.4) !important;
}

/* selectbox & slider */
[data-baseweb="select"]>div {
    background:#FFFBF0 !important; border-color:#F5D99A !important; border-radius:10px !important;
}
.stProgress>div>div>div>div {
    background:linear-gradient(90deg,#FF8C00,#FFC300) !important;
}
hr { border-color:#F0EAD6 !important; margin:.8rem 0 !important; }

/* 分级说明 */
.grade-info {
    background:#FFFBF0; border-radius:10px; padding:.65rem .85rem;
    margin:.35rem 0; border-left:4px solid; font-size:.8rem;
}

/* 空状态 */
.empty-state {
    height:260px; display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    background:#FFFBF0; border-radius:16px;
    border:2px dashed #F5D99A; color:#C8A060;
}
/* 检测参数设置 expander 缩小 */
[data-testid="stExpander"] {
    font-size:0.75rem !important;
}
[data-testid="stExpander"] summary {
    padding:0.3rem 0.8rem !important;
    font-size:0.75rem !important;
    color:#A07040 !important;
}
[data-testid="stExpander"] summary:hover {
    color:#FF8C00 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
# 找到WEIGHT_OPTIONS定义
WEIGHT_OPTIONS = {
    "YOLOv11n (原始 PT)":  "runs/detect/runs/detect/v2_16class/weights/best.pt",
    "ONNX FP32 ⚡":        "exported_models/model_fp32.onnx",
    "ONNX INT8 (量化)":    "exported_models/model_int8.onnx",
    "TorchScript":         "exported_models/model.torchscript",
}

# 模型性能数据（论文实验结果）
MODEL_PERF = {
    "YOLOv11n (原始 PT)":  {"size_mb":15.3, "latency_ms":64.4, "fps":15.5, "map":0.971, "params":"2.6M", "gflops":6.3},
    "ONNX FP32 ⚡":        {"size_mb":10.1, "latency_ms":28.0, "fps":35.7, "map":0.971, "params":"2.6M", "gflops":6.3},
    "ONNX INT8 (量化)":    {"size_mb":3.0,  "latency_ms":57.3, "fps":17.4, "map":"—",   "params":"2.6M", "gflops":"—"},
    "TorchScript":         {"size_mb":10.0, "latency_ms":74.8, "fps":13.4, "map":0.971, "params":"2.6M", "gflops":6.3},
}

CHART_BG   = "#FFFBF0"
CHART_FONT = "#3D2B1F"

@st.cache_resource(show_spinner="🍊 加载模型中...")
def load_model(path: str) -> YOLO:
    model = YOLO(path)
    # ── 预热：消除首次推理的冷启动延迟 ──
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(source=dummy, verbose=False, device="cpu")
    return model

def run_inference(model, source, conf, iou):
    return model.predict(source=source, conf=conf, iou=iou, verbose=False, device="cpu")

def results_to_detections(results) -> list[dict]:
    dets = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            dets.append({
                "class_id":   int(box.cls.item()),
                "confidence": round(float(box.conf.item()), 4),
                "xyxy":       box.xyxy[0].tolist(),
            })
    return dets
def cross_class_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """
    跨类别 NMS：当同一位置出现多个不同类别的检测框时，只保留置信度最高的那个。

    业务场景：
    - 富士苹果同时被识别为 ripe_apple + unripe_persimmon(橙子) → 保留 conf 高的
    - 同一水果同时触发 ripe_xxx + rotten_fruit → 保留 conf 高的

    与 YOLO 默认 NMS 的区别:YOLO 默认是按 class 分组做 NMS,跨类不去重。
    """
    if not detections:
        return []

    # 按置信度从高到低排序
    sorted_dets = sorted(detections, key=lambda d: -d["confidence"])
    kept = []

    for det in sorted_dets:
        ax1, ay1, ax2, ay2 = det["xyxy"]
        suppressed = False
        for k in kept:
            bx1, by1, bx2, by2 = k["xyxy"]
            # 计算 IOU
            ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
            iw  = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
            inter = iw * ih
            area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
            iou = inter / (area_a + area_b - inter)

            if iou >= iou_threshold:
                suppressed = True
                break

        if not suppressed:
            kept.append(det)

    return kept
def annotate_image(orig: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    在原图上绘制检测框 + 中文等级标签。
    标签格式: "苹果 成熟 [A] 0.85"
    注意：不使用彩色 emoji（🟢🟡🔴），中文字体不含 emoji glyph 会显示为 □。
    """
    from PIL import ImageFont, ImageDraw

    img = orig.copy()
    grade_bgr = {"A": (74, 197, 34), "B": (11, 158, 245), "C": (68, 68, 239)}

    # ── 加载中文字体（只加载一次，所有目标共用）──
    font = None
    for fc in [
        "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc", # 宋体
        "C:/Windows/Fonts/simhei.ttf", # 黑体
    ]:
        try:
            font = ImageFont.truetype(fc, 20)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.truetype("arial.ttf", 20)

    # ── 先画所有矩形框（用 cv2，速度快）──
    for det in detections:
        gr = classify_grade(det["class_id"], det["confidence"])
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        color = grade_bgr[gr.grade]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    # ── 再统一用 PIL 绘制所有中文标签（一次转换，避免每个目标都来回转）──
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)

    for det in detections:
        gr = classify_grade(det["class_id"], det["confidence"])
        x1, y1, _, _ = [int(v) for v in det["xyxy"]]
        color = grade_bgr[gr.grade]

        # 用 [A]/[B]/[C] 替代彩色 emoji，避免字体不支持导致的 □ 方块
        label = f"{gr.fruit} {gr.ripeness} [{gr.grade}] {gr.confidence:.2f}"

        bbox = draw.textbbox((x1, y1 - 28), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 28), label, font=font, fill=(255, 255, 255))

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

def add_to_history(filename: str, detections: list, grades: list, latency: float, model_name: str):
    summary = summarize_grades(grades)
    st.session_state.history.insert(0, {
        "time":      datetime.now().strftime("%H:%M:%S"),
        "filename":  filename,
        "total":     summary["total"],
        "A":         summary["A"],
        "B":         summary["B"],
        "C":         summary["C"],
        "avg_conf":  summary["avg_confidence"],
        "latency":   round(latency, 1),
        "model":     model_name,
    })
    if len(st.session_state.history) > 20:
        st.session_state.history.pop()

def add_to_board(grades: list, filename: str):
    summary = summarize_grades(grades)
    for g in ["A","B","C"]:
        st.session_state.board_total[g] += summary[g]
    for fruit, gcnt in summary["by_fruit"].items():
        if fruit not in st.session_state.board_fruits:
            st.session_state.board_fruits[fruit] = {"A":0,"B":0,"C":0}
        for g in ["A","B","C"]:
            st.session_state.board_fruits[fruit][g] += gcnt.get(g,0)
    st.session_state.board_records.append({
        "时间":   datetime.now().strftime("%H:%M:%S"),
        "来源":   filename,
        "一级(A)": summary["A"],
        "二级(B)": summary["B"],
        "三级(C)": summary["C"],
        "总数":   summary["total"],
    })


# ─────────────────────────────────────────────
# 图表函数
# ─────────────────────────────────────────────
def chart_pie(summary):
    fig = go.Figure(go.Pie(
        labels=["一级果 A","二级果 B","三级果 C"],
        values=[summary["A"],summary["B"],summary["C"]],
        marker=dict(colors=["#22C55E","#F59E0B","#EF4444"],
                    line=dict(color="#FFFFFF",width=3)),
        hole=.52, textinfo="label+percent",
        textfont=dict(size=12,color=CHART_FONT),
    ))
    fig.update_layout(paper_bgcolor=CHART_BG,plot_bgcolor=CHART_BG,
                      font_color=CHART_FONT,showlegend=False,
                      margin=dict(t=10,b=10,l=10,r=10),height=230)
    return fig

def chart_bar(summary):
    fruits = list(summary["by_fruit"].keys())
    colors = {"A":"#22C55E","B":"#F59E0B","C":"#EF4444"}
    fig = go.Figure()
    for g in ["A","B","C"]:
        fig.add_trace(go.Bar(
            name=f"Grade {g}", x=fruits,
            y=[summary["by_fruit"][f].get(g,0) for f in fruits],
            marker_color=colors[g],
            marker_line=dict(color="#FFFFFF",width=1.5),
        ))
    fig.update_layout(barmode="group",paper_bgcolor=CHART_BG,plot_bgcolor=CHART_BG,
                      font_color=CHART_FONT,xaxis=dict(gridcolor="#F0EAD6"),
                      yaxis=dict(gridcolor="#F0EAD6"),
                      legend=dict(bgcolor=CHART_BG,bordercolor="#F0EAD6"),
                      margin=dict(t=10,b=30,l=30,r=10),height=230)
    return fig

def chart_conf(detections):
    confs = [d["confidence"] for d in detections]
    fig = go.Figure(go.Histogram(x=confs,nbinsx=20,
                                  marker_color="#FF8C00",opacity=.85,
                                  marker_line=dict(color="#FFFFFF",width=1)))
    fig.update_layout(paper_bgcolor=CHART_BG,plot_bgcolor=CHART_BG,
                      font_color=CHART_FONT,
                      xaxis=dict(title="置信度",gridcolor="#F0EAD6"),
                      yaxis=dict(title="数量",gridcolor="#F0EAD6"),
                      margin=dict(t=10,b=30,l=30,r=10),height=200)
    return fig

def chart_history_trend():
    h = st.session_state.history[::-1]
    if len(h) < 2: return None
    fig = go.Figure()
    for g,color in [("A","#22C55E"),("B","#F59E0B"),("C","#EF4444")]:
        fig.add_trace(go.Scatter(
            x=[r["time"] for r in h],
            y=[r[g] for r in h],
            name=f"Grade {g}", mode="lines+markers",
            line=dict(color=color,width=2),
            marker=dict(size=7),
        ))
    fig.update_layout(paper_bgcolor=CHART_BG,plot_bgcolor=CHART_BG,
                      font_color=CHART_FONT,
                      xaxis=dict(gridcolor="#F0EAD6"),
                      yaxis=dict(gridcolor="#F0EAD6"),
                      legend=dict(bgcolor=CHART_BG),
                      margin=dict(t=10,b=30,l=30,r=10),height=220)
    return fig

def chart_board_pie():
    bt = st.session_state.board_total
    total = sum(bt.values())
    if total == 0: return None
    fig = go.Figure(go.Pie(
        labels=["一级果 A","二级果 B","三级果 C"],
        values=[bt["A"],bt["B"],bt["C"]],
        marker=dict(colors=["#22C55E","#F59E0B","#EF4444"],
                    line=dict(color="#FFFFFF",width=3)),
        hole=.55,textinfo="label+percent",
        textfont=dict(size=13,color=CHART_FONT),
    ))
    fig.update_layout(paper_bgcolor=CHART_BG,plot_bgcolor=CHART_BG,
                      font_color=CHART_FONT,showlegend=False,
                      margin=dict(t=10,b=10,l=10,r=10),height=260)
    return fig


# ── 顶部导航栏 ──────────────────────────────
st.markdown("""
<div class="top-nav">
    <div class="top-nav-left">
        <div class="top-nav-logo">🍎</div>
        <div>
            <div class="top-nav-title">水果入库盘点与品质分级系统</div>
            <div class="top-nav-sub">帮助经销商快速完成入库盘点 · 品质分级 · 报表存档</div>
        </div>
    </div>
    '<div class="top-nav-right">基于轻量化 YOLOv11 &nbsp;|&nbsp; mAP@0.5 = 0.971</div>'
</div>
""", unsafe_allow_html=True)

# ── 参数控制栏（替代侧栏）──────────────────
# ── Step 4.5: 用户信息 + 登出入口(顶部紧凑条)──
_uc1, _uc2, _uc3 = st.columns([6, 2, 1])
with _uc1:
    st.markdown(
        f'<div style="background:#FFF8ED;border-radius:10px;'
        f'padding:0.45rem 0.9rem;border:1.5px solid #F5D99A;'
        f'font-size:0.85rem;color:#3D2B1F">'
        f'<span style="font-size:0.7rem;color:#A07040;font-weight:600">当前用户</span>'
        f'  &nbsp; <b style="color:#C05C00">{current_user.full_name}</b>'
        f'  &nbsp; <span style="background:#FF8C00;color:white;padding:1px 8px;'
        f'border-radius:50px;font-weight:700;font-size:0.72rem">'
        f'{current_user.role.display_name}</span>'
        f'  &nbsp; <span style="color:#A07040;font-size:0.78rem">'
        f'@{current_user.username}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
with _uc3:
    if st.button("🚪 登出", use_container_width=True, key="btn_logout_top"):
        # 只清当前用户身份和该用户的 UI 选中状态
        # 保留 all_batches(共享数据,模拟 SQLite 表)
        # 这样 operator 创建的批次,manager 登录后能在复核中心看到
        del st.session_state.current_user
        for _k in ["current_batch", "selected_review_batch"]:
            if _k in st.session_state:
                del st.session_state[_k]
        st.rerun()

with st.expander("⚙️ 检测参数设置", expanded=False):
    pc1, pc2, pc3 = st.columns([2, 2, 3])
    with pc1:
        model_name = st.selectbox("模型版本", list(WEIGHT_OPTIONS.keys()), index=0)
        if "INT8" in model_name:
            st.warning("⚠️ INT8量化模型精度略有损失，建议日常使用 ONNX FP32 版本。")
    with pc2:
        conf_thres = st.slider("置信度阈值", .10, .95, .50, .05)
        iou_thres  = st.slider("NMS IoU 阈值", .10, .90, .45, .05)
    with pc3:
        st.markdown("**分级规则**")
        for color, title, desc in [
            ("#22C55E", "🟢 A级优品",    "成熟度佳 · 建议直接上架销售"),
            ("#F59E0B", "🟡 B级普通果",  "成熟度不足 · 建议低价促销或继续催熟"),
            ("#EF4444", "🔴 C级次品",    "存在腐烂 · 建议人工抽检复核后处理"),
        ]:
            st.markdown(
                f'<div class="grade-info" style="border-color:{color}">'
                f'<b>{title}</b>：<span style="color:#A07040">{desc}</span>'
                f'</div>', unsafe_allow_html=True)

model_path = WEIGHT_OPTIONS[model_name]
if not Path(model_path).exists():
    st.error(f"权重未找到：`{model_path}`")
    st.stop()
model = load_model(model_path)

# 7个标签页
# ── Step 4.3a: 动态 tabs 列表(按 role 生成)──
_tab_specs = [("batch", "📦 入库检测")]
if current_user.role in {UserRole.MANAGER, UserRole.ADMIN}:
    _tab_specs.append(("review", "🔍 复核中心"))
# Step 7.5: 供应商档案(仅 admin 可见)
if current_user.role == UserRole.ADMIN:
    _tab_specs.append(("supplier", "📇 供应商档案"))
_tab_specs.extend([
    ("img",             "📷 单张检测"),
    ("video",           "🎬 视频巡检"),
    ("cam",             "📹 摄像头"),
    ("inbound_history", "📜 入库历史"),
    ("history",         "🔍 检测日志"),
    ("board",           "📊 本批汇总"),
    ("compare",         "⚙️ 系统设置"),
])
_tabs = st.tabs([_label for _, _label in _tab_specs])
T = {_key: _tabs[_i] for _i, (_key, _) in enumerate(_tab_specs)}


# ══════════════════════════════════════════════
# Tab1: 图片检测
# ══════════════════════════════════════════════
with T["img"]:
    col_l, col_r = st.columns([1,1], gap="large")
    with col_l:
        st.markdown(
            '<div style="font-size:1.1rem;font-weight:900;color:#C05C00;margin-bottom:0.2rem">📷 单张图片检测</div>',
            unsafe_allow_html=True
        )
        uploaded = st.file_uploader("", type=["jpg","jpeg","png","bmp","webp"],
                                    label_visibility="collapsed")
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="原始图片", use_container_width=True)

    with col_r:
        if uploaded:
            img_np  = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            with st.spinner("🔍 检测中..."):
                t0      = time.perf_counter()
                results = run_inference(model, img_bgr, conf_thres, iou_thres)
                latency = (time.perf_counter()-t0)*1000
            dets = results_to_detections(results)
            dets = cross_class_nms(dets, iou_threshold=0.5)
            grades = [classify_grade(d["class_id"], d["confidence"]) for d in dets]
            summary = summarize_grades(grades)

            ann_bgr = annotate_image(img_bgr, dets)
            ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, caption="检测结果", use_container_width=True)

            # ── 零目标警告:模型未识别任何水果时,提示人工复核 ──
            # 防止用户误以为"无目标 = 一切正常"。常见原因:背景复杂、
            # 严重腐烂超出训练分布、光照过差、目标过小等。
            if not dets:
                st.error(
                    "⚠️ **未检测到任何水果目标** — 系统无法对该图片做出判断。\n\n"
                    "**可能原因**:图像质量问题、严重腐烂超出训练分布、背景过于复杂。\n\n"
                    "**建议操作**:**必须由人工复核**,不可凭本系统结果直接入库。"
                )

            # 自动存入历史和看板
            add_to_history(uploaded.name, dets, grades, latency, model_name)
            add_to_board(grades, uploaded.name)

            dom = GRADE_CONFIG[summary["dominant_grade"]]
            st.markdown(
                f'<div class="kpi-row">'
                f'<div class="kpi-card"><div class="kpi-num" style="color:#FF8C00">{summary["total"]}</div><div class="kpi-lbl">检测目标</div></div>'
                f'<div class="kpi-card"><div class="kpi-num" style="color:#7C3AED">{latency:.0f}<span style="font-size:.9rem">ms</span></div><div class="kpi-lbl">推理时延</div></div>'
                f'<div class="kpi-card"><div class="kpi-num" style="color:#0EA5E9">{summary["avg_confidence"]:.3f}</div><div class="kpi-lbl">平均置信度</div></div>'
                f'<div class="kpi-card"><div class="kpi-num" style="color:{dom["color"]}">{summary["dominant_grade"]}</div><div class="kpi-lbl">主导等级</div></div>'
                f'</div>', unsafe_allow_html=True)

            if dets:
                st.markdown("---")
                gc1,gc2 = st.columns(2)
                with gc1:
                    st.markdown("**分级占比**")
                    st.plotly_chart(chart_pie(summary),  use_container_width=True, key="img_pie")
                with gc2:
                    st.markdown("**置信度分布**")
                    st.plotly_chart(chart_conf(dets),    use_container_width=True, key="img_conf")

                st.markdown("**逐目标分拣建议**")
                for i,(det,gr) in enumerate(zip(dets,grades)):
                    st.markdown(
                        f'<div class="action-card" style="border-left-color:{gr.grade_color}">'
                        f'<b>目标{i+1}</b> &nbsp; {gr.fruit}·{gr.ripeness} &nbsp;'
                        f'<span class="badge" style="background:{gr.grade_color}">{gr.grade_label}</span>'
                        f'&nbsp; conf <b>{gr.confidence:.3f}</b><br>'
                        f'<span style="color:#A07040;margin-top:3px;display:block">{gr.action}</span>'
                        f'</div>', unsafe_allow_html=True)

                buf = io.BytesIO()
                Image.fromarray(ann_rgb).save(buf, format="PNG")

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button("⬇️ 下载结果图", buf.getvalue(),
                                       file_name="result.png", mime="image/png")
                with dl2:
                    with st.spinner("生成PDF中..."):
                        try:
                            pdf_bytes = generate_pdf_report(
                                grades=grades,
                                summary=summary,
                                annotated_img=ann_bgr,
                                latency_ms=latency,
                                model_name=model_name,
                                source_name=uploaded.name,
                                history=st.session_state.history,
                            )
                            st.download_button(
                                "📄 导出PDF检测报告", pdf_bytes,
                                file_name=f"fruit_report_{datetime.now().strftime('%H%M%S')}.pdf",
                                mime="application/pdf",
                            )
                        except Exception as e:
                            st.warning(f"PDF生成失败: {e}\n请先运行: pip install reportlab")
        else:
            st.markdown('<div class="empty-state"><div style="font-size:2.8rem">🍊</div>'
                        '<div style="font-weight:700;margin-top:.5rem">请在左侧上传水果图片</div>'
                        '<div style="font-size:.8rem;margin-top:.3rem">支持 JPG · PNG · BMP · WEBP</div></div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Tab2: 视频检测
# ══════════════════════════════════════════════
st.markdown(
    '<div style="background:#EFF6FF;border-radius:8px;padding:0.5rem 0.8rem;'
    'font-size:0.75rem;color:#4A7AAF;border:1px solid #BFDBFE;">'
    '💡 建议使用白背景、正常白光、无水印的水果图片以获得最佳效果。'
    '含水印、强暖光或复杂背景的图片可能影响检测准确率。'
    '</div>',
    unsafe_allow_html=True
)
# ══════════════════════════════════════════════
# Tab3: 视频巡检
# ══════════════════════════════════════════════
with T["video"]:
    st.markdown(
        '<div style="font-size:1.1rem;font-weight:900;color:#C05C00;margin-bottom:0.2rem">🎬 视频巡检检测</div>'
        '<div style="font-size:0.82rem;color:#A07040;margin-bottom:0.8rem">'
        '上传库房巡检视频，系统自动抽帧分析水果品质。</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="background:#EFF6FF;border-radius:8px;padding:0.5rem 0.8rem;'
        'font-size:0.75rem;color:#4A7AAF;border:1px solid #BFDBFE;margin-bottom:0.8rem">'
        '📌 适用场景：在仓库中边走边录像，上传后系统自动盘点。'
        '数量为估算值，建议人工复核。光照均匀、水果单层摆放时效果最佳。'
        '</div>',
        unsafe_allow_html=True
    )

    video_batch_id = st.text_input(
        "入库批次号（选填）",
        placeholder="例如：2026-04-13-002",
        key="video_batch_id",
        help="用于报表归档，不填则自动生成"
    )
    if not video_batch_id:
        video_batch_id = datetime.now().strftime("巡检-%Y%m%d-%H%M%S")

    vf = st.file_uploader(
        "上传巡检视频",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="visible"
    )

    if vf:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vf.read())
        tfile.flush()
        cap   = cv2.VideoCapture(tfile.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.release()

        sample_interval_sec = 0.5
        sample_interval_frames = max(1, int(fps_v * sample_interval_sec))
        est_frames = max(1, int(total / sample_interval_frames))

        st.info(f"📹 视频共 {total} 帧 · 原始 FPS {fps_v:.1f} · 每 {sample_interval_sec}s 抽1帧 · 预计抽取 {est_frames} 帧")

        if st.button("▶ 开始巡检分析"):

            cap = cv2.VideoCapture(tfile.name)
            ph  = st.empty()
            pb  = st.progress(0, "抽帧中...")

            sampled_frames = []
            fi = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if fi % sample_interval_frames == 0:
                    sampled_frames.append(frame.copy())
                pb.progress(min(fi / max(total, 1), 0.4))
                fi += 1
            cap.release()

            pb.progress(0.4, "批量检测中...")

            # ── 复用批量检测逻辑，收集所有检测框 ──
            all_raw_dets = []
            PATCH_SIZE = 640
            PATCH_STRIDE = 480  # 重叠160px，避免边缘目标漏检

            for idx, frame in enumerate(sampled_frames):
                oh, ow = frame.shape[:2]
                frame_dets = []

                # ── 切patch：按stride滑窗，覆盖全图 ──
                ys = list(range(0, oh - PATCH_SIZE + 1, PATCH_STRIDE))
                xs = list(range(0, ow - PATCH_SIZE + 1, PATCH_STRIDE))
                # 确保最后一列/行被覆盖
                if not ys or ys[-1] + PATCH_SIZE < oh:
                    ys.append(max(0, oh - PATCH_SIZE))
                if not xs or xs[-1] + PATCH_SIZE < ow:
                    xs.append(max(0, ow - PATCH_SIZE))

                for py in ys:
                    for px in xs:
                        patch = frame[py:py + PATCH_SIZE, px:px + PATCH_SIZE]
                        res = run_inference(model, patch, conf_thres, iou_thres)
                        dets = results_to_detections(res)
                        for d in dets:
                            # 坐标映射回原图
                            x1, y1, x2, y2 = d["xyxy"]
                            frame_dets.append({
                                "class_id": d["class_id"],
                                "confidence": d["confidence"],
                                "xyxy": [x1 + px, y1 + py, x2 + px, y2 + py],
                                "frame_idx": idx,
                            })


                # ── patch内NMS：同帧内去除重叠框 ──
                def patch_nms(dets, iou_thresh=0.45):
                    if not dets:
                        return []
                    sorted_d = sorted(dets, key=lambda x: -x["confidence"])
                    kept_p = []
                    for d in sorted_d:
                        overlap = False
                        for k in kept_p:
                            ax1, ay1, ax2, ay2 = d["xyxy"]
                            bx1, by1, bx2, by2 = k["xyxy"]
                            ix1 = max(ax1, bx1);
                            iy1 = max(ay1, by1)
                            ix2 = min(ax2, bx2);
                            iy2 = min(ay2, by2)
                            iw = max(0, ix2 - ix1);
                            ih = max(0, iy2 - iy1)
                            inter = iw * ih
                            aA = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
                            aB = max(1e-6, (bx2 - bx1) * (by2 - by1))
                            if inter / (aA + aB - inter) >= iou_thresh:
                                overlap = True;
                                break
                        if not overlap:
                            kept_p.append(d)
                    return kept_p


                frame_dets = patch_nms(frame_dets)
                frame_dets = cross_class_nms(frame_dets, iou_threshold=0.5)  # ← 跨类去重
                all_raw_dets.extend(frame_dets)

                # 预览：在原图上画框
                ann = annotate_image(frame, frame_dets)
                ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                pb.progress(0.4 + 0.5 * (idx + 1) / max(len(sampled_frames), 1))

            pb.progress(0.9, "去重计数中...")



            # ── IOU去重：纯空间位置去重，不按class_id分组 ──
            # 同一位置框无论识别为什么类别，只保留置信度最高的那个
            # ── 跨帧去重：按归一化中心点距离合并同一水果 ──
            # 苹果依次出现时坐标不同，IOU无法跨帧合并；改用中心点距离判断
            def center_dist_ratio(boxA, boxB, frame_w, frame_h):
                """归一化中心点距离，< 阈值视为同一目标"""
                cx_a = (boxA[0] + boxA[2]) / 2 / frame_w
                cy_a = (boxA[1] + boxA[3]) / 2 / frame_h
                cx_b = (boxB[0] + boxB[2]) / 2 / frame_w
                cy_b = (boxB[1] + boxB[3]) / 2 / frame_h
                return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


            def box_iou(boxA, boxB):
                ax1, ay1, ax2, ay2 = boxA
                bx1, by1, bx2, by2 = boxB
                ix1 = max(ax1, bx1);
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2);
                iy2 = min(ay2, by2)
                iw = max(0, ix2 - ix1);
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                areaA = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
                areaB = max(1e-6, (bx2 - bx1) * (by2 - by1))
                return inter / (areaA + areaB - inter)


            # 取第一帧尺寸用于归一化
            if sampled_frames:
                fh, fw = sampled_frames[0].shape[:2]
            else:
                fh, fw = 640, 640

            DIST_THRESH = 0.45  # 原0.30 → 继续放宽，覆盖跨帧位置漂移
            IOU_THRESH = 0.40

            sorted_dets = sorted(all_raw_dets, key=lambda x: -x["confidence"])

            kept = []
            for det in sorted_dets:
                if det["confidence"] < 0.60:  # 原0.45 → 提高，过滤低置信误检橙子/香蕉
                    continue
                merged = False
                for k in kept:
                    same_frame = (k.get("frame_idx", -1) == det.get("frame_idx", -2))
                    if same_frame:
                        # 同帧：用IOU判断
                        if box_iou(k["xyxy"], det["xyxy"]) >= IOU_THRESH:
                            merged = True
                            break
                    else:
                        # 跨帧：用归一化中心点距离判断
                        if center_dist_ratio(k["xyxy"], det["xyxy"], fw, fh) < DIST_THRESH:
                            merged = True
                            break
                if not merged:
                    kept.append({
                        "class_id": det["class_id"],
                        "confidence": det["confidence"],
                        "xyxy": det["xyxy"],
                        "frame_idx": det.get("frame_idx", 0),
                    })

            pb.progress(1.0, "完成")

            if kept:
                smoothed_grades = [classify_grade(d["class_id"], d["confidence"]) for d in kept]
                sm = summarize_grades(smoothed_grades)
                add_to_board(smoothed_grades, vf.name)
                add_to_history(f"巡检视频({vf.name})", [], smoothed_grades, 0, model_name)

                total_count = sm["total"]
                a_pct = sm["A"] / total_count * 100 if total_count else 0
                b_pct = sm["B"] / total_count * 100 if total_count else 0
                c_pct = sm["C"] / total_count * 100 if total_count else 0

                st.markdown("---")
                st.markdown(
                    '<div style="font-size:1.1rem;font-weight:900;'
                    'color:#C05C00;margin-bottom:0.2rem">📋 巡检盘点结论</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="background:#FFFBF0;border-radius:16px;'
                    f'padding:1.2rem 1.5rem;border:1.5px solid #F5D99A;margin-bottom:1rem;">'
                    f'<div style="font-size:0.9rem;color:#3D2B1F;line-height:2.2;">'
                    f'📹 批次号：<b>{video_batch_id}</b><br>'
                    f'🔢 本次巡检去重后检测到水果目标 <b>{total_count}</b> 个（抽取 {len(sampled_frames)} 帧，IOU≥0.5合并重复）<br>'
                    f'🟢 <b>A级优品</b>：{sm["A"]} 个（{a_pct:.1f}%）'
                    f'&nbsp;→&nbsp; 建议直接上架销售<br>'
                    f'🟡 <b>B级普通果</b>：{sm["B"]} 个（{b_pct:.1f}%）'
                    f'&nbsp;→&nbsp; 建议低价促销或继续催熟<br>'
                    f'🔴 <b>C级次品</b>：{sm["C"]} 个（{c_pct:.1f}%）'
                    f'&nbsp;→&nbsp; <b>建议人工抽检复核后处理</b>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
                st.warning(
                    "⚠️ 以上为系统估算结果，实际数量以人工复核为准。"
                    "建议在光照均匀、水果单层摆放的环境下录制以提高准确率。"
                )

                vc1, vc2 = st.columns(2)
                with vc1:
                    st.markdown("**品质分级占比**")
                    st.plotly_chart(chart_pie(sm), use_container_width=True, key="video_pie")
                with vc2:
                    st.markdown("**各品类分级数量**")
                    st.plotly_chart(chart_bar(sm), use_container_width=True, key="video_bar")

                import pandas as pd
                video_rows = []
                for g in smoothed_grades:
                    video_rows.append({
                        "批次号":   video_batch_id,
                        "视频文件": vf.name,
                        "水果种类": g.fruit,
                        "成熟度":   g.ripeness,
                        "品质等级": g.grade_label,
                        "置信度":   round(g.confidence, 3),
                        "处理建议": g.action.replace("✅","").replace("⚠️","").replace("❌","").strip(),
                        "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                df_v  = pd.DataFrame(video_rows)
                csv_v = df_v.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "⬇️ 导出巡检CSV报表", csv_v,
                    file_name=f"{video_batch_id}_巡检报表.csv",
                    mime="text/csv", key="video_csv"
                )
            else:
                st.warning("未检测到任何水果，请检查视频内容或调低置信度阈值后重试。")

# ══════════════════════════════════════════════
# Tab3: 摄像头
# ══════════════════════════════════════════════
with T["cam"]:
    st.markdown(
        '<div style="font-size:1.1rem;font-weight:900;color:#C05C00;margin-bottom:0.2rem">📹 实时摄像头检测</div>'
        '<div style="font-size:0.82rem;color:#A07040;margin-bottom:0.8rem">连接摄像头实时检测，结果自动同步至本批汇总。</div>',
        unsafe_allow_html=True
    )
    cc1, cc2 = st.columns([3, 1])
    with cc1: run_cam = st.toggle("🔴 开启实时检测")
    with cc2: cam_idx = st.number_input("摄像头编号", 0, 5, 0)

    if run_cam:
        ph_c = st.empty()
        ph_s = st.empty()
        cap  = cv2.VideoCapture(int(cam_idx))
        fps_b          = []
        fi             = 0
        all_cam_grades = []

        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("无法读取摄像头")
                break

            t0      = time.perf_counter()
            res     = run_inference(model, frame, conf_thres, iou_thres)
            elapsed = time.perf_counter() - t0

            # ✅ 修复除零风险
            fps_b.append(1 / elapsed if elapsed > 0 else 0)
            if len(fps_b) > 20:
                fps_b.pop(0)

            dets = results_to_detections(res)
            dets = cross_class_nms(dets, iou_threshold=0.5)  # ← 跨类去重
            ann = annotate_image(frame, dets)
            ph_c.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)

            # ✅ 修复重复计算：grades_frame 只算一次，sm 直接复用
            grades_frame = [classify_grade(d["class_id"], d["confidence"]) for d in dets]
            all_cam_grades.extend(grades_frame)
            sm = summarize_grades(grades_frame)

            ph_s.markdown(
                f"**FPS** `{np.mean(fps_b):.1f}` | "
                f"目标 `{sm['total']}` | "
                f"🟢`{sm['A']}` 🟡`{sm['B']}` 🔴`{sm['C']}`"
            )

            # 每30帧写一次看板
            if fi % 30 == 0 and grades_frame:
                add_to_board(grades_frame, f"摄像头#{int(cam_idx)}")
            fi += 1

        cap.release()

        # 退出后写入历史记录
        if all_cam_grades:
            add_to_history(
                f"摄像头#{int(cam_idx)}实时检测",
                [], all_cam_grades,
                float(np.mean(fps_b)) if fps_b else 0,
                model_name
            )


# Tab1: 入库检测（批量）
# ══════════════════════════════════════════════
with T["batch"]:
    # ══════════════════════════════════════════════
    # 阶段 2.1: 顶部双卡挤一行(最近状态 + 活跃进度 + 新批次按钮)
    # ══════════════════════════════════════════════
    _cur_batch = st.session_state.get("current_batch")

    # 无活跃批次:渲染欢迎引导卡片
    if _cur_batch is None:
        _empty_col1, _empty_col2 = st.columns([3, 1])
        with _empty_col1:
            st.markdown(
                '<div style="background:linear-gradient(135deg,#FFFBF0 0%,#FFF8ED 100%);'
                'border:1.5px solid #F5D99A;border-radius:14px;padding:1.1rem 1.4rem;'
                'box-shadow:0 2px 12px rgba(200,130,0,.06)">'
                '<div style="display:flex;align-items:center;gap:14px">'
                '<div style="font-size:2.2rem">📦</div>'
                '<div>'
                '<div style="font-size:1.05rem;font-weight:900;color:#3D2B1F">'
                '欢迎使用入库检测系统</div>'
                '<div style="font-size:0.82rem;color:#A07040;margin-top:4px">'
                '尚无活跃批次。请填写下方批次基础信息并上传水果图片,系统将自动完成盘点与品质分级。'
                '</div></div></div></div>',
                unsafe_allow_html=True
            )
        with _empty_col2:
            st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
            st.button("➕ 新批次",
                      key="btn_new_batch_empty",
                      use_container_width=True,
                      help="清空表单,开始一个全新的批次",
                      disabled=True)
        st.markdown("")

    # 有活跃批次:渲染顶部三栏(原阶段 2.1 逻辑)
    if _cur_batch is not None:
        _cb_color = _cur_batch.status.color
        _cb_name  = _cur_batch.status.display_name
        _cb_terminal = _cur_batch.status.is_terminal

        # 状态时间戳(右半边显示用)
        _stamp = ""
        if _cur_batch.status == BatchStatus.CONFIRMED and _cur_batch.confirmed_at:
            _stamp = f"入库于 {_cur_batch.confirmed_at.strftime('%H:%M:%S')}"
        elif _cur_batch.status == BatchStatus.REJECTED and _cur_batch.rejected_at:
            _stamp = f"拒收于 {_cur_batch.rejected_at.strftime('%H:%M:%S')}"
        elif _cur_batch.status == BatchStatus.READY_TO_CONFIRM:
            _stamp = "等待您确认入库"
        elif _cur_batch.status == BatchStatus.PENDING_REVIEW:
            _stamp = "等待经理复核"
        else:
            _stamp = "进行中"

        _ac_locked_msg = "批次已锁定,请创建新批次" if _cb_terminal else "可继续补充检测或创建新批次"

        # 三列布局:左 最近状态 / 中 活跃进度 / 右 新批次按钮
        _hd_left, _hd_mid, _hd_btn = st.columns([3, 4, 1.1], vertical_alignment="center")

        with _hd_left:
            st.markdown(
                f'<div style="background:#FFFFFF;border-radius:14px;'
                f'padding:0.7rem 1rem;border:1.5px solid #F0EAD6;'
                f'border-left:5px solid {_cb_color};font-size:0.85rem;'
                f'min-height:78px;display:flex;flex-direction:column;justify-content:center">'
                f'<div style="font-size:0.7rem;color:#A07040;font-weight:600;margin-bottom:3px">'
                f'📌 最近批次</div>'
                f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
                f'<span style="font-weight:800;color:#3D2B1F;font-size:0.92rem">{_cur_batch.batch_id}</span>'
                f'<span style="background:{_cb_color};color:white;padding:2px 10px;'
                f'border-radius:50px;font-weight:700;font-size:0.72rem">{_cb_name}</span>'
                f'</div>'
                f'<div style="color:#A07040;font-size:0.76rem;margin-top:3px">{_stamp}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with _hd_mid:
            st.markdown(
                f'<div style="background:#FFFBF0;border:1.5px solid #F5D99A;'
                f'border-left:6px solid {_cb_color};border-radius:14px;'
                f'padding:0.7rem 1rem;font-size:0.85rem;color:#3D2B1F;'
                f'min-height:78px;display:flex;flex-direction:column;justify-content:center">'
                f'<div style="font-size:0.7rem;color:#A07040;font-weight:600;margin-bottom:3px">'
                f'⚡ 活跃进度</div>'
                f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;'
                f'font-size:0.82rem">'
                f'<span>🔁 已检测 <b style="color:#7C3AED">{_cur_batch.detection_rounds}</b> 轮</span>'
                f'<span>🎯 累计 <b style="color:#FF8C00">{_cur_batch.detected_total}</b> 个目标</span>'
                f'<span>📁 处理 <b style="color:#0EA5E9">{len(_cur_batch.processed_files)}</b> 个文件</span>'
                f'</div>'
                f'<div style="color:#A07040;font-size:0.76rem;margin-top:3px">{_ac_locked_msg}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with _hd_btn:
            if st.button("➕ 新批次", key="btn_new_batch_top",
                         use_container_width=True,
                         help="清空当前活跃批次,开始一个全新的入库批次"):
                for _k in ("current_batch",):
                    if _k in st.session_state:
                        del st.session_state[_k]
                st.session_state.uploader_version = st.session_state.get("uploader_version", 0) + 1
                st.rerun()
        st.markdown("")  # 卡片和下方标题之间留一点空隙

    st.markdown(
        '<div style="font-size:1.1rem;font-weight:900;color:#C05C00;margin-bottom:0.2rem">📦 入库批量检测</div>'
        '<div style="font-size:0.82rem;color:#A07040;margin-bottom:0.8rem">'
        '填写批次信息后上传水果图片,系统自动完成盘点、对账与品质分级。</div>',
        unsafe_allow_html=True
    )

    # ══════════════════════════════════════════════
    # Section 1: 批次信息表单(对账核心字段)
    # ══════════════════════════════════════════════

    # ══════════════════════════════════════════════
    # 阶段 2.2: 主体两列容器(左 表单+检测,右 实时预览)
    # ══════════════════════════════════════════════
    _left_col, _right_col = st.columns([3, 2], gap="medium")

    with _left_col:
        st.markdown("#### 📋 批次基础信息")

        # ── Step 5.4b: 表单锁定开关(有活跃批次则锁定基础信息字段)──
        _active_lock = (st.session_state.get("current_batch") is not None)
        _lock_src    = st.session_state.get("current_batch")
        if _active_lock:
            st.caption("🔒 当前已有活跃批次,基础信息已锁定。如需修改,请点击顶部 ➕ 新批次")

        bf_col1, bf_col2, bf_col3 = st.columns([1.2, 1, 1])
        with bf_col1:
            batch_id = st.text_input(
                "批次号 *",
                value=(_lock_src.batch_id if _active_lock else ""),
                placeholder="留空则自动生成",
                help="留空将以 入库-YYYYMMDD-HHMMSS 格式自动生成",
                key="batch_id_input",
                disabled=_active_lock,
            )
            if not batch_id:
                batch_id = datetime.now().strftime("入库-%Y%m%d-%H%M%S")
        with bf_col2:
            supplier_name = st.text_input(
                "供应商名称",
                value=(_lock_src.supplier_name if _active_lock else ""),
                placeholder="例如:山东烟台果园",
                help="可选,供应商档案后续可独立管理",
                key="supplier_name_input",
                disabled=_active_lock,
            )
        with bf_col3:
            _fc_opts = ["苹果", "香蕉", "葡萄", "橙子", "混合"]
            fruit_category = st.selectbox(
                "主要品类 *",
                _fc_opts,
                index=(_fc_opts.index(_lock_src.fruit_category)
                       if _active_lock and _lock_src.fruit_category in _fc_opts else 0),
                help="本批次的主要水果品类",
                key="fruit_category_input",
                disabled=_active_lock,
            )

        st.markdown("#### 🏭 仓储与对账")
        sf_col1, sf_col2, sf_col3 = st.columns([1, 1, 1.2])
        with sf_col1:
            _wh_opts = ["一号仓", "二号仓", "三号仓", "冷链仓"]
            warehouse = st.selectbox(
                "入库仓库 *",
                _wh_opts,
                index=(_wh_opts.index(_lock_src.warehouse)
                       if _active_lock and _lock_src.warehouse in _wh_opts else 0),
                key="warehouse_input",
                disabled=_active_lock,
            )
        with sf_col2:
            _sz_opts = ["A区", "B区", "C区", "暂存区"]
            storage_zone = st.selectbox(
                "库区 *",
                _sz_opts,
                index=(_sz_opts.index(_lock_src.storage_zone)
                       if _active_lock and _lock_src.storage_zone in _sz_opts else 0),
                key="storage_zone_input",
                disabled=_active_lock,
            )
        with sf_col3:
            declared_count = st.number_input(
                "📦 申报数量(供应商声称)*",
                min_value=0, max_value=100000,
                value=(int(_lock_src.declared_count) if _active_lock else 100),
                step=1,
                help="🎯 对账核心字段。系统将与实际检测数量自动比对。",
                key="declared_count_input",
                disabled=_active_lock,
            )

        st.markdown("---")

        st.markdown("#### 📷 上传水果图片")
        bfs = st.file_uploader(
            "支持多张图片,系统将合并盘点",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"batch_uploader_v{st.session_state.uploader_version}"
        )

        if bfs:
            _info_col, _clear_col = st.columns([5, 1])
            with _info_col:
                st.markdown(
                    f'<div style="background:#FFF8ED;border:1.5px solid #F5D99A;'
                    f'border-left:5px solid #FF8C00;border-radius:10px;'
                    f'padding:0.5rem 0.95rem;font-size:0.85rem;color:#3D2B1F;'
                    f'height:38px;display:flex;align-items:center;'
                    f'box-sizing:border-box;line-height:1.2">'
                    f'📥 <b style="margin-left:0.4rem">已选择 {len(bfs)} 张图片</b>'
                    f'  &nbsp;|&nbsp; 申报数量:<b>{declared_count}</b>'
                    f'  &nbsp;|&nbsp; 批次号:<b>{batch_id}</b>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with _clear_col:
                if st.button("🗑️ 清空", key="btn_clear_uploads", use_container_width=True,
                             help="一键清空所有已选图片"):
                    st.session_state.uploader_version += 1
                    st.rerun()

        # ══════════════════════════════════════════════
        # Step 5.4c: 三按钮分支(D 段)
        # ══════════════════════════════════════════════
        _action = None       # None / "first" / "supp" / "reset"
        _files_to_run = []

        if bfs:
            if _active_lock:
                if _lock_src.status.is_terminal:
                    # 终态:批次已锁,不允许任何检测
                    st.warning(
                        "⚠️ 批次已锁定(已入库/已拒收),不能再做检测。"
                        "请点击顶部 ➕ 新批次 创建新批次。"
                    )
                else:
                    # 非终态:补充 + 重检
                    _new_files_list = [f for f in bfs if f.name not in _lock_src.processed_files]
                    _n_new = len(_new_files_list)
                    _n_all = len(bfs)
                    _supp_label  = f"➕ 补充检测({_n_new} 张新图)"
                    _reset_label = f"🔄 重新检测({_n_all} 张全部)"

                    # ── Step 5.5b: 补充检测使用说明 ──
                    st.caption(
                        "ℹ️ 提示:补充检测时可重新选择全部图片(包括已检测过的),"
                        "系统会自动跳过已处理的图。"
                    )

                    # 阶段 2.3: 改为纵向堆叠(每个按钮独占一行,在窄列里更醒目)
                    if st.button(_supp_label, type="primary",
                                 disabled=(_n_new == 0),
                                 use_container_width=True,
                                 key="btn_supp_detect"):
                        _action = "supp"
                        _files_to_run = _new_files_list
                    if st.button(_reset_label, type="secondary",
                                 use_container_width=True,
                                 key="btn_reset_detect"):
                        _action = "reset"
                        _files_to_run = list(bfs)
                    if _n_new == 0:
                        st.caption("ℹ️ 当前选中的图片均已检测过,如需补充请先上传新图,或点重新检测。")
                    st.caption("⚠️ 重新检测会清空当前批次的所有累积结果,从零重跑。")
            else:
                # 无活跃批次:首检
                if st.button("🚀 开始入库检测", type="primary",
                             use_container_width=True,
                             key="btn_first_detect"):
                    _action = "first"
                    _files_to_run = list(bfs)

        # ══════════════════════════════════════════════
        # Step 5.4c: 检测执行(C' 段)
        # ══════════════════════════════════════════════
        if _action is not None:
            # ── Bug 修复:护栏 ──
            # 防止 rerun 时残留状态触发对终态批次的误重检
            _existing_batch = st.session_state.get("current_batch")
            if (_existing_batch is not None
                    and _existing_batch.status.is_terminal
                    and _action == "first"):
                # 终态批次不应该再新建对象,直接吞掉本次动作
                _action = None
                _files_to_run = []

            from services.batch_service import (
                apply_supplementary_detection as _svc_apply_supp,
                reset_detection as _svc_reset,
            )

            if _action == "first":
                # ── Step 7.4: 首检即建供应商档案(幂等)──
                # 命中返回已有档案,未建档则自动建空白档案。supplier_id 写进 batch,
                # 之后 batch_service 的 KPI 聚合 / 供应商档案 Tab 都能正确关联。
                from services import supplier_service as _svc_sup
                _sup_for_batch = _svc_sup.get_or_create_supplier(
                    st.session_state.suppliers, supplier_name
                )
                _new_supplier_id = _sup_for_batch.supplier_id if _sup_for_batch else None
                batch = InboundBatch(
                    batch_id       = batch_id,
                    operator_id    = current_user.user_id,
                    operator_name  = current_user.full_name,
                    supplier_id    = _new_supplier_id,
                    supplier_name  = supplier_name,
                    fruit_category = fruit_category,
                    declared_count = int(declared_count),
                    warehouse      = warehouse,
                    storage_zone   = storage_zone,
                )
            elif _action == "reset":
                batch = _lock_src
                _svc_reset(batch)
            else:  # supp
                batch = _lock_src

            # ── Step 5.5a: 即时刷新活跃批次徽章(批次准备好之后,跑推理之前)──
            # 解决"首检完成后顶部活跃批次卡片要切 tab 才出现"的问题
            # 由于 5.4a 卡片在表单顶部、_action 块在表单下方,Streamlit 顺序渲染下
            # 上面的卡片读到 current_batch 还是旧值。这里在 _action 块内重新渲染
            # 一次"即时状态条",让用户当次就能看到状态变化。
            _imm_color = batch.status.color
            _imm_name  = batch.status.display_name
            _action_label = {"first": "首次检测", "reset": "重新检测", "supp": "补充检测"}[_action]
            st.markdown(
                f'<div style="background:#FFFBF0;border:1.5px solid #F5D99A;'
                f'border-left:6px solid {_imm_color};border-radius:14px;'
                f'padding:0.7rem 1.1rem;font-size:0.85rem;color:#3D2B1F;'
                f'margin:0.6rem 0 0.4rem;'
                f'display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
                f'<span style="font-size:0.7rem;color:#A07040;font-weight:600">本次操作</span>'
                f'<span style="font-weight:800;color:#3D2B1F">{_action_label}</span>'
                f'<span style="font-weight:800;color:#3D2B1F">{batch.batch_id}</span>'
                f'<span style="background:{_imm_color};color:white;padding:2px 10px;'
                f'border-radius:50px;font-weight:700;font-size:0.75rem">{_imm_name}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            pb    = st.progress(0, "检测中...")
            all_g = []
            rows  = []
            total_conf = 0.0

            # 跑 _files_to_run(补充检测时只跑新图)
            for i, f in enumerate(_files_to_run):
                img  = np.array(Image.open(f).convert("RGB"))
                bgr  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                res  = run_inference(model, bgr, conf_thres, iou_thres)
                dets = results_to_detections(res)
                dets = cross_class_nms(dets, iou_threshold=0.5)
                grs  = [classify_grade(d["class_id"], d["confidence"]) for d in dets]
                all_g.extend(grs)
                for g in grs:
                    rows.append({
                        "批次号":   batch.batch_id,
                        "文件名":   f.name,
                        "水果种类": g.fruit,
                        "成熟度":   g.ripeness,
                        "品质等级": g.grade_label,
                        "置信度":   g.confidence,
                        "处理建议": g.action.replace("✅","").replace("⚠️","").replace("❌","").strip(),
                        "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    total_conf += g.confidence
                pb.progress((i + 1) / max(len(_files_to_run), 1))

            if not all_g:
                if _action == "supp":
                    st.warning(
                        "ℹ️ **补充检测未识别到新目标** — 本轮上传的新图片中没有检出水果。\n\n"
                        "批次累积数据保持不变。建议检查图片质量后重试,或点击重新检测。"
                    )
                else:
                    st.error(
                        "⚠️ **本批次未检测到任何水果目标** — 系统无法完成盘点。\n\n"
                        "**可能原因**:整批图像质量问题、严重腐烂超出训练分布、"
                        "图像内容非水果或背景过于复杂。\n\n"
                        "**建议操作**:**整批人工复核**,不可凭本系统结果直接入库。"
                    )
            else:
                sm = summarize_grades(all_g)
                avg_conf = total_conf / len(all_g) if all_g else 0.0

                if _action == "supp":
                    # 累加式
                    new_status, status_reason = _svc_apply_supp(
                        batch,
                        new_count = sm["total"],
                        new_a = sm["A"], new_b = sm["B"], new_c = sm["C"],
                        new_avg_confidence = avg_conf,
                        newly_processed_files = {f.name for f in _files_to_run},
                    )
                else:
                    # first / reset:覆盖式
                    new_status, status_reason = apply_detection_result(
                        batch,
                        detected_total = sm["total"],
                        grade_a = sm["A"], grade_b = sm["B"], grade_c = sm["C"],
                        avg_confidence = avg_conf,
                    )
                    # 覆盖式服务函数不自增审计字段,UI 层补
                    batch.detection_rounds += 1
                    batch.processed_files = {f.name for f in _files_to_run}

                st.session_state.current_batch = batch
                # ── Step 4.1 + 9.2.2.c.2: 入池 + SQLite 持久化(双写)──
                # DRAFT 状态(零目标兜底)不入池,因为业务上还未"提交"
                if batch.status != BatchStatus.DRAFT:
                    _pool = st.session_state.all_batches
                    # 同 batch_id 去重:重新检测则替换旧记录
                    _pool[:] = [b for b in _pool if b.batch_id != batch.batch_id]
                    _pool.append(batch)
                    # Step 9.2.2.c.2: 同步写 DB(upsert)
                    _batch_repo.save(batch)
                add_to_history(f"批量({len(bfs)}张)", [], all_g, 0, model_name)
                add_to_board(all_g, f"批量检测{len(bfs)}张")

                # ── Bug fix: 检测完成后强制 rerun,让顶部三栏读到新状态 ──
                # Bug fix: rerun to sync top-bar state
                st.rerun()

                st.markdown("---")

                st.markdown("---")
                st.markdown("---")
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown("**品质分级占比**")
                    st.plotly_chart(chart_pie(sm), use_container_width=True, key="batch_pie")
                with rc2:
                    st.markdown("**各品类分级数量**")
                    st.plotly_chart(chart_bar(sm), use_container_width=True, key="batch_bar")

                st.markdown("---")
                st.markdown("**逐项检测明细**")
                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                st.markdown("---")
                st.markdown("**导出报表**")
                csv = df.to_csv(index=False, encoding="utf-8-sig")
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "⬇️ 导出 CSV 盘点报表",
                        csv,
                        file_name=f"{batch_id}_盘点报表.csv",
                        mime="text/csv",
                        key="batch_csv"
                    )
                with dl2:
                    with st.spinner("生成PDF中..."):
                        try:
                            pdf_bytes = generate_pdf_report(
                                grades=all_g,
                                summary=sm,
                                annotated_img=None,
                                latency_ms=0,
                                model_name=model_name,
                                source_name=f"{batch_id}({len(bfs)}张)",
                                history=st.session_state.history,
                            )
                            st.download_button(
                                "📄 导出 PDF 检测报告",
                                pdf_bytes,
                                file_name=f"{batch_id}_检测报告.pdf",
                                mime="application/pdf",
                                key="batch_pdf"
                            )
                        except Exception as e:
                            st.warning(f"PDF生成失败: {e}")

    # ══════════════════════════════════════════════


        # ══════════════════════════════════════════════
        # 阶段 2.4-修复: 后续操作块独立渲染(从 if _action 内移出)
        # 修复 bug: 点击"确认入库"按钮无效(rerun 后按钮不再渲染,信号丢失)
        # ══════════════════════════════════════════════
        if st.session_state.get("current_batch") is not None:
            st.markdown("---")
            st.markdown("#### ⚙️ 后续操作")

            from services.batch_service import (
                confirm_batch as svc_confirm_batch,
                review_approve as svc_review_approve,
                review_reject as svc_review_reject,
                StateTransitionError,
            )
            from services.permission_service import (
                can_confirm_batch as can_confirm,
                can_review_batch as can_review,
                can_reject_batch as can_reject,
            )

            cb = st.session_state.current_batch

            # 终态批次:不再可流转
            if cb.status in {BatchStatus.CONFIRMED, BatchStatus.REJECTED}:
                if cb.status == BatchStatus.CONFIRMED:
                    st.success(
                        f"✅ **批次已入库** — 操作人:{current_user.full_name},"
                        f"时间:{cb.confirmed_at.strftime('%Y-%m-%d %H:%M:%S') if cb.confirmed_at else '—'}"
                    )
                    if cb.reviewed_note:
                        st.caption(f"📝 复核备注:{cb.reviewed_note}")
                else:
                    st.error(
                        f"❌ **批次已拒收** — 操作人:{current_user.full_name},"
                        f"时间:{cb.rejected_at.strftime('%Y-%m-%d %H:%M:%S') if cb.rejected_at else '—'}"
                    )
                    if cb.rejection_reason:
                        st.caption(f"📝 拒收原因:{cb.rejection_reason}")

            elif cb.status == BatchStatus.DRAFT:
                st.warning("⚠️ 当前批次未检测到任何水果目标,无法进行后续操作。")

            elif cb.status == BatchStatus.READY_TO_CONFIRM:
                if can_confirm(current_user, cb.status):
                    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
                    with btn_col1:
                        if st.button("✅ 确认入库", type="primary",
                                     use_container_width=True, key="btn_confirm"):
                            try:
                                svc_confirm_batch(cb, current_user)
                                _batch_repo.save(cb)   # Step 9.2.2.c.2
                                st.success(f"✅ 批次 **{cb.batch_id}** 已成功入库!")
                                st.balloons()
                                st.rerun()
                            except StateTransitionError as e:
                                st.error(f"❌ 操作失败:{e}")

                    with btn_col2:
                        if can_reject(current_user, cb.status):
                            with st.popover("❌ 拒收", use_container_width=True):
                                st.markdown("**拒收原因(必填)**")
                                reject_reason = st.text_area(
                                    "请详细说明拒收原因",
                                    key="reject_reason_ready",
                                    placeholder="例如:实物核验数量与系统检测不一致..."
                                )
                                if st.button("确认拒收", type="primary",
                                             key="btn_reject_ready_confirm"):
                                    if not reject_reason.strip():
                                        st.error("拒收必须填写原因")
                                    else:
                                        try:
                                            svc_review_reject(cb, current_user,
                                                              reason=reject_reason)
                                            _batch_repo.save(cb)   # Step 9.2.2.c.2
                                            st.success(f"❌ 批次 **{cb.batch_id}** 已拒收")
                                            st.rerun()
                                        except StateTransitionError as e:
                                            st.error(f"操作失败:{e}")

                    with btn_col3:
                        st.caption(
                            "🟢 系统判定该批次符合快速入库条件,可直接确认。"
                            "如对实物有疑问,可点击拒收。"
                        )
                else:
                    st.info("您当前角色无权对此批次进行操作。")

            elif cb.status == BatchStatus.PENDING_REVIEW:
                if can_review(current_user, cb.status):
                    rv_col1, rv_col2 = st.columns([1, 1])
                    with rv_col1:
                        with st.popover("✅ 复核通过", use_container_width=True):
                            st.markdown("**复核备注(可选)**")
                            review_note = st.text_area(
                                "实物核验情况说明",
                                key="review_note_input",
                                placeholder="例如:已抽检 10 个,实物状态良好,准予入库"
                            )
                            if st.button("确认通过", type="primary",
                                         key="btn_approve_confirm"):
                                try:
                                    svc_review_approve(cb, current_user,
                                                       note=review_note)
                                    _batch_repo.save(cb)   # Step 9.2.2.c.2
                                    st.success(f"✅ 批次 **{cb.batch_id}** 复核通过,已入库")
                                    st.balloons()
                                    st.rerun()
                                except StateTransitionError as e:
                                    st.error(f"操作失败:{e}")

                    with rv_col2:
                        with st.popover("❌ 复核拒收", use_container_width=True):
                            st.markdown("**拒收原因(必填)**")
                            reject_reason2 = st.text_area(
                                "请详细说明拒收原因",
                                key="reject_reason_review",
                                placeholder="例如:次品率过高,与供应商申报不符..."
                            )
                            if st.button("确认拒收", type="primary",
                                         key="btn_reject_review_confirm"):
                                if not reject_reason2.strip():
                                    st.error("拒收必须填写原因")
                                else:
                                    try:
                                        svc_review_reject(cb, current_user,
                                                          reason=reject_reason2)
                                        _batch_repo.save(cb)   # Step 9.2.2.c.2
                                        st.success(f"❌ 批次 **{cb.batch_id}** 已拒收")
                                        st.rerun()
                                    except StateTransitionError as e:
                                        st.error(f"操作失败:{e}")

                    st.caption(
                        f"🟡 当前用户【{current_user.role.display_name}】可对此批次进行复核操作。"
                    )
                else:
                    st.info(
                        f"🟡 该批次需要**经理或管理员**复核后方可入库。"
                        f"您当前角色【{current_user.role.display_name}】无复核权限。"
                    )
                    st.caption(f"系统判定原因:{cb.status_reason}")

    with _right_col:
        # ══════════════════════════════════════════════
        # 阶段 2.4-A: 实时预览面板(从 current_batch 读取)
        # 三种状态:
        #   1. 无 current_batch         → 引导占位
        #   2. 有 batch 但未检测过      → "请上传图片"
        #   3. 有 batch 且 detection_rounds > 0 → 完整精简面板
        # ══════════════════════════════════════════════
        _rp_batch = st.session_state.get("current_batch")

        # 公共面板外壳头部
        _RP_HEADER = (
            '<div style="background:linear-gradient(135deg,#FFFBF0 0%,#FFF8ED 100%);'
            'border:1.5px solid #F5D99A;border-radius:16px;padding:1.1rem 1.2rem;'
            'box-shadow:0 2px 12px rgba(200,130,0,.06)">'
            '<div style="font-size:0.95rem;font-weight:900;color:#C05C00;'
            'margin-bottom:0.7rem">📊 实时预览面板</div>'
        )
        _RP_FOOTER = '</div>'

        if _rp_batch is None:
            # 状态 1: 引导占位
            st.markdown(
                _RP_HEADER +
                '<div style="font-size:0.78rem;color:#A07040;line-height:1.7;'
                'border-bottom:1px dashed #F5D99A;padding-bottom:0.7rem;margin-bottom:0.7rem">'
                '本面板将在你 <b>上传图片</b> 后实时显示:<br>'
                '· 检测目标数与平均置信度<br>'
                '· 数量差异与三级分级占比<br>'
                '· 系统自动判定结果(快速通道 / 复核)'
                '</div>'
                '<div style="min-height:240px;display:flex;align-items:center;'
                'justify-content:center;flex-direction:column;color:#C8A060">'
                '<div style="font-size:2.2rem;margin-bottom:0.4rem">📦</div>'
                '<div style="font-size:0.82rem;font-weight:600">等待检测数据</div>'
                '<div style="font-size:0.72rem;margin-top:0.3rem;color:#A07040">'
                '请先在左侧上传水果图片</div>'
                '</div>'
                + _RP_FOOTER,
                unsafe_allow_html=True
            )
        elif _rp_batch.detection_rounds == 0 or _rp_batch.detected_total == 0:
            # 状态 2: 批次已创建但未检测
            st.markdown(
                _RP_HEADER +
                f'<div style="font-size:0.78rem;color:#A07040;line-height:1.6;'
                f'background:#FFF8ED;border-left:3px solid #FF8C00;'
                f'padding:0.5rem 0.8rem;border-radius:6px;margin-bottom:0.7rem">'
                f'已创建批次 <b>{_rp_batch.batch_id}</b>'
                f'</div>'
                f'<div style="min-height:240px;display:flex;align-items:center;'
                f'justify-content:center;flex-direction:column;color:#C8A060">'
                f'<div style="font-size:2.2rem;margin-bottom:0.4rem">📷</div>'
                f'<div style="font-size:0.82rem;font-weight:600">等待图片上传</div>'
                f'<div style="font-size:0.72rem;margin-top:0.3rem;color:#A07040;'
                f'text-align:center;max-width:240px">'
                f'请在左侧选择图片并点击「🚀 开始入库检测」</div>'
                f'</div>'
                + _RP_FOOTER,
                unsafe_allow_html=True
            )
        else:
            # 状态 3: 完整精简面板
            _st_color = _rp_batch.status.color
            _st_name  = _rp_batch.status.display_name
            _diff_c = ("#22C55E" if _rp_batch.count_diff_pct <= 0.03 else
                       "#F59E0B" if _rp_batch.count_diff_pct <= 0.10 else "#EF4444")
            _def_c = ("#22C55E" if _rp_batch.defect_rate <= 0.05 else
                      "#F59E0B" if _rp_batch.defect_rate <= 0.15 else "#EF4444")
            _diff_sign = "+" if _rp_batch.count_diff > 0 else ""

            # 状态原因截短(避免过长撑爆窄列)
            _reason_short = (_rp_batch.status_reason[:60] + "...") \
                if _rp_batch.status_reason and len(_rp_batch.status_reason) > 60 \
                else (_rp_batch.status_reason or "")

            st.markdown(
                _RP_HEADER +
                # 状态徽章 + 批次号(突出显示)
                f'<div style="text-align:center;margin-bottom:0.8rem">'
                f'<div style="display:inline-block;background:{_st_color};color:white;'
                f'padding:0.5rem 1.4rem;border-radius:50px;font-size:1.05rem;'
                f'font-weight:900;letter-spacing:1px;'
                f'box-shadow:0 3px 12px {_st_color}55">{_st_name}</div>'
                f'<div style="font-size:0.72rem;color:#A07040;margin-top:0.4rem">'
                f'批次 <b style="color:#3D2B1F">{_rp_batch.batch_id}</b></div>'
                f'</div>'
                # 系统判定原因
                + (f'<div style="font-size:0.72rem;color:#6B4F2F;line-height:1.5;'
                   f'background:#FFF8ED;border-left:3px solid {_st_color};'
                   f'padding:0.45rem 0.7rem;border-radius:6px;margin-bottom:0.7rem">'
                   f'{_reason_short}</div>' if _reason_short else '') +
                # 4 项核心 KPI(2x2 网格)
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;'
                f'margin-bottom:0.7rem">'
                # 申报→实际
                f'<div style="background:#FFFFFF;border:1px solid #F0EAD6;border-radius:10px;'
                f'padding:0.55rem;text-align:center">'
                f'<div style="font-size:1rem;font-weight:900;color:#FF8C00">'
                f'{_rp_batch.declared_count}→{_rp_batch.detected_total}</div>'
                f'<div style="font-size:0.66rem;color:#A07040;margin-top:2px">申报→实际</div>'
                f'</div>'
                # 数量差异
                f'<div style="background:#FFFFFF;border:1px solid #F0EAD6;border-radius:10px;'
                f'padding:0.55rem;text-align:center">'
                f'<div style="font-size:1rem;font-weight:900;color:{_diff_c}">'
                f'{_diff_sign}{_rp_batch.count_diff} '
                f'<span style="font-size:0.7rem">({_rp_batch.count_diff_pct*100:.1f}%)</span></div>'
                f'<div style="font-size:0.66rem;color:#A07040;margin-top:2px">数量差异</div>'
                f'</div>'
                # 合格率
                f'<div style="background:#FFFFFF;border:1px solid #F0EAD6;border-radius:10px;'
                f'padding:0.55rem;text-align:center">'
                f'<div style="font-size:1rem;font-weight:900;color:#22C55E">'
                f'{_rp_batch.qualified_rate*100:.1f}%</div>'
                f'<div style="font-size:0.66rem;color:#A07040;margin-top:2px">合格率(A+B)</div>'
                f'</div>'
                # 次品率
                f'<div style="background:#FFFFFF;border:1px solid #F0EAD6;border-radius:10px;'
                f'padding:0.55rem;text-align:center">'
                f'<div style="font-size:1rem;font-weight:900;color:{_def_c}">'
                f'{_rp_batch.defect_rate*100:.1f}%</div>'
                f'<div style="font-size:0.66rem;color:#A07040;margin-top:2px">次品率(C)</div>'
                f'</div>'
                f'</div>'
                # ABC 分级 一行三小卡
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.4rem">'
                f'<div style="background:#F0FFF4;border-left:3px solid #22C55E;'
                f'border-radius:8px;padding:0.4rem;text-align:center">'
                f'<div style="font-size:1rem;font-weight:900;color:#22C55E">'
                f'{_rp_batch.grade_a_count}</div>'
                f'<div style="font-size:0.66rem;color:#A07040">🟢 A 优品</div></div>'
                f'<div style="background:#FFFBEB;border-left:3px solid #F59E0B;'
                f'border-radius:8px;padding:0.4rem;text-align:center">'
                f'<div style="font-size:1rem;font-weight:900;color:#F59E0B">'
                f'{_rp_batch.grade_b_count}</div>'
                f'<div style="font-size:0.66rem;color:#A07040">🟡 B 普通</div></div>'
                f'<div style="background:#FFF1F2;border-left:3px solid #EF4444;'
                f'border-radius:8px;padding:0.4rem;text-align:center">'
                f'<div style="font-size:1rem;font-weight:900;color:#EF4444">'
                f'{_rp_batch.grade_c_count}</div>'
                f'<div style="font-size:0.66rem;color:#A07040">🔴 C 次品</div></div>'
                f'</div>'
                # 平均置信度小字
                f'<div style="font-size:0.7rem;color:#A07040;text-align:center;'
                f'margin-top:0.5rem;padding-top:0.5rem;border-top:1px dashed #F5D99A">'
                f'平均置信度 <b style="color:#7C3AED">{_rp_batch.avg_confidence:.3f}</b> '
                f'· 共检测 <b>{_rp_batch.detection_rounds}</b> 轮'
                f'</div>'
                + _RP_FOOTER,
                unsafe_allow_html=True
            )
with T["history"]:
    st.markdown(
        '<div style="font-size:1.1rem;font-weight:900;color:#C05C00;margin-bottom:0.2rem">'
        '🔍 检测日志</div>'
        '<div style="font-size:0.82rem;color:#A07040;margin-bottom:0.8rem">'
        '记录每次检测调用的明细(单图/视频/批量),用于复盘和趋势分析。'
        '与「📜 入库历史」(批次级)互补。</div>',
        unsafe_allow_html=True
    )
    hist = st.session_state.history

    if not hist:
        st.markdown('<div class="empty-state"><div style="font-size:2.5rem">🔍</div>'
                    '<div style="font-weight:700;margin-top:.5rem">暂无检测日志</div>'
                    '<div style="font-size:.8rem;margin-top:.3rem">'
                    '完成任意检测后自动记录在此</div></div>',
                    unsafe_allow_html=True)
    else:
        hc1,hc2,hc3 = st.columns(3)
        total_det = sum(r["total"] for r in hist)
        total_A   = sum(r["A"] for r in hist)
        avg_lat   = np.mean([r["latency"] for r in hist if r["latency"]>0]) if any(r["latency"]>0 for r in hist) else 0
        hc1.metric("累计检测批次", len(hist))
        hc2.metric("累计检测目标", total_det)
        hc3.metric("平均推理时延", f"{avg_lat:.1f} ms")

        st.markdown("---")

        # 趋势图
        trend = chart_history_trend()
        if trend:
            st.markdown("**等级趋势**")
            st.plotly_chart(trend,               use_container_width=True, key="hist_trend")

        st.markdown("**历史明细**")
        for r in hist:
            total_r = r["A"]+r["B"]+r["C"]
            a_pct = f"{r['A']/total_r*100:.0f}%" if total_r>0 else "—"
            st.markdown(
                f'<div class="hist-row">'
                f'<div>'
                f'<b>{r["filename"]}</b> &nbsp;'
                f'<span class="badge" style="background:#0EA5E9">{r["model"].split()[0]}</span>'
                f'<br><span style="color:#A07040;font-size:.78rem">目标:{r["total"]} &nbsp; '
                f'🟢{r["A"]} 🟡{r["B"]} 🔴{r["C"]} &nbsp; 一级率:{a_pct}</span>'
                f'</div>'
                f'<div style="text-align:right">'
                f'<div class="hist-time">{r["time"]}</div>'
                f'<div style="font-size:.8rem;color:#FF8C00;font-weight:700">{r["latency"]:.0f} ms</div>'
                f'</div>'
                f'</div>', unsafe_allow_html=True)

        # 导出历史
        import pandas as pd
        df_h = pd.DataFrame(hist)
        csv_h = df_h.to_csv(index=False, encoding="utf-8-sig")
        col_dl, col_clr = st.columns([3,1])
        with col_dl:
            st.download_button("⬇️ 导出历史记录 CSV", csv_h,
                               file_name="detection_history.csv", mime="text/csv")
        with col_clr:
            if st.button("🗑️ 清空历史"):
                st.session_state.history = []; st.rerun()


# ══════════════════════════════════════════════
# Tab6: 货架看板 ★新增★
# ══════════════════════════════════════════════
with T["board"]:
    st.markdown(
        '<div style="font-size:1.1rem;font-weight:900;color:#C05C00;margin-bottom:0.2rem">📊 本批次入库汇总</div>',
        unsafe_allow_html=True
    )
    bt    = st.session_state.board_total
    total = sum(bt.values())

    if total == 0:
        st.markdown('<div class="empty-state"><div style="font-size:2.5rem">📊</div>'
                    '<div style="font-weight:700;margin-top:.5rem">看板数据为空</div>'
                    '<div style="font-size:.8rem;margin-top:.3rem">检测图片/视频后自动汇总到看板</div></div>',
                    unsafe_allow_html=True)
    else:
        # 三大分拣道
        st.markdown("#### 🏭 实时分拣道状态")
        bc1,bc2,bc3 = st.columns(3)
        board_cards = [
            (bc1, "A", "#22C55E", "#F0FFF4", "直销 / 礼盒"),
            (bc2, "B", "#F59E0B", "#FFFBEB", "超市 / 批发"),
            (bc3, "C", "#EF4444", "#FFF1F2", "降级 / 废弃"),
        ]
        grade_names = {"A":"一级果","B":"二级果","C":"三级果"}
        for col, g, color, bg, action in board_cards:
            pct = f"{bt[g]/total*100:.1f}%" if total>0 else "0%"
            col.markdown(
                f'<div class="board-card" style="border-color:{color};background:{bg}">'
                f'<div style="font-size:2rem">'
                f'{"🟢" if g=="A" else "🟡" if g=="B" else "🔴"}</div>'
                f'<div class="board-num" style="color:{color}">{bt[g]}</div>'
                f'<div class="board-lbl" style="color:{color}">{grade_names[g]}</div>'
                f'<div class="board-pct">{pct} · {action}</div>'
                f'</div>', unsafe_allow_html=True)

        st.markdown("---")
        bd1, bd2 = st.columns([1,1])
        with bd1:
            st.markdown("**整体分级占比**")
            fig_b = chart_board_pie()
            if fig_b: st.plotly_chart(fig_b,               use_container_width=True, key="board_pie")

        with bd2:
            st.markdown("**分品类统计**")
            if st.session_state.board_fruits:
                sm_b = {"total":total,"A":bt["A"],"B":bt["B"],"C":bt["C"],
                        "by_fruit":st.session_state.board_fruits,
                        "avg_confidence":0,"dominant_grade":"A"}
                st.plotly_chart(chart_bar(sm_b),     use_container_width=True, key="board_bar")

        # 批次记录表
        if st.session_state.board_records:
            st.markdown("**批次入库记录**")
            import pandas as pd
            df_b = pd.DataFrame(st.session_state.board_records)
            st.dataframe(df_b, use_container_width=True)
            csv_b = df_b.to_csv(index=False, encoding="utf-8-sig")
            bc_dl, bc_clr = st.columns([3,1])
            with bc_dl:
                st.download_button("⬇️ 导出看板数据", csv_b,
                                   file_name="board_data.csv", mime="text/csv")
            with bc_clr:
                if st.button("🔄 重置看板"):
                    st.session_state.board_total   = {"A":0,"B":0,"C":0}
                    st.session_state.board_fruits  = {}
                    st.session_state.board_records = []
                    st.rerun()


# ══════════════════════════════════════════════
# Tab7: 模型性能对比 ★新增★
# ══════════════════════════════════════════════
with T["compare"]:
    st.markdown(
        '<div style="font-size:1.1rem;font-weight:900;color:#C05C00;margin-bottom:0.2rem">⚙️ 系统设置 · 模型性能对比</div>',
        unsafe_allow_html=True
    )
    st.markdown("基于相同验证集（480张）和测试环境（AMD Ryzen 7 5800H · CPU推理）的实测数据")
    st.markdown("---")

    # 对比表格
    rows_cmp = []
    for name, p in MODEL_PERF.items():
        is_best = "ONNX FP32" in name
        tag     = '<span class="cmp-tag">推荐</span>' if is_best else ""
        rows_cmp.append((name, tag, p))

    table_html = """
    <table class="cmp-table">
    <tr>
        <th>模型</th><th>大小(MB)</th><th>时延(ms)</th>
        <th>FPS</th><th>mAP@0.5</th><th>参数量</th><th>GFLOPs</th>
    </tr>"""
    for name, tag, p in rows_cmp:
        is_h = "ONNX FP32" in name
        cls  = 'class="highlight"' if is_h else ""
        fps_str = f'<span class="cmp-best">{p["fps"]}</span>' if is_h else str(p["fps"])
        lat_str = f'<span class="cmp-best">{p["latency_ms"]}</span>' if is_h else str(p["latency_ms"])
        table_html += f"""
        <tr {cls}>
            <td><b>{name}</b> {tag}</td>
            <td>{p["size_mb"]}</td>
            <td>{lat_str}</td>
            <td>{fps_str}</td>
            <td>{p["map"]}</td>
            <td>{p["params"]}</td>
            <td>{p["gflops"]}</td>
        </tr>"""
    table_html += "</table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("---")

    # 可视化对比图
    cc1,cc2 = st.columns(2)
    names  = list(MODEL_PERF.keys())
    colors = ["#F0EAD6","#FF8C00","#F0EAD6","#F0EAD6"]

    with cc1:
        st.markdown("**推理时延对比 (ms，越低越好)**")
        fig_l = go.Figure(go.Bar(
            x=names,
            y=[p["latency_ms"] for p in MODEL_PERF.values()],
            marker_color=colors,
            marker_line=dict(color="#FFFFFF",width=2),
            text=[f'{p["latency_ms"]}ms' for p in MODEL_PERF.values()],
            textposition="outside",
        ))
        fig_l.update_layout(paper_bgcolor=CHART_BG,plot_bgcolor=CHART_BG,
                            font_color=CHART_FONT,showlegend=False,
                            yaxis=dict(gridcolor="#F0EAD6"),
                            margin=dict(t=20,b=10,l=30,r=10),height=280)
        st.plotly_chart(fig_l, use_container_width=True)

    with cc2:
        st.markdown("**FPS 对比（越高越好）**")
        fig_f = go.Figure(go.Bar(
            x=names,
            y=[p["fps"] for p in MODEL_PERF.values()],
            marker_color=colors,
            marker_line=dict(color="#FFFFFF",width=2),
            text=[f'{p["fps"]} FPS' for p in MODEL_PERF.values()],
            textposition="outside",
        ))
        fig_f.update_layout(paper_bgcolor=CHART_BG,plot_bgcolor=CHART_BG,
                            font_color=CHART_FONT,showlegend=False,
                            yaxis=dict(gridcolor="#F0EAD6"),
                            margin=dict(t=20,b=10,l=30,r=10),height=280)
        st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Noto+Sans+SC:wght@400;500;700&display=swap');

    /* 全局 */
    .stApp { background:#FDFAF5; font-family:'Noto Sans SC','Nunito',sans-serif; font-size:0.82rem; }
    #MainMenu,footer,header{ visibility:hidden; }
    .stDeployButton{ display:none; }

    /* 隐藏默认侧边栏 */
    section[data-testid="stSidebar"] { display:none; }

    /* 顶部导航栏 */
    .top-nav {
        display:flex; align-items:center; justify-content:space-between;
        padding:0.6rem 1.5rem; background:#FFFFFF;
        border-bottom:2px solid #F5D99A;
        position:sticky; top:0; z-index:999;
    }
    .top-nav-left { display:flex; align-items:center; gap:0.7rem; }
    .top-nav-logo { font-size:1.6rem; line-height:1; }
    .top-nav-title {
        font-size:1rem; font-weight:900; color:#C05C00;
        font-family:'Nunito','Noto Sans SC',sans-serif; line-height:1.2;
    }
    .top-nav-sub { font-size:0.72rem; color:#A07040; font-weight:400; }
    .top-nav-right { font-size:0.75rem; color:#A07040; }

    /* 参数控制区 */
    .param-bar {
        background:#FFF8ED; border-bottom:1.5px solid #F5D99A;
        padding:0.4rem 1.5rem; display:flex; gap:2rem; align-items:center;
        font-size:0.8rem; color:#3D2B1F;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background:#FFF3E0; border-radius:50px; padding:4px; gap:2px;
        border:1.5px solid #F5D99A; margin:0.8rem 0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius:50px !important; padding:.35rem 1rem !important;
        font-weight:700 !important; color:#A07040 !important;
        background:transparent !important; font-size:.82rem !important;
    }
    .stTabs [aria-selected="true"] { background:#FF8C00 !important; color:white !important; }

    /* 卡片 */
    .kpi-row { display:flex; gap:10px; margin:.8rem 0; }
    .kpi-card {
        flex:1; background:#FFFFFF; border-radius:16px; padding:.9rem 1rem;
        text-align:center; border:1.5px solid #F0EAD6;
        box-shadow:0 2px 10px rgba(200,130,0,.07);
    }
    .kpi-num  { font-size:1.8rem; font-weight:900; line-height:1.1; }
    .kpi-lbl  { font-size:.72rem; color:#A07040; margin-top:3px; font-weight:600; }

    /* 分拣建议 */
    .action-card {
        background:#FFFFFF; border-radius:12px; padding:.8rem 1rem; margin:.4rem 0;
        border:1.5px solid #F0EAD6; border-left-width:5px;
        box-shadow:0 2px 8px rgba(0,0,0,.04); font-size:.86rem; color:#3D2B1F;
    }
    .badge {
        display:inline-block; padding:2px 11px; border-radius:50px;
        font-weight:800; font-size:.78rem; color:white; vertical-align:middle;
    }

    /* 历史记录行 */
    .hist-row {
        background:#FFFFFF; border-radius:12px; padding:.7rem 1rem; margin:.35rem 0;
        border:1.5px solid #F0EAD6; font-size:.84rem; color:#3D2B1F;
        display:flex; justify-content:space-between; align-items:center;
    }
    .hist-time { color:#A07040; font-size:.76rem; }

    /* 看板大卡 */
    .board-card {
        background:#FFFFFF; border-radius:20px; padding:1.5rem;
        text-align:center; border:2px solid; box-shadow:0 4px 20px rgba(0,0,0,.07);
    }
    .board-num  { font-size:3.5rem; font-weight:900; line-height:1; }
    .board-lbl  { font-size:.9rem; font-weight:700; margin-top:.4rem; }
    .board-pct  { font-size:.8rem; color:#A07040; margin-top:.2rem; }

    /* 模型对比表格 */
    .cmp-table {
        width:100%; border-collapse:collapse; font-size:.86rem;
        border-radius:12px; overflow:hidden;
    }
    .cmp-table th {
        background:#FFF3E0; color:#C05C00; font-weight:800;
        padding:.6rem .9rem; text-align:center; border-bottom:2px solid #F5D99A;
    }
    .cmp-table td {
        padding:.55rem .9rem; text-align:center;
        border-bottom:1px solid #F5EDD8; color:#3D2B1F;
    }
    .cmp-table tr:hover td { background:#FFFBF0; }
    .cmp-table .highlight td { background:#FFF8ED; font-weight:700; }
    .cmp-best { color:#22C55E; font-weight:900; }
    .cmp-tag {
        display:inline-block; background:#FF8C00; color:white;
        border-radius:50px; padding:1px 8px; font-size:.72rem; font-weight:700;
    }

    /* 上传区 */
    [data-testid="stFileUploader"] {
        border:2.5px dashed #F5B800 !important; border-radius:16px !important;
        background:#FFFBF0 !important; padding:1rem !important;
    }

    /* 按钮 */
    .stButton>button {
        background:linear-gradient(135deg,#FF8C00,#FFA500) !important;
        color:white !important; border:none !important; border-radius:50px !important;
        font-weight:700 !important; padding:.45rem 1.6rem !important;
        box-shadow:0 3px 12px rgba(255,140,0,.3) !important;
    }
    .stButton>button:hover {
        transform:translateY(-1px) !important;
        box-shadow:0 6px 18px rgba(255,140,0,.4) !important;
    }

    [data-baseweb="select"]>div {
        background:#FFFBF0 !important; border-color:#F5D99A !important; border-radius:10px !important;
    }
    .stProgress>div>div>div>div {
        background:linear-gradient(90deg,#FF8C00,#FFC300) !important;
    }
    hr { border-color:#F0EAD6 !important; margin:.8rem 0 !important; }

    .grade-info {
        background:#FFFBF0; border-radius:10px; padding:.65rem .85rem;
        margin:.35rem 0; border-left:4px solid; font-size:.8rem;
    }
    .empty-state {
        height:260px; display:flex; flex-direction:column;
        align-items:center; justify-content:center;
        background:#FFFBF0; border-radius:16px;
        border:2px dashed #F5D99A; color:#C8A060;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Step 4.4: 复核中心(仅 manager/admin 可见)
# ══════════════════════════════════════════════
if "review" in T:
    with T["review"]:
        from services.batch_service import (
            review_approve as _svc_review_approve,
            review_reject as _svc_review_reject,
            StateTransitionError as _StateTransitionError,
        )
        from services.permission_service import (
            can_review_batch as _can_review,
            can_reject_batch as _can_reject,
        )

        st.markdown(
            '<div style="font-size:1.1rem;font-weight:900;color:#C05C00;margin-bottom:0.2rem">'
            '🔍 复核中心</div>'
            '<div style="font-size:0.82rem;color:#A07040;margin-bottom:0.8rem">'
            '处理所有等待人工介入的批次,通过或拒收以完成入库流程。</div>',
            unsafe_allow_html=True
        )

        # ── 最近操作记录卡片(从 batch.review_snapshot 读取,rerun/换登录后保留)──
        _acted = [b for b in st.session_state.all_batches if b.review_snapshot]
        if _acted:
            def _act_time_of(b):
                return b.rejected_at or b.reviewed_at or datetime.min
            _last_b = max(_acted, key=_act_time_of)
            _is_approve = (_last_b.review_snapshot.get("action") == "approved")
            _act_color = "#16A34A" if _is_approve else "#EF4444"
            _act_label = "复核通过" if _is_approve else "已拒收"
            _act_icon  = "✅" if _is_approve else "❌"
            _act_t = _act_time_of(_last_b)
            _act_time_str = _act_t.strftime("%H:%M:%S") if _act_t != datetime.min else "—"
            _act_user = _last_b.reviewed_by_name or "—"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;'
                f'background:#FFFFFF;border-radius:14px;padding:0.7rem 1.1rem;'
                f'border:1.5px solid #F0EAD6;border-left:5px solid {_act_color};'
                f'margin-bottom:0.9rem;font-size:0.85rem">'
                f'<span style="font-size:0.7rem;color:#A07040;font-weight:600">最近操作</span>'
                f'<span style="font-weight:800;color:#3D2B1F">{_act_icon} {_act_label}</span>'
                f'<span style="font-weight:700;color:#3D2B1F">{_last_b.batch_id}</span>'
                f'<span style="color:#A07040;font-size:0.78rem">·  {_act_time_str}'
                f'  ·  操作人 {_act_user}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── 筛出待 manager 处理的批次:PENDING_REVIEW(必须复核)+ READY_TO_CONFIRM(兜底)──
        # 排序规则:PENDING_REVIEW 优先(必须经理出手),其次 READY_TO_CONFIRM,组内按入库时间倒序
        _status_priority = {BatchStatus.PENDING_REVIEW: 0, BatchStatus.READY_TO_CONFIRM: 1}
        _pending = sorted(
            [b for b in st.session_state.all_batches
             if b.status in {BatchStatus.PENDING_REVIEW, BatchStatus.READY_TO_CONFIRM}],
            key=lambda b: (_status_priority[b.status],
                           -(b.inbound_date.timestamp() if b.inbound_date else 0))
        )

        if not _pending:
            st.markdown(
                '<div class="empty-state"><div style="font-size:2.5rem">🎉</div>'
                '<div style="font-weight:700;margin-top:.5rem">当前没有待处理批次</div>'
                '<div style="font-size:.8rem;margin-top:.3rem">'
                '所有批次均已处理完毕,或暂无新批次进入流程</div></div>',
                unsafe_allow_html=True
            )
        else:
            _n_review = sum(1 for b in _pending if b.status == BatchStatus.PENDING_REVIEW)
            _n_ready  = sum(1 for b in _pending if b.status == BatchStatus.READY_TO_CONFIRM)
            st.markdown(
                f'<div style="display:flex;gap:14px;margin-bottom:0.6rem;font-size:0.88rem">'
                f'<span style="background:#FFF8ED;border:1.5px solid #F59E0B;'
                f'border-left:5px solid #F59E0B;border-radius:10px;'
                f'padding:0.4rem 0.9rem;color:#3D2B1F">'
                f'🟡 <b>待复核</b> {_n_review} 个'
                f'</span>'
                f'<span style="background:#F0FFF4;border:1.5px solid #22C55E;'
                f'border-left:5px solid #22C55E;border-radius:10px;'
                f'padding:0.4rem 0.9rem;color:#3D2B1F">'
                f'🟢 <b>待确认</b> {_n_ready} 个'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            if _n_ready > 0:
                st.caption(
                    "ℹ️ 「待确认」批次本应由 operator 自助处理,"
                    "经理可在此代为兜底处理(如 operator 长时间未操作)。"
                )

            # ── radio 单选列表 ──
            def _fmt_batch_label(b):
                _diff_sign = "+" if b.count_diff > 0 else ""
                # 6.2-fix-2: 加状态前缀,便于区分必复核 vs 可兜底
                _prefix = "🟡 待复核" if b.status == BatchStatus.PENDING_REVIEW else "🟢 待确认"
                return (
                    f"[{_prefix}]  {b.batch_id}  ·  {b.supplier_name or '(未填供应商)'}"
                    f"  ·  {b.fruit_category}"
                    f"  ·  申报{b.declared_count}/实际{b.detected_total}"
                    f"  ({_diff_sign}{b.count_diff_pct*100:.1f}%)"
                    f"  ·  次品率 {b.defect_rate*100:.1f}%"
                )

            _options = [b.batch_id for b in _pending]

            # 默认选中:如果之前有选中且仍在列表里,保持;否则选第一个
            _prev = st.session_state.get("selected_review_batch")
            _default_idx = _options.index(_prev) if _prev in _options else 0

            _selected_id = st.radio(
                "选择要复核的批次",
                options=_options,
                format_func=lambda bid: _fmt_batch_label(
                    next(b for b in _pending if b.batch_id == bid)
                ),
                index=_default_idx,
                key="review_radio",
                label_visibility="collapsed",
            )
            st.session_state.selected_review_batch = _selected_id

            # ── 选中批次详情 ──
            _sel = next(b for b in _pending if b.batch_id == _selected_id)

            st.markdown("---")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:14px;'
                f'background:#FFFBF0;border-radius:16px;padding:1.1rem 1.4rem;'
                f'border:1.5px solid #F59E0B;border-left:6px solid #F59E0B;'
                f'margin-bottom:0.8rem">'
                f'<div style="font-size:1.4rem;font-weight:900;color:#F59E0B;'
                f'background:white;padding:0.35rem 1.1rem;border-radius:50px;'
                f'border:2px solid #F59E0B;white-space:nowrap">待复核</div>'
                f'<div style="flex:1;font-size:0.88rem;color:#3D2B1F;line-height:1.5">'
                f'<b>系统判定原因:</b>{_sel.status_reason}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # 详情 KPI(精简版,4 列)
            _diff_color = ("#22C55E" if _sel.count_diff_pct <= 0.03 else
                           "#F59E0B" if _sel.count_diff_pct <= 0.10 else "#EF4444")
            _defect_color = ("#22C55E" if _sel.defect_rate <= 0.05 else
                             "#F59E0B" if _sel.defect_rate <= 0.15 else "#EF4444")
            _sign = "+" if _sel.count_diff > 0 else ""

            _kc1, _kc2, _kc3, _kc4 = st.columns(4)
            with _kc1:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-num" style="color:#FF8C00">'
                    f'{_sel.declared_count} → {_sel.detected_total}</div>'
                    f'<div class="kpi-lbl">申报 → 实际</div></div>',
                    unsafe_allow_html=True
                )
            with _kc2:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-num" style="color:{_diff_color}">'
                    f'{_sign}{_sel.count_diff} '
                    f'<span style="font-size:1rem">({_sel.count_diff_pct*100:.1f}%)</span></div>'
                    f'<div class="kpi-lbl">数量差异</div></div>',
                    unsafe_allow_html=True
                )
            with _kc3:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-num" style="color:#22C55E">'
                    f'{_sel.qualified_rate*100:.1f}%</div>'
                    f'<div class="kpi-lbl">合格率(A+B)</div></div>',
                    unsafe_allow_html=True
                )
            with _kc4:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-num" style="color:{_defect_color}">'
                    f'{_sel.defect_rate*100:.1f}%</div>'
                    f'<div class="kpi-lbl">次品率(C)</div></div>',
                    unsafe_allow_html=True
                )

            # 分级数量
            st.markdown("")
            _gc1, _gc2, _gc3 = st.columns(3)
            with _gc1:
                st.markdown(
                    f'<div class="kpi-card" style="border-left:4px solid #22C55E">'
                    f'<div class="kpi-num" style="color:#22C55E">{_sel.grade_a_count}</div>'
                    f'<div class="kpi-lbl">🟢 A 级优品</div></div>',
                    unsafe_allow_html=True
                )
            with _gc2:
                st.markdown(
                    f'<div class="kpi-card" style="border-left:4px solid #F59E0B">'
                    f'<div class="kpi-num" style="color:#F59E0B">{_sel.grade_b_count}</div>'
                    f'<div class="kpi-lbl">🟡 B 级普通果</div></div>',
                    unsafe_allow_html=True
                )
            with _gc3:
                st.markdown(
                    f'<div class="kpi-card" style="border-left:4px solid #EF4444">'
                    f'<div class="kpi-num" style="color:#EF4444">{_sel.grade_c_count}</div>'
                    f'<div class="kpi-lbl">🔴 C 级次品</div></div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.markdown("#### ⚖️ 复核操作")

            # ── 权限守卫 ──
            if not _can_review(current_user, _sel.status):
                st.error(
                    f"权限不足:您当前角色【{current_user.role.display_name}】无复核权限。"
                )
            else:
                _rv1, _rv2 = st.columns([1, 1])
                with _rv1:
                    with st.popover("✅ 复核通过", use_container_width=True):
                        st.markdown("**复核备注(可选)**")
                        _rv_note = st.text_area(
                            "实物核验情况说明",
                            key=f"rc_note_{_sel.batch_id}",
                            placeholder="例如:已抽检 10 个,实物状态良好,准予入库"
                        )
                        if st.button("确认通过", type="primary",
                                     key=f"rc_btn_approve_{_sel.batch_id}"):
                            try:
                                _svc_review_approve(_sel, current_user, note=_rv_note)
                                _batch_repo.save(_sel)   # Step 9.2.2.c.2
                                # 复核痕迹已通过 batch.review_snapshot + reviewed_by_name 持久化
                                # 选中状态清掉,因为该批次已离开 PENDING_REVIEW 列表
                                st.session_state.selected_review_batch = None
                                st.rerun()
                            except _StateTransitionError as e:
                                st.error(f"操作失败:{e}")

                with _rv2:
                    with st.popover("❌ 复核拒收", use_container_width=True):
                        st.markdown("**拒收原因(必填)**")
                        _rj_reason = st.text_area(
                            "请详细说明拒收原因",
                            key=f"rc_reason_{_sel.batch_id}",
                            placeholder="例如:次品率过高,与供应商申报不符..."
                        )
                        if st.button("确认拒收", type="primary",
                                     key=f"rc_btn_reject_{_sel.batch_id}"):
                            if not _rj_reason.strip():
                                st.error("拒收必须填写原因")
                            else:
                                try:
                                    _svc_review_reject(_sel, current_user, reason=_rj_reason)
                                    _batch_repo.save(_sel)   # Step 9.2.2.c.2
                                    # 复核痕迹已通过 batch.review_snapshot + reviewed_by_name 持久化
                                    st.session_state.selected_review_batch = None
                                    st.rerun()
                                except _StateTransitionError as e:
                                    st.error(f"操作失败:{e}")

                st.caption(
                    f"🟡 当前用户【{current_user.role.display_name}】可对此批次进行复核操作。"
                )



# ══════════════════════════════════════════════
# Step 6.2: 入库历史 Tab(所有人可见,只读展示)
# ══════════════════════════════════════════════
with T["inbound_history"]:
    st.markdown(
        '<div style="font-size:1.1rem;font-weight:900;color:#C05C00;margin-bottom:0.2rem">'
        '📜 入库历史</div>'
        '<div style="font-size:0.82rem;color:#A07040;margin-bottom:0.8rem">'
        '查看系统中所有批次的入库流转记录(包含已入库/已拒收/待复核/待确认)。</div>',
        unsafe_allow_html=True
    )

    _all = st.session_state.all_batches

    if not _all:
        st.markdown(
            '<div class="empty-state"><div style="font-size:2.5rem">📜</div>'
            '<div style="font-weight:700;margin-top:.5rem">暂无批次记录</div>'
            '<div style="font-size:.8rem;margin-top:.3rem">'
            '完成入库检测后,批次会自动出现在这里</div></div>',
            unsafe_allow_html=True
        )
    else:
        # ── 顶部 KPI:状态计数 ──
        _cnt_total     = len(_all)
        _cnt_confirmed = sum(1 for b in _all if b.status == BatchStatus.CONFIRMED)
        _cnt_rejected  = sum(1 for b in _all if b.status == BatchStatus.REJECTED)
        _cnt_pending   = sum(1 for b in _all if b.status == BatchStatus.PENDING_REVIEW)
        _cnt_ready     = sum(1 for b in _all if b.status == BatchStatus.READY_TO_CONFIRM)

        _ih_k1, _ih_k2, _ih_k3, _ih_k4, _ih_k5 = st.columns(5)
        with _ih_k1:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-num" style="color:#FF8C00">{_cnt_total}</div>'
                f'<div class="kpi-lbl">批次总数</div></div>',
                unsafe_allow_html=True
            )
        with _ih_k2:
            st.markdown(
                f'<div class="kpi-card" style="border-left:4px solid #16A34A">'
                f'<div class="kpi-num" style="color:#16A34A">{_cnt_confirmed}</div>'
                f'<div class="kpi-lbl">✅ 已入库</div></div>',
                unsafe_allow_html=True
            )
        with _ih_k3:
            st.markdown(
                f'<div class="kpi-card" style="border-left:4px solid #EF4444">'
                f'<div class="kpi-num" style="color:#EF4444">{_cnt_rejected}</div>'
                f'<div class="kpi-lbl">❌ 已拒收</div></div>',
                unsafe_allow_html=True
            )
        with _ih_k4:
            st.markdown(
                f'<div class="kpi-card" style="border-left:4px solid #F59E0B">'
                f'<div class="kpi-num" style="color:#F59E0B">{_cnt_pending}</div>'
                f'<div class="kpi-lbl">🟡 待复核</div></div>',
                unsafe_allow_html=True
            )
        with _ih_k5:
            st.markdown(
                f'<div class="kpi-card" style="border-left:4px solid #22C55E">'
                f'<div class="kpi-num" style="color:#22C55E">{_cnt_ready}</div>'
                f'<div class="kpi-lbl">🟢 待确认</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ══════════════════════════════════════════════
        # Step 6.4-2: 筛选区(一行四列)
        # ══════════════════════════════════════════════
        st.markdown("#### 🔎 筛选")

        _f_col1, _f_col2, _f_col3, _f_col4 = st.columns([1.5, 1.2, 1.6, 1.2])

        # ── 状态多选(默认全选)──
        _all_status_options = [
            BatchStatus.DRAFT,
            BatchStatus.READY_TO_CONFIRM,
            BatchStatus.PENDING_REVIEW,
            BatchStatus.CONFIRMED,
            BatchStatus.REJECTED,
        ]
        with _f_col1:
            _f_status = st.multiselect(
                "状态",
                options=_all_status_options,
                default=_all_status_options,
                format_func=lambda s: s.display_name,
                key="ih_filter_status",
            )

        # ── 供应商下拉 ──
        _suppliers_set = {(b.supplier_name or "(未填)") for b in _all}
        _suppliers_list = ["全部"] + sorted(_suppliers_set)
        with _f_col2:
            _f_supplier = st.selectbox(
                "供应商",
                options=_suppliers_list,
                index=0,
                key="ih_filter_supplier",
            )

        # ── 日期范围 ──
        with _f_col3:
            _f_date_range = st.date_input(
                "日期范围",
                value=(),  # 空元组 = 全部
                key="ih_filter_date",
                help="留空表示不限日期",
            )

        # ── 批次号搜索 ──
        with _f_col4:
            _f_search = st.text_input(
                "批次号搜索",
                value="",
                placeholder="模糊匹配...",
                key="ih_filter_search",
            )

        # ══════════════════════════════════════════════
        # 应用筛选
        # ══════════════════════════════════════════════
        def _apply_filters(batches):
            result = []
            for b in batches:
                # 状态
                if b.status not in _f_status:
                    continue
                # 供应商
                if _f_supplier != "全部":
                    _b_supp = b.supplier_name or "(未填)"
                    if _b_supp != _f_supplier:
                        continue
                # 日期范围
                if isinstance(_f_date_range, tuple) and len(_f_date_range) == 2:
                    _start, _end = _f_date_range
                    if b.inbound_date:
                        _bd = b.inbound_date.date()
                        if _bd < _start or _bd > _end:
                            continue
                # 批次号搜索
                if _f_search.strip():
                    if _f_search.strip().lower() not in b.batch_id.lower():
                        continue
                result.append(b)
            return result

        _filtered = _apply_filters(_all)

        # ── 按入库日期倒序(新的在上)──
        _sorted = sorted(_filtered, key=lambda b: b.inbound_date or datetime.min, reverse=True)

        st.markdown("---")

        if not _sorted:
            st.markdown(
                '<div class="empty-state"><div style="font-size:2.5rem">🔎</div>'
                '<div style="font-weight:700;margin-top:.5rem">当前筛选条件下无批次</div>'
                '<div style="font-size:.8rem;margin-top:.3rem">'
                '请调整筛选项或清空筛选</div></div>',
                unsafe_allow_html=True
            )
        else:
            _total_n = len(_all)
            _filt_n  = len(_sorted)
            if _filt_n == _total_n:
                st.markdown(f"**共 {_filt_n} 个批次**")
            else:
                st.markdown(f"**筛选结果:{_filt_n} / {_total_n} 个批次**")
            st.caption("ℹ️ 当前为只读展示。复核操作请使用「复核中心」Tab。")

            for _b in _sorted:
                _st_color = _b.status.color
                _st_name  = _b.status.display_name

                # 时间戳:优先显示终态时间,否则显示创建时间
                if _b.status == BatchStatus.CONFIRMED and _b.confirmed_at:
                    _time_label = f"入库于 {_b.confirmed_at.strftime('%m-%d %H:%M')}"
                elif _b.status == BatchStatus.REJECTED and _b.rejected_at:
                    _time_label = f"拒收于 {_b.rejected_at.strftime('%m-%d %H:%M')}"
                else:
                    _time_label = f"创建于 {_b.inbound_date.strftime('%m-%d %H:%M')}" if _b.inbound_date else "—"

                # 数量差异颜色
                _diff_c = ("#22C55E" if _b.count_diff_pct <= 0.03 else
                           "#F59E0B" if _b.count_diff_pct <= 0.10 else "#EF4444")
                _diff_sign = "+" if _b.count_diff > 0 else ""

                # 次品率颜色
                _def_c = ("#22C55E" if _b.defect_rate <= 0.05 else
                          "#F59E0B" if _b.defect_rate <= 0.15 else "#EF4444")

                st.markdown(
                    f'<div style="background:#FFFFFF;border-radius:14px;'
                    f'padding:0.85rem 1.1rem;margin-bottom:0.55rem;'
                    f'border:1.5px solid #F0EAD6;border-left:5px solid {_st_color};'
                    f'box-shadow:0 2px 8px rgba(0,0,0,.04);font-size:0.85rem;color:#3D2B1F">'
                    # 第一行:批次号 + 状态徽章 + 时间
                    f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;'
                    f'margin-bottom:0.4rem">'
                    f'<span style="font-weight:800;color:#3D2B1F;font-size:0.95rem">'
                    f'{_b.batch_id}</span>'
                    f'<span style="background:{_st_color};color:white;padding:2px 10px;'
                    f'border-radius:50px;font-weight:700;font-size:0.72rem">{_st_name}</span>'
                    f'<span style="color:#A07040;font-size:0.76rem">· {_time_label}</span>'
                    f'</div>'
                    # 第二行:供应商 + 品类 + 仓库
                    f'<div style="color:#6B4F2F;font-size:0.78rem;margin-bottom:0.45rem">'
                    f'🏭 {_b.supplier_name or "(未填供应商)"}'
                    f'  &nbsp;·&nbsp; 🍎 {_b.fruit_category or "—"}'
                    f'  &nbsp;·&nbsp; 📦 {_b.warehouse or "—"} / {_b.storage_zone or "—"}'
                    f'</div>'
                    # 第三行:KPI
                    f'<div style="display:flex;gap:18px;flex-wrap:wrap;'
                    f'font-size:0.78rem;color:#3D2B1F">'
                    f'<span>申报→实际 <b style="color:#FF8C00">'
                    f'{_b.declared_count}→{_b.detected_total}</b></span>'
                    f'<span>差异 <b style="color:{_diff_c}">'
                    f'{_diff_sign}{_b.count_diff} ({_b.count_diff_pct*100:.1f}%)</b></span>'
                    f'<span>合格率 <b style="color:#22C55E">{_b.qualified_rate*100:.1f}%</b></span>'
                    f'<span>次品率 <b style="color:{_def_c}">{_b.defect_rate*100:.1f}%</b></span>'
                    f'<span>检测轮数 <b style="color:#7C3AED">{_b.detection_rounds}</b></span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # ══════════════════════════════════════════════
                # Step 6.4-3: 流转痕迹时间线(默认收起)
                # ══════════════════════════════════════════════
                with st.expander(f"📜 查看 {_b.batch_id} 的流转详情", expanded=False):
                    _node_html = []

                    # ── 节点 1: 创建 ──
                    _create_time = _b.inbound_date.strftime("%Y-%m-%d %H:%M:%S") if _b.inbound_date else "—"
                    _node_html.append(
                        f'<div style="display:flex;gap:12px;padding:0.6rem 0;'
                        f'border-left:3px solid #FF8C00;padding-left:14px;margin-left:6px">'
                        f'<div>'
                        f'<div style="font-weight:700;color:#3D2B1F;font-size:0.88rem">'
                        f'📦 创建批次'
                        f'</div>'
                        f'<div style="color:#A07040;font-size:0.78rem;margin-top:2px">'
                        f'{_create_time}  ·  操作人 <b>{_b.operator_name or f"#{_b.operator_id}"}</b>'
                        f'</div>'
                        f'<div style="color:#6B4F2F;font-size:0.76rem;margin-top:3px">'
                        f'申报数量:<b>{_b.declared_count}</b>  ·  '
                        f'品类:<b>{_b.fruit_category or "—"}</b>  ·  '
                        f'仓储:<b>{_b.warehouse or "—"} / {_b.storage_zone or "—"}</b>'
                        f'</div>'
                        f'</div></div>'
                    )

                    # ── 节点 2: 检测 ──
                    if _b.detection_rounds > 0:
                        _node_html.append(
                            f'<div style="display:flex;gap:12px;padding:0.6rem 0;'
                            f'border-left:3px solid #7C3AED;padding-left:14px;margin-left:6px">'
                            f'<div>'
                            f'<div style="font-weight:700;color:#3D2B1F;font-size:0.88rem">'
                            f'🔍 模型检测'
                            f'</div>'
                            f'<div style="color:#A07040;font-size:0.78rem;margin-top:2px">'
                            f'检测轮数:<b>{_b.detection_rounds}</b>  ·  '
                            f'累计目标:<b>{_b.detected_total}</b>  ·  '
                            f'已处理文件:<b>{len(_b.processed_files)}</b> 个'
                            f'</div>'
                            f'<div style="color:#6B4F2F;font-size:0.76rem;margin-top:3px">'
                            f'分级 — A: <b style="color:#22C55E">{_b.grade_a_count}</b>  ·  '
                            f'B: <b style="color:#F59E0B">{_b.grade_b_count}</b>  ·  '
                            f'C: <b style="color:#EF4444">{_b.grade_c_count}</b>  ·  '
                            f'平均置信度:<b>{_b.avg_confidence:.3f}</b>'
                            f'</div>'
                            f'<div style="color:#A07040;font-size:0.76rem;margin-top:3px;font-style:italic">'
                            f'判定原因:{_b.status_reason or "—"}'
                            f'</div>'
                            f'</div></div>'
                        )

                    # ── 节点 3: 复核(reviewed_at 非空)──
                    if _b.reviewed_at:
                        _rv_time = _b.reviewed_at.strftime("%Y-%m-%d %H:%M:%S")
                        _snap = _b.review_snapshot or {}
                        _snap_action = _snap.get("action", "—")
                        _snap_color = "#16A34A" if _snap_action == "approved" else "#EF4444"
                        _snap_label = "通过" if _snap_action == "approved" else "拒收"
                        _node_html.append(
                            f'<div style="display:flex;gap:12px;padding:0.6rem 0;'
                            f'border-left:3px solid #F59E0B;padding-left:14px;margin-left:6px">'
                            f'<div style="flex:1">'
                            f'<div style="font-weight:700;color:#3D2B1F;font-size:0.88rem">'
                            f'⚖️ 经理复核 '
                            f'<span style="background:{_snap_color};color:white;'
                            f'padding:1px 8px;border-radius:50px;font-size:0.7rem;'
                            f'font-weight:700;margin-left:4px">{_snap_label}</span>'
                            f'</div>'
                            f'<div style="color:#A07040;font-size:0.78rem;margin-top:2px">'
                            f'{_rv_time}  ·  操作人 <b>{_b.reviewed_by_name or f"#{_b.reviewed_by}"}</b>'
                            f'</div>'
                            + (f'<div style="color:#6B4F2F;font-size:0.76rem;margin-top:3px">'
                               f'📝 复核备注:{_b.reviewed_note}'
                               f'</div>' if _b.reviewed_note else '')
                            + (f'<div style="background:#FFFBF0;border:1px solid #F0EAD6;'
                               f'border-radius:8px;padding:0.4rem 0.7rem;margin-top:5px;'
                               f'color:#6B4F2F;font-size:0.75rem">'
                               f'<b>当时数据快照:</b>  '
                               f'申报<b>{_snap.get("declared_total","—")}</b> / '
                               f'实际<b>{_snap.get("detected_total","—")}</b>  ·  '
                               f'差异率<b>{_snap.get("count_diff_pct",0)*100:.1f}%</b>  ·  '
                               f'次品率<b>{_snap.get("defect_rate",0)*100:.1f}%</b>  ·  '
                               f'已检测<b>{_snap.get("detection_rounds","—")}</b> 轮'
                               f'</div>' if _snap else '')
                            + f'</div></div>'
                        )

                    # ── 节点 4: 入库确认 ──
                    if _b.confirmed_at:
                        _cf_time = _b.confirmed_at.strftime("%Y-%m-%d %H:%M:%S")
                        _node_html.append(
                            f'<div style="display:flex;gap:12px;padding:0.6rem 0;'
                            f'border-left:3px solid #16A34A;padding-left:14px;margin-left:6px">'
                            f'<div>'
                            f'<div style="font-weight:700;color:#16A34A;font-size:0.88rem">'
                            f'✅ 已入库'
                            f'</div>'
                            f'<div style="color:#A07040;font-size:0.78rem;margin-top:2px">'
                            f'{_cf_time}  ·  操作人 <b>{_b.confirmed_by_name or f"#{_b.confirmed_by}"}</b>'
                            f'</div>'
                            f'</div></div>'
                        )

                    # ── 节点 5: 拒收 ──
                    if _b.rejected_at:
                        _rj_time = _b.rejected_at.strftime("%Y-%m-%d %H:%M:%S")
                        _node_html.append(
                            f'<div style="display:flex;gap:12px;padding:0.6rem 0;'
                            f'border-left:3px solid #EF4444;padding-left:14px;margin-left:6px">'
                            f'<div>'
                            f'<div style="font-weight:700;color:#EF4444;font-size:0.88rem">'
                            f'❌ 已拒收'
                            f'</div>'
                            f'<div style="color:#A07040;font-size:0.78rem;margin-top:2px">'
                            f'{_rj_time}  ·  操作人 <b>{_b.rejected_by_name or f"#{_b.rejected_by}"}</b>'
                            f'</div>'
                            + (f'<div style="color:#6B4F2F;font-size:0.76rem;margin-top:3px">'
                               f'📝 拒收原因:{_b.rejection_reason}'
                               f'</div>' if _b.rejection_reason else '')
                            + (f'<div style="background:#FFFBF0;border:1px solid #F0EAD6;'
                               f'border-radius:8px;padding:0.4rem 0.7rem;margin-top:5px;'
                               f'color:#6B4F2F;font-size:0.75rem">'
                               f'<b>当时数据快照:</b>  '
                               f'申报<b>{(_b.review_snapshot or {}).get("declared_total","—")}</b> / '
                               f'实际<b>{(_b.review_snapshot or {}).get("detected_total","—")}</b>  ·  '
                               f'差异率<b>{(_b.review_snapshot or {}).get("count_diff_pct",0)*100:.1f}%</b>  ·  '
                               f'次品率<b>{(_b.review_snapshot or {}).get("defect_rate",0)*100:.1f}%</b>  ·  '
                               f'已检测<b>{(_b.review_snapshot or {}).get("detection_rounds","—")}</b> 轮'
                               f'</div>' if _b.review_snapshot else '')
                            + f'</div></div>'
                        )

                    st.markdown(
                        '<div style="background:#FFFBF0;border-radius:10px;'
                        'padding:0.5rem 0.8rem;margin-top:0.3rem">'
                        + "".join(_node_html) +
                        '</div>',
                        unsafe_allow_html=True
                    )


# ══════════════════════════════════════════════
# Step 7.5: 供应商档案 Tab(admin only)
# ══════════════════════════════════════════════
if "supplier" in T:
    with T["supplier"]:
        from pages.supplier_tab import render_supplier_tab
        render_supplier_tab()
