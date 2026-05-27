"""
供应商档案 Tab(admin only)— Step 7.5 + 7.6 + 7.7
================================================
路由(session_state.selected_supplier_key):
    None        → 列表视图(7.5)
    "<key>"     → 详情视图(7.6) + 编辑 popover(7.7)

不做新建(operator 建批次时自动建档,手动新建冗余)。
"""
from __future__ import annotations
import streamlit as st

from services import supplier_service as svc
from db.repositories import supplier_repo as sup_repo  # Step 9.2.2.b


# ─────── 样式 ───────
_STYLE = """
<style>
.sup-card {
    background: #FFFAF3;
    border: 1px solid #F2D5A8;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(192, 92, 0, 0.05);
}
.sup-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}
.sup-name {
    font-size: 1.05rem;
    font-weight: 800;
    color: #7A3A00;
}
.sup-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #FFF;
    margin-left: 8px;
}
.sup-meta { color: #A07040; font-size: 0.85rem; }
.sup-kpi-row {
    display: flex; gap: 16px; margin-top: 6px; flex-wrap: wrap;
}
.sup-kpi-item { font-size: 0.88rem; }
.sup-kpi-label { color: #A07040; }
.sup-kpi-value { font-weight: 800; color: #7A3A00; margin-left: 4px; }
.sup-empty {
    text-align: center; color: #A07040;
    padding: 40px 20px; background: #FFFAF3;
    border-radius: 12px; border: 1px dashed #F2D5A8;
}
.sup-detail-card {
    background: #FFFAF3;
    border: 1px solid #F2D5A8;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.sup-detail-label {
    color: #A07040; font-size: 0.85rem; margin-right: 6px;
}
.sup-detail-value {
    color: #7A3A00; font-weight: 700;
}
.sup-section-title {
    font-size: 1.0rem; font-weight: 800; color: #C05C00;
    margin: 14px 0 8px 0;
}
</style>
"""


# ═══════════════════════════════════════════════════════════
# 主入口:根据 session_state.selected_supplier_key 路由
# ═══════════════════════════════════════════════════════════
def render_supplier_tab():
    st.markdown(_STYLE, unsafe_allow_html=True)

    # Step 9.2.2.b: 路由 key 改为 supplier_id (int),比 normalized_name 更稳定
    # (重命名后不会失效)。命名仍然保留 selected_supplier_key 不变。
    if "selected_supplier_key" not in st.session_state:
        st.session_state.selected_supplier_key = None

    sel_id = st.session_state.selected_supplier_key

    if sel_id is not None:
        sup = sup_repo.find_by_id(sel_id)
        if sup is None:
            # 路由失效(被彻底删了或重置),回到列表
            st.session_state.selected_supplier_key = None
            _render_supplier_list()
        else:
            _render_supplier_detail(sup)
    else:
        _render_supplier_list()


# ═══════════════════════════════════════════════════════════
# 列表视图(Step 7.5)
# ═══════════════════════════════════════════════════════════
def _render_supplier_list():
    st.markdown(
        '<div style="font-size:1.1rem;font-weight:900;color:#C05C00;margin-bottom:0.2rem">'
        '📇 供应商档案</div>',
        unsafe_allow_html=True,
    )
    st.caption("展示所有供应商的关键指标。点击卡片查看详情或编辑档案。")

    # Step 9.2.2.b: 直接从 repository 读
    all_active = sup_repo.list_all(include_inactive=False)
    all_batches = st.session_state.all_batches or []

    if not all_active:
        st.markdown(
            '<div class="sup-empty">还没有任何供应商档案。<br>'
            '当 operator 创建批次并填写供应商名称时,系统会自动建档。</div>',
            unsafe_allow_html=True,
        )
        return

    # 预聚合
    sup_records = []
    for sup in all_active:
        kpi = svc.aggregate_kpi(sup, all_batches)
        label = svc.classify_supplier(kpi)
        main_cats = svc.aggregate_main_categories(sup, all_batches)
        sup_records.append({
            "key": sup.supplier_id,  # 路由 key 改为 supplier_id
            "supplier": sup, "kpi": kpi,
            "label": label, "main_cats": main_cats,
        })

    # 顶部 KPI
    label_counts = {"premium": 0, "warning": 0, "normal": 0, "new": 0}
    for rec in sup_records:
        label_counts[rec["label"]] = label_counts.get(rec["label"], 0) + 1

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("供应商总数", len(sup_records))
    k2.metric("🏆 优质", label_counts["premium"])
    k3.metric("⚠️ 警示", label_counts["warning"])
    k4.metric("普通", label_counts["normal"])
    k5.metric("⚪ 新供应商", label_counts["new"])

    st.markdown("---")

    # 筛选/排序
    f1, f2 = st.columns([2, 1])
    with f1:
        label_filter = st.multiselect(
            "按标签筛选",
            options=["premium", "warning", "normal", "new"],
            default=[],
            format_func=lambda x: {
                "premium": "🏆 优质", "warning": "⚠️ 警示",
                "normal": "普通", "new": "⚪ 新供应商",
            }[x],
            key="sup_label_filter",
        )
    with f2:
        sort_by = st.selectbox(
            "排序",
            options=["batches_desc", "qualified_desc", "name_asc"],
            format_func=lambda x: {
                "batches_desc": "按总批次 ↓",
                "qualified_desc": "按合格率 ↓",
                "name_asc": "按名称 A→Z",
            }[x],
            key="sup_sort_by",
        )

    filtered = sup_records
    if label_filter:
        filtered = [r for r in filtered if r["label"] in label_filter]

    if sort_by == "batches_desc":
        filtered.sort(key=lambda r: r["kpi"]["total_batches"], reverse=True)
    elif sort_by == "qualified_desc":
        filtered.sort(key=lambda r: r["kpi"]["avg_qualified"], reverse=True)
    else:
        filtered.sort(key=lambda r: r["supplier"].name)

    if not filtered:
        st.info("没有符合筛选条件的供应商。")
        return

    st.markdown(f"**展示 {len(filtered)} 个供应商**")
    for rec in filtered:
        _render_supplier_card(rec)


def _render_supplier_card(rec: dict):
    sup = rec["supplier"]
    kpi = rec["kpi"]
    label = rec["label"]
    main_cats = rec["main_cats"]

    badge_text, badge_color, badge_emoji = svc.label_display(label)
    badge_html = ""
    if label != "normal":
        badge_html = (
            f'<span class="sup-badge" style="background:{badge_color}">'
            f'{badge_emoji} {badge_text}</span>'
        )

    cats_str = " / ".join(main_cats) if main_cats else "—"
    region_str = sup.region or "—"

    html = f"""
    <div class="sup-card">
        <div class="sup-card-header">
            <div>
                <span class="sup-name">🏭 {sup.name}</span>
                {badge_html}
            </div>
            <span class="sup-meta">ID #{sup.supplier_id}  ·  {region_str}</span>
        </div>
        <div class="sup-kpi-row">
            <span class="sup-kpi-item"><span class="sup-kpi-label">总批次</span><span class="sup-kpi-value">{kpi['total_batches']}</span></span>
            <span class="sup-kpi-item"><span class="sup-kpi-label">已入库</span><span class="sup-kpi-value">{kpi['confirmed']}</span></span>
            <span class="sup-kpi-item"><span class="sup-kpi-label">入库率</span><span class="sup-kpi-value">{kpi['confirm_rate']*100:.1f}%</span></span>
            <span class="sup-kpi-item"><span class="sup-kpi-label">合格率</span><span class="sup-kpi-value">{kpi['avg_qualified']*100:.1f}%</span></span>
            <span class="sup-kpi-item"><span class="sup-kpi-label">次品率</span><span class="sup-kpi-value">{kpi['avg_defect']*100:.1f}%</span></span>
            <span class="sup-kpi-item"><span class="sup-kpi-label">主营品类</span><span class="sup-kpi-value">{cats_str}</span></span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # 按钮放在 markdown 卡片外部(streamlit 限制),用 columns 让按钮靠左
    bc1, bc2, _ = st.columns([1, 1, 6])
    with bc1:
        if st.button("📂 查看详情", key=f"sup_detail_{rec['key']}", use_container_width=True):
            st.session_state.selected_supplier_key = rec["key"]
            st.rerun()


# ═══════════════════════════════════════════════════════════
# 详情视图(Step 7.6)+ 编辑 popover(Step 7.7)
# ═══════════════════════════════════════════════════════════
def _render_supplier_detail(sup):
    """Step 9.2.2.b: 直接接收 dataclass sup,不再从 session_state.suppliers 取"""
    sel_key = sup.supplier_id   # 后续 popover key / confirm flag 等仍可用
    all_batches = st.session_state.all_batches or []

    # ─── Step 7.8.1: 顶部改 4 列,软删除按钮从 popover 移出来 ───
    top_l, top_m, top_e, top_d = st.columns([1, 4, 1, 1])
    with top_l:
        if st.button("← 返回", key="sup_back", use_container_width=True):
            st.session_state.selected_supplier_key = None
            st.rerun()
    with top_m:
        st.markdown(
            f'<div style="font-size:1.1rem;font-weight:900;color:#C05C00;margin-top:4px">'
            f'🏭 {sup.name} <span style="font-size:0.85rem;color:#A07040">ID #{sup.supplier_id}</span></div>',
            unsafe_allow_html=True,
        )
    with top_e:
        with st.popover("✏️ 编辑", use_container_width=True):
            _render_edit_form(sel_key, sup)
    with top_d:
        _render_delete_button(sel_key, sup)

    # ─── KPI 聚合(详情页用完整 11 个 key)───
    kpi = svc.aggregate_kpi(sup, all_batches)
    label = svc.classify_supplier(kpi)
    main_cats = svc.aggregate_main_categories(sup, all_batches)
    badge_text, badge_color, badge_emoji = svc.label_display(label)

    # ─── 基础信息卡片 ───
    region_str = sup.region or "—"
    notes_str = sup.notes or "—"
    phone_str = sup.contact_phone or "—"
    addr_str = sup.address or "—"
    created_str = sup.created_at.strftime("%Y-%m-%d %H:%M") if sup.created_at else "—"
    updated_str = sup.updated_at.strftime("%Y-%m-%d %H:%M") if sup.updated_at else "—"
    cats_str = " / ".join(main_cats) if main_cats else "—"
    badge_html = (
        f'<span class="sup-badge" style="background:{badge_color}">'
        f'{badge_emoji} {badge_text}</span>'
    ) if label != "normal" else f'<span class="sup-detail-value">普通</span>'

    info_html = f"""
    <div class="sup-detail-card">
        <div style="margin-bottom:8px"><span class="sup-detail-label">标签</span>{badge_html}</div>
        <div style="margin-bottom:6px"><span class="sup-detail-label">地区</span><span class="sup-detail-value">{region_str}</span></div>
        <div style="margin-bottom:6px"><span class="sup-detail-label">联系电话</span><span class="sup-detail-value">{phone_str}</span></div>
        <div style="margin-bottom:6px"><span class="sup-detail-label">地址</span><span class="sup-detail-value">{addr_str}</span></div>
        <div style="margin-bottom:6px"><span class="sup-detail-label">主营品类</span><span class="sup-detail-value">{cats_str}</span></div>
        <div style="margin-bottom:6px"><span class="sup-detail-label">备注</span><span class="sup-detail-value">{notes_str}</span></div>
        <div style="margin-bottom:6px"><span class="sup-detail-label">建档时间</span><span class="sup-detail-value">{created_str}</span></div>
        <div><span class="sup-detail-label">最近更新</span><span class="sup-detail-value">{updated_str}</span></div>
    </div>
    """
    st.markdown('<div class="sup-section-title">📋 基础信息</div>', unsafe_allow_html=True)
    st.markdown(info_html, unsafe_allow_html=True)

    # ─── KPI 横排(metric)───
    st.markdown('<div class="sup-section-title">📊 关键指标</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("总批次", kpi["total_batches"])
    m2.metric("已入库", kpi["confirmed"])
    m3.metric("已拒收", kpi["rejected"])
    m4.metric("待处理", kpi["pending"])

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("入库率", f"{kpi['confirm_rate']*100:.1f}%")
    m6.metric("拒收率", f"{kpi['reject_rate']*100:.1f}%")
    m7.metric("平均合格率", f"{kpi['avg_qualified']*100:.1f}%")
    m8.metric("平均次品率", f"{kpi['avg_defect']*100:.1f}%")

    last_str = "—"
    if kpi["last_delivery"]:
        last_str = kpi["last_delivery"].strftime("%Y-%m-%d %H:%M")
    st.caption(f"最近一次到货:{last_str}  ·  对应批次:{kpi['last_batch_id'] or '—'}")

    # ─── 历史批次表格 ───
    st.markdown('<div class="sup-section-title">📜 历史批次</div>', unsafe_allow_html=True)
    batches = svc.get_supplier_batches(sup, all_batches)
    if not batches:
        st.info("该供应商暂无批次记录。")
    else:
        # 按入库日期倒序
        batches_sorted = sorted(batches, key=lambda b: b.inbound_date or 0, reverse=True)
        rows = []
        for b in batches_sorted:
            rows.append({
                "批次号":   b.batch_id,
                "品类":     b.fruit_category or "—",
                "申报":     b.declared_count,
                "检测":     b.detected_total,
                "A":        b.grade_a_count,
                "B":        b.grade_b_count,
                "C":        b.grade_c_count,
                "差异率":   f"{b.count_diff_pct*100:.1f}%",
                "次品率":   f"{b.defect_rate*100:.1f}%",
                "状态":     b.status.value if hasattr(b.status, "value") else str(b.status),
                "入库日期": b.inbound_date.strftime("%Y-%m-%d") if b.inbound_date else "—",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# 编辑 popover 表单(Step 7.7)
# ═══════════════════════════════════════════════════════════
def _render_edit_form(sel_key: str, sup):
    st.markdown("**编辑供应商档案**")
    st.caption("只允许修改地区、联系电话、备注。名称由系统通过批次同步,不可在此修改。")

    new_region = st.text_input("地区", value=sup.region, key=f"edit_region_{sel_key}")
    new_phone  = st.text_input("联系电话", value=sup.contact_phone, key=f"edit_phone_{sel_key}")
    new_addr   = st.text_input("地址", value=sup.address, key=f"edit_addr_{sel_key}")
    new_notes  = st.text_area("备注", value=sup.notes, key=f"edit_notes_{sel_key}", height=80)

    # Step 7.8.1: 软删除按钮已搬到详情页顶部,这里只保留保存
    if st.button("💾 保存修改", key=f"edit_save_{sel_key}", type="primary", use_container_width=True):
        svc.update_supplier(
            sup,
            region=new_region.strip(),
            contact_phone=new_phone.strip(),
            address=new_addr.strip(),
            notes=new_notes.strip(),
        )
        st.success("已保存")
        st.rerun()

# ═══════════════════════════════════════════════════════════
# Step 7.8.1: 软删除按钮 + 二次确认(放在 popover 外,避开生命周期问题)
# ═══════════════════════════════════════════════════════════
def _render_delete_button(sel_key: str, sup):
    """两段式删除:第一次点击展开确认,第二次才真删。"""
    confirm_key = f"_confirm_delete_{sel_key}"
    if confirm_key not in st.session_state:
        st.session_state[confirm_key] = False

    if not st.session_state[confirm_key]:
        # 第一次:展示软删除入口
        if st.button("🗑️ 软删除", key=f"del_init_{sel_key}", use_container_width=True):
            st.session_state[confirm_key] = True
            st.rerun()
    else:
        # 第二次:确认/取消
        if st.button("✅ 确认删除", key=f"del_confirm_{sel_key}",
                     type="primary", use_container_width=True):
            svc.soft_delete_supplier(sup)
            st.session_state.selected_supplier_key = None
            st.session_state[confirm_key] = False
            st.toast(f"已软删除 {sup.name}", icon="🗑️")
            st.rerun()
        if st.button("取消", key=f"del_cancel_{sel_key}", use_container_width=True):
            st.session_state[confirm_key] = False
            st.rerun()

