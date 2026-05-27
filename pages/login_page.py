"""
pages/login_page.py — 登录页面

未登录时由 app.py 主入口调用,登录后自动跳转主界面。
"""
import streamlit as st
from auth import authenticate, get_demo_accounts


def render_login_page():
    """渲染登录页面"""
    st.markdown("""
    <style>
    .login-hero {
        background: linear-gradient(135deg, #FFF8ED 0%, #FFE8C2 100%);
        border-radius: 24px; padding: 2rem 2.5rem;
        margin-bottom: 1.5rem; border: 1.5px solid #F5D99A;
    }
    .login-title {
        font-size: 1.6rem; font-weight: 900; color: #C05C00;
        margin: 0 0 0.3rem;
    }
    .login-sub { font-size: 0.9rem; color: #A07040; margin: 0; }
    .demo-box {
        background: #FFFBF0; border-radius: 12px;
        padding: 0.8rem 1rem; border: 1px dashed #F5D99A;
        font-size: 0.82rem; color: #6B4F2F;
    }
    .demo-box code {
        background: #FFE8C2; padding: 1px 6px; border-radius: 4px;
        color: #C05C00; font-weight: 700;
    }
    </style>

    <div class="login-hero">
        <div class="login-title">🍎 水果入库盘点与品质分级系统</div>
        <p class="login-sub">基于轻量化 YOLOv11 · 入库筛查与对账复核</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("### 🔐 用户登录")

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("用户名", placeholder="例如:operator")
            password = st.text_input("密码", type="password",
                                     placeholder="演示账号请见右侧提示")
            submitted = st.form_submit_button("登 录", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("请输入用户名和密码")
            else:
                user = authenticate(username, password)
                if user is None:
                    st.error("❌ 用户名或密码错误")
                else:
                    st.session_state.current_user = user
                    st.success(f"✅ 欢迎,{user.full_name}!")
                    st.rerun()

    with col_r:
        st.markdown("### 👥 演示账号")
        st.markdown(
            '<div class="demo-box">'
            '本系统为毕业设计演示版,提供以下三个角色账号:'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("")

        for acc in get_demo_accounts():
            st.markdown(
                f'<div class="demo-box" style="margin-bottom:0.5rem">'
                f'<b>{acc["role"]}</b><br>'
                f'用户名:<code>{acc["username"]}</code> &nbsp; '
                f'密码:<code>{acc["password"]}</code>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.caption(
            "💡 不同角色拥有不同权限:操作员可创建批次和快速入库,"
            "经理可复核,管理员可维护主数据。"
        )