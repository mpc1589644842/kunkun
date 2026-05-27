"""
db/session.py — SQLAlchemy 引擎与会话工厂(Step 9.2.1)
================================================
- 单文件 SQLite 数据库 fruit_app.db(项目根目录)
- echo=False(生产模式);开发调试时可临时改 True 看 SQL
- check_same_thread=False:允许 Streamlit 多线程访问(Streamlit 每次 rerun 是同一线程,
  但仍然显式声明以避免未来踩坑)
"""
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from db.orm_models import Base

# 数据库文件位置:项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _PROJECT_ROOT / "fruit_app.db"
DB_URL  = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DB_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,   # 重要:让对象在 commit 后仍可访问字段(供 UI 层用)
    future=True,
)


def init_db() -> None:
    """建表(幂等:已存在的表跳过)"""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """获取一个新 session(调用方负责 close)"""
    return SessionLocal()
