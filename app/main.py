import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.database import engine, Base, SessionLocal
import app.models.models  # Base에 테이블 등록
from app.routers import sessions, branches, graph, tags, files, comparison, trash, personas

# uploaded_files 스키마 변경(session_id 추가) 대응: 구버전 테이블이면 삭제 후 재생성
_pre_inspector = inspect(engine)
if "uploaded_files" in _pre_inspector.get_table_names():
    _file_cols = {c["name"] for c in _pre_inspector.get_columns("uploaded_files")}
    if "session_id" not in _file_cols:
        with engine.connect() as _conn:
            _conn.execute(text("DROP TABLE uploaded_files"))
            _conn.commit()

Base.metadata.create_all(bind=engine)

# create_all은 기존 테이블에 새 컬럼을 추가해주지 않으므로, 누락된 컬럼은 직접 보강한다.
_BOOL_FALSE = "FALSE" if engine.dialect.name == "postgresql" else "0"
_BOOL_TRUE = "TRUE" if engine.dialect.name == "postgresql" else "1"
_TEXT_TIMESTAMP_DEFAULT = "'1970-01-01T00:00:00'"


def _add_missing_columns(table_name: str, columns: dict[str, str]) -> None:
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return

    existing_columns = {c["name"] for c in inspector.get_columns(table_name)}
    missing_columns = [
        column_sql
        for column_name, column_sql in columns.items()
        if column_name not in existing_columns
    ]
    if not missing_columns:
        return

    with engine.begin() as conn:
        for column_sql in missing_columns:
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}"))


_add_missing_columns(
    "conversations",
    {
        "memory": "memory TEXT",
        "status": "status TEXT NOT NULL DEFAULT 'active'",
        "deleted_at": "deleted_at TEXT",
        "created_at": f"created_at TEXT NOT NULL DEFAULT {_TEXT_TIMESTAMP_DEFAULT}",
        "updated_at": f"updated_at TEXT NOT NULL DEFAULT {_TEXT_TIMESTAMP_DEFAULT}",
    },
)

_add_missing_columns(
    "branches",
    {
        "parent_branch_id": "parent_branch_id TEXT",
        "fork_from_message_id": "fork_from_message_id INTEGER",
        "name": "name TEXT NOT NULL DEFAULT 'main'",
        "head_id": "head_id INTEGER",
        "status": "status TEXT NOT NULL DEFAULT 'active'",
        "deleted_at": "deleted_at TEXT",
        "summary": "summary TEXT",
        "description": "description TEXT",
        "live_summary": "live_summary TEXT",
        "live_summary_cutoff_id": "live_summary_cutoff_id INTEGER",
        "persona_id": "persona_id TEXT",
        "is_collapsed": f"is_collapsed BOOLEAN NOT NULL DEFAULT {_BOOL_FALSE}",
        "is_merge": f"is_merge BOOLEAN NOT NULL DEFAULT {_BOOL_FALSE}",
        "is_main": f"is_main BOOLEAN NOT NULL DEFAULT {_BOOL_FALSE}",
        "created_at": f"created_at TEXT NOT NULL DEFAULT {_TEXT_TIMESTAMP_DEFAULT}",
        "updated_at": f"updated_at TEXT NOT NULL DEFAULT {_TEXT_TIMESTAMP_DEFAULT}",
    },
)

_add_missing_columns(
    "messages",
    {
        "parent_id": "parent_id INTEGER",
        "model_provider": "model_provider TEXT",
        "model_name": "model_name TEXT",
        "input_tokens": "input_tokens INTEGER",
        "output_tokens": "output_tokens INTEGER",
        "status": "status TEXT NOT NULL DEFAULT 'active'",
        "created_at": f"created_at TEXT NOT NULL DEFAULT {_TEXT_TIMESTAMP_DEFAULT}",
    },
)

_add_missing_columns(
    "tags",
    {
        "session_id": "session_id TEXT",
        "color": "color TEXT",
        "type": "type TEXT NOT NULL DEFAULT 'normal'",
        "created_at": f"created_at TEXT NOT NULL DEFAULT {_TEXT_TIMESTAMP_DEFAULT}",
    },
)

_add_missing_columns(
    "message_tags",
    {
        "created_at": f"created_at TEXT NOT NULL DEFAULT {_TEXT_TIMESTAMP_DEFAULT}",
    },
)

_add_missing_columns(
    "branch_tags",
    {
        "created_at": f"created_at TEXT NOT NULL DEFAULT {_TEXT_TIMESTAMP_DEFAULT}",
    },
)

_add_missing_columns(
    "branch_merge_parents",
    {
        "summary": "summary TEXT NOT NULL DEFAULT ''",
        "created_at": f"created_at TEXT NOT NULL DEFAULT {_TEXT_TIMESTAMP_DEFAULT}",
    },
)

_add_missing_columns(
    "uploaded_files",
    {
        "branch_id": "branch_id TEXT",
        "message_id": "message_id INTEGER",
        "summary": "summary TEXT",
        "file_type": "file_type TEXT NOT NULL DEFAULT 'text'",
        "mime_type": "mime_type TEXT",
        "content_data": "content_data TEXT",
        "image_data": "image_data TEXT",
        "created_at": f"created_at TEXT NOT NULL DEFAULT {_TEXT_TIMESTAMP_DEFAULT}",
    },
)

_add_missing_columns(
    "embeddings",
    {
        "created_at": f"created_at TEXT NOT NULL DEFAULT {_TEXT_TIMESTAMP_DEFAULT}",
    },
)

_add_missing_columns(
    "comparison_sessions",
    {
        "status": "status TEXT NOT NULL DEFAULT 'pending'",
        "created_at": f"created_at TEXT NOT NULL DEFAULT {_TEXT_TIMESTAMP_DEFAULT}",
    },
)

PURGE_CHECK_INTERVAL_SECONDS = 3600


async def _purge_expired_trash_loop():
    while True:
        try:
            from app.services import trash_cleanup
            trash_cleanup.purge_expired()
        except Exception:
            pass
        await asyncio.sleep(PURGE_CHECK_INTERVAL_SECONDS)


def _repair_legacy_nulls() -> None:
    """기존 배포 DB에서 nullable로 남은 상태/boolean 값을 현재 조회 조건에 맞게 보정한다."""
    with engine.begin() as conn:
        conn.execute(text("UPDATE conversations SET status = 'active' WHERE status IS NULL OR status = ''"))
        conn.execute(text("UPDATE messages SET status = 'active' WHERE status IS NULL OR status = ''"))
        conn.execute(text("UPDATE branches SET status = 'active' WHERE status IS NULL OR status = ''"))
        conn.execute(text(f"UPDATE branches SET is_collapsed = {_BOOL_FALSE} WHERE is_collapsed IS NULL"))
        conn.execute(text(f"UPDATE branches SET is_merge = {_BOOL_FALSE} WHERE is_merge IS NULL"))
        conn.execute(text(f"UPDATE branches SET is_main = {_BOOL_FALSE} WHERE is_main IS NULL"))
        conn.execute(text(f"""
            UPDATE branches
            SET is_main = {_BOOL_TRUE}
            WHERE parent_branch_id IS NULL
              AND is_merge IS NOT {_BOOL_TRUE}
              AND NOT EXISTS (
                  SELECT 1
                  FROM branches sibling
                  WHERE sibling.session_id = branches.session_id
                    AND sibling.is_main IS {_BOOL_TRUE}
              )
        """))


_repair_legacy_nulls()

_seed_db = SessionLocal()
try:
    from app.repositories import repository as _repository
    from app.services.persona_seeds import PERSONAS as _PERSONAS
    _repository.seed_personas(_seed_db, _PERSONAS)
finally:
    _seed_db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_purge_expired_trash_loop())
    yield
    task.cancel()


app = FastAPI(title="LLM 채팅 브랜치 시각화 서비스", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://ramo-9uburl0kv-3ramo.vercel.app",
        "https://ramo-pi.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="서버 상태 확인", tags=["Health"])
def health():
    """서버가 정상적으로 실행 중인지 확인합니다.

    - 정상이면 `{"status": "ok"}`를 반환합니다.
    - 배포 환경에서 서버가 살아있는지 주기적으로 체크할 때 사용합니다.
    """
    return {"status": "ok"}


app.include_router(sessions.router)
app.include_router(branches.router)
app.include_router(graph.router)
app.include_router(tags.router)
app.include_router(files.router)
app.include_router(comparison.router)
app.include_router(trash.router)
app.include_router(personas.router)
