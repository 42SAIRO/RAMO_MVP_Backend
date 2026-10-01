from datetime import datetime, timedelta, timezone

from app.database import SessionLocal
from app.models.models import Conversation, Branch
from app.repositories import repository

RETENTION_DAYS = 7


def _is_expired(deleted_at: str | None) -> bool:
    """deleted_at을 파싱해 RETENTION_DAYS일이 지났는지 판단한다.

    과거 데이터에 'YYYY-MM-DD HH:MM:SS'와 'YYYY-MM-DDTHH:MM:SS' 형식이 섞여 있을 수 있어
    문자열 비교 대신 datetime으로 파싱해서 비교한다 (구분자가 다르면 문자열 비교 순서가
    실제 시간 순서와 어긋날 수 있다).
    """
    if not deleted_at:
        return False
    try:
        deleted_time = datetime.fromisoformat(deleted_at.replace("T", " ")).replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return datetime.now(timezone.utc) - deleted_time > timedelta(days=RETENTION_DAYS)


def purge_expired():
    """휴지통에 RETENTION_DAYS일 넘게 있던 세션/브랜치를 완전히 삭제한다."""
    db = SessionLocal()
    try:
        expired_sessions = [
            c for c in db.query(Conversation).filter(Conversation.status == "deleted").all()
            if _is_expired(c.deleted_at)
        ]
        for conv in expired_sessions:
            repository.purge_session(db, conv.id)

        expired_branches = [
            b for b in db.query(Branch).filter(Branch.status == "deleted").all()
            if _is_expired(b.deleted_at)
        ]
        for branch in expired_branches:
            repository.purge_branch_cascade(db, branch.id)
    finally:
        db.close()
