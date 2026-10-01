from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import ConversationOut
from app.repositories import repository

router = APIRouter(tags=["Trash"])


@router.get("/trash", response_model=list[ConversationOut], summary="세션 휴지통 목록 조회")
def list_trash(db: Session = Depends(get_db)):
    """휴지통으로 이동된 세션 목록을 반환합니다. 브랜치 휴지통은 포함되지 않습니다.

    - 브랜치 단위 휴지통은 `GET /sessions/{session_id}/branch-trash`를 사용하세요.
    """
    return repository.list_deleted_sessions(db)


@router.post("/trash/{session_id}/restore", response_model=ConversationOut, summary="세션 복원")
def restore_session(session_id: str, db: Session = Depends(get_db)):
    """휴지통의 세션을 원래 상태로 복원합니다. 브랜치 상태에는 영향을 주지 않습니다."""
    conv = repository.restore_session(db, session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="휴지통에서 session을 찾을 수 없습니다")
    return conv


@router.delete("/trash/{session_id}", summary="세션 영구 삭제")
def purge_session(session_id: str, db: Session = Depends(get_db)):
    """세션과 그 안의 모든 브랜치/메시지/태그를 완전히 삭제합니다. 복원할 수 없습니다."""
    purged = repository.purge_session(db, session_id)
    if not purged:
        raise HTTPException(status_code=404, detail="session을 찾을 수 없습니다")
    return {"session_id": session_id, "purged": True}
