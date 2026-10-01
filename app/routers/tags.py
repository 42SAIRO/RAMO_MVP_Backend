from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import CreateTagRequest, AddTagRequest, TagOut
from app.repositories import repository

router = APIRouter(tags=["Tags"])


@router.post("/tags", response_model=TagOut, summary="태그 생성")
def create_tag(req: CreateTagRequest, db: Session = Depends(get_db)):
    """새 태그를 수동으로 생성합니다.

    - `type`: normal(일반) / highlight(중요 표시)
    - `color`: 원하는 색상 코드 (예: "#FACC15"). 없으면 null
    - 태그는 session 단위로 관리됩니다.
    """
    return repository.create_tag(db, req.session_id, req.name, req.color, req.type)


@router.get("/sessions/{session_id}/tags", response_model=list[TagOut], summary="세션의 태그 목록 조회")
def session_tags(session_id: str, db: Session = Depends(get_db)):
    """특정 세션에 속한 태그 목록을 반환합니다."""
    return repository.get_session_tags(db, session_id)


@router.get("/branches/{branch_id}/tags", response_model=list[TagOut], summary="브랜치 태그 목록 조회")
def branch_tags(branch_id: str, db: Session = Depends(get_db)):
    """특정 브랜치에 연결된 태그 목록을 반환합니다.

    - 태그가 없는 브랜치는 빈 배열을 반환합니다.
    - 존재하지 않는 브랜치는 404를 반환합니다.
    """
    if repository.get_branch(db, branch_id) is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    return repository.get_branch_tags(db, branch_id)


@router.post("/branches/{branch_id}/tags", status_code=201, summary="브랜치에 태그 수동 부여")
def add_branch_tag(branch_id: str, req: AddTagRequest, db: Session = Depends(get_db)):
    """특정 브랜치에 태그를 수동으로 추가합니다. 이미 달려있으면 무시합니다.

    자동 태그는 `POST /branches/{branch_id}/auto-tag`를 사용하세요.
    """
    if repository.get_branch(db, branch_id) is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    if repository.get_tag(db, req.tag_id) is None:
        raise HTTPException(status_code=404, detail="tag를 찾을 수 없습니다")
    repository.add_branch_tag(db, branch_id, req.tag_id)
    return {"branch_id": branch_id, "tag_id": req.tag_id}


@router.delete("/branches/{branch_id}/tags/{tag_id}", status_code=200, summary="브랜치 태그 제거")
def remove_branch_tag(branch_id: str, tag_id: str, db: Session = Depends(get_db)):
    """브랜치에서 태그를 제거합니다.

    - 태그 자체는 삭제되지 않습니다. 브랜치와 태그의 연결만 끊습니다.
    - 같은 태그를 다른 브랜치에 다시 붙이는 것은 가능합니다.
    - 이미 달려있지 않은 태그를 제거하려 하면 404를 반환합니다.
    """
    if repository.get_branch(db, branch_id) is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    removed = repository.remove_branch_tag(db, branch_id, tag_id)
    if not removed:
        raise HTTPException(status_code=404, detail="해당 브랜치에 해당 태그가 달려있지 않습니다")
    return {"branch_id": branch_id, "tag_id": tag_id, "removed": True}
