from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import (
    CreateSessionRequest, SessionOut, ConversationOut, BranchOut,
    UpdateSessionTitleRequest, BranchSearchResult,
    UpdateMemoryRequest, MemoryOut, homemessage, eggmessage
)
from app.repositories import repository
from app.services import auto_tagger

router = APIRouter(tags=["Sessions"])


def _with_branch_metadata(db, branch):
    branch.merge_parent_ids = repository.get_merge_parent_ids(db, branch.id)
    persona = repository.get_persona(db, branch.persona_id) if branch.persona_id else None
    branch.persona_slug = persona.slug if persona else None
    branch.persona_key = persona.slug if persona else None
    branch.persona_name = persona.name if persona else None
    return branch


@router.get("/home", response_model=homemessage, summary="이스터에그1")
def home_message():
    """홈메세지"""
    return homemessage(message="MY과 SC의 비밀노트\nJH, MS, SH의 말랑이 대작전~!")


@router.get("/egg", response_model=eggmessage, summary="이스터에그2")
def egg_message():
    """프론트팀 이스터에그"""
    return eggmessage(message="교수님 사랑합니다. 쿠다 화이팅!")

@router.post("/sessions", response_model=SessionOut, summary="세션 생성")
def new_session(req: CreateSessionRequest, db: Session = Depends(get_db)):
    """새 대화 세션과 root branch(main)를 함께 생성합니다.

    - 세션은 전체 대화 그래프를 담는 단위입니다.
    - 생성과 동시에 기본 브랜치(main)가 자동으로 만들어집니다.
    - 응답의 `main_branch_id`로 바로 채팅을 시작할 수 있습니다.
    """
    conv, main = repository.create_conversation(db, title=req.title)
    return SessionOut(id=conv.id, title=conv.title, main_branch_id=main.id)


@router.get("/sessions", response_model=list[ConversationOut], summary="세션 목록 조회")
def all_sessions(db: Session = Depends(get_db)):
    """전체 대화 세션 목록을 최신순으로 반환합니다. 사이드바 표시 용도입니다."""
    return repository.list_conversation_summaries(db)


@router.patch("/sessions/{session_id}/title", response_model=ConversationOut, summary="세션 이름 수정")
def update_session_title(session_id: str, req: UpdateSessionTitleRequest, db: Session = Depends(get_db)):
    """세션 제목을 직접 수정합니다.

    - 자동 생성된 제목이 마음에 들지 않을 때 사용합니다.
    - 빈 문자열은 허용되지 않습니다.
    """
    if not req.title.strip():
        raise HTTPException(status_code=422, detail="title은 빈 문자열일 수 없습니다.")
    conv = repository.update_session_title(db, session_id, req.title.strip())
    if conv is None:
        raise HTTPException(status_code=404, detail="session을 찾을 수 없습니다.")
    return conv


@router.get("/sessions/{session_id}/search", response_model=list[BranchSearchResult], summary="브랜치 검색")
def search_branches(session_id: str, q: str, db: Session = Depends(get_db)):
    """브랜치 이름 또는 태그 이름으로 브랜치를 검색합니다.

    - `q`: 검색어 (부분 일치, 대소문자 무관)
    - 브랜치 이름에 검색어가 포함되거나, 해당 브랜치에 달린 태그 이름에 포함되면 결과에 포함됩니다.
    - 결과에는 브랜치 id, 이름, 상태, 태그 목록이 포함됩니다.
    - deleted 상태 브랜치는 제외됩니다.
    """
    if not q.strip():
        raise HTTPException(status_code=422, detail="검색어를 입력하세요.")
    return repository.search_branches(db, session_id, q.strip())


@router.get("/sessions/{session_id}/branches", response_model=list[BranchOut], summary="세션의 브랜치 목록 조회")
def session_branches(session_id: str, db: Session = Depends(get_db)):
    """특정 세션에 속한 모든 브랜치 목록을 반환합니다.

    - root branch(main)와 사용자가 생성한 child branch가 모두 포함됩니다.
    - `parent_branch_id`와 `fork_from_message_id`로 분기 관계를 확인할 수 있습니다.
    - `is_merge`가 true인 브랜치는 `merge_parent_ids`에 담긴 여러 브랜치를 부모로 갖습니다.
    """
    branches = repository.list_branches(db, session_id)
    return [_with_branch_metadata(db, branch) for branch in branches]


@router.delete("/sessions/{session_id}", summary="세션 휴지통 이동")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """세션을 휴지통으로 이동합니다 (소프트 삭제).

    - 세션 안의 브랜치 상태는 바뀌지 않습니다. 목록/화면에서만 숨겨집니다.
    - 7일 후 자동으로 영구 삭제되거나, `/trash`에서 직접 복원/영구삭제할 수 있습니다.
    """
    conv = repository.soft_delete_session(db, session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="session을 찾을 수 없습니다")
    return {"session_id": session_id, "status": "deleted"}


@router.get("/sessions/{session_id}/branch-trash", response_model=list[BranchOut], summary="세션별 브랜치 휴지통 조회")
def branch_trash(session_id: str, db: Session = Depends(get_db)):
    """이 세션 안에서 휴지통으로 이동된 브랜치만 반환합니다.

    - 다른 세션의 삭제된 브랜치는 섞이지 않습니다.
    - `deleted_at` 기준 7일 후 자동으로 영구 삭제됩니다.
    """
    return [_with_branch_metadata(db, branch) for branch in repository.list_branch_trash(db, session_id)]


@router.get("/sessions/{session_id}/memory", response_model=MemoryOut, summary="세션 메모리 조회")
def get_session_memory(session_id: str, db: Session = Depends(get_db)):
    """세션에 저장된 사용자 정보 메모리를 조회합니다.

    - 모든 브랜치에서 공유되는 사용자 정보(이름, 직업, 관심사 등)를 반환합니다.
    - 메모리가 없으면 `memory` 필드가 null입니다.
    """
    conv = repository.list_conversations(db)
    exists = any(c.id == session_id for c in conv)
    if not exists:
        raise HTTPException(status_code=404, detail="session을 찾을 수 없습니다.")
    memory = repository.get_session_memory(db, session_id)
    return MemoryOut(session_id=session_id, memory=memory)


@router.patch("/sessions/{session_id}/memory", response_model=MemoryOut, summary="세션 메모리 수동 수정")
def update_session_memory(session_id: str, req: UpdateMemoryRequest, db: Session = Depends(get_db)):
    """세션 메모리를 직접 수정합니다.

    - 사용자 정보를 수동으로 입력하거나 교정할 때 사용합니다.
    - LLM 자동 추출 결과를 덮어쓸 수 있습니다.
    """
    conv = repository.update_session_memory(db, session_id, req.memory)
    if conv is None:
        raise HTTPException(status_code=404, detail="session을 찾을 수 없습니다.")
    return MemoryOut(session_id=session_id, memory=conv.memory)


@router.post("/sessions/{session_id}/memory/extract", response_model=MemoryOut, summary="세션 메모리 LLM 자동 추출")
def extract_session_memory(session_id: str, db: Session = Depends(get_db)):
    """세션의 모든 브랜치 대화를 분석해 사용자 정보를 자동으로 추출하고 저장합니다.

    - 모든 브랜치의 메시지를 수집해 LLM에 전달합니다.
    - 추출된 정보는 세션 메모리에 저장되며, 이후 채팅 context에 자동 주입됩니다.
    """
    branches = repository.list_branches(db, session_id)
    if not branches:
        raise HTTPException(status_code=404, detail="session을 찾을 수 없습니다.")

    all_messages = []
    for branch in branches:
        msgs = repository.get_branch_messages(db, branch.id)
        all_messages.extend(msgs)

    if not all_messages:
        raise HTTPException(status_code=422, detail="추출할 메시지가 없습니다.")

    result = auto_tagger.extract_user_memory(all_messages)
    conv = repository.update_session_memory(db, session_id, result)
    return MemoryOut(session_id=session_id, memory=conv.memory)
