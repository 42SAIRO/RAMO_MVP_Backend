from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import GraphOut
from app.repositories import repository
from app.services import graph_service

router = APIRouter(tags=["Graph"])


@router.get("/sessions/{session_id}/graph", response_model=GraphOut, summary="브랜치 그래프 조회")
def session_graph(
    session_id: str,
    include_inactive: bool = False,
    db: Session = Depends(get_db),
):
    """세션의 브랜치 구조를 nodes / edges 형태로 반환합니다. 프론트엔드 그래프 시각화 용도입니다.

    - **nodes**: 브랜치 하나하나가 node. label(브랜치 이름), description, status, is_merge, message_count 포함
    - **edges**: parent → child 분기 관계. type="fork"는 fork_from_message_id로 어느 메시지에서
      갈라졌는지 표시하고, type="merge"는 머지 브랜치가 가진 여러 부모 관계를 각각 하나의 edge로 표시
    - **include_inactive**: true 이면 deleted 브랜치도 포함 (기본값 false)
    """
    branches = repository.list_branches(db, session_id)
    message_counts = repository.get_message_count_by_branch(db, session_id)
    merge_edges = repository.list_merge_edges(db, session_id)
    personas_by_id = {persona.id: persona for persona in repository.list_personas(db)}
    return graph_service.build_branch_graph(branches, message_counts, include_inactive, merge_edges, personas_by_id)
