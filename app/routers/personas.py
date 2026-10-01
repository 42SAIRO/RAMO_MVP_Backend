from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import PersonaOut
from app.repositories import repository

router = APIRouter(tags=["Personas"])


@router.get("/personas", response_model=list[PersonaOut], summary="페르소나 목록 조회")
def list_personas(db: Session = Depends(get_db)):
    """브랜치 생성 시 선택할 수 있는 페르소나 목록을 반환합니다.

    - 각 페르소나는 시스템 프롬프트(역할)와 추천 모델(`default_model_provider`/`default_model_name`)을 가집니다.
    - `POST /branches`의 `persona_id`에 이 목록의 `id`를 넣으면 해당 역할로 대화하는 브랜치가 만들어집니다.
    - 추천 모델은 프론트에서 모델 선택 UI를 미리 채우는 용도이며, 실제 사용 모델은 `/chat` 요청의 `model_provider`/`model_name`이 결정합니다.
    """
    return repository.list_personas(db)
