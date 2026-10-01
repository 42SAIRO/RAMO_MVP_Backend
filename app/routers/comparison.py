from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import (
    CompareRequest, CompareResponse, ModelResponse,
    AnalyzeResponse,
    SelectRequest, SelectResponse,
    MergeRequest, MergeResponse,
)
from app.services import comparison_service

router = APIRouter(prefix="/compare", tags=["Model Comparison"])


@router.post("", response_model=CompareResponse, summary="두 모델 동시 답변 요청")
def compare_chat(req: CompareRequest, db: Session = Depends(get_db)):
    """같은 질문을 두 모델에 동시에 보내고 각 답변을 반환합니다.

    - 답변은 브랜치에 저장되지 않습니다. 이후 `/select` 또는 `/merge`로 선택하면 저장됩니다.
    - 반환된 `comparison_id`를 이후 API에 사용하세요.
    """
    comp = comparison_service.compare_chat(
        db,
        branch_id=req.branch_id,
        message=req.message,
        model_a_provider=req.model_a.provider,
        model_a_name=req.model_a.name,
        model_b_provider=req.model_b.provider,
        model_b_name=req.model_b.name,
    )
    return CompareResponse(
        comparison_id=comp.id,
        response_a=ModelResponse(
            content=comp.response_a_content,
            model_provider=comp.response_a_provider,
            model_name=comp.response_a_model,
        ),
        response_b=ModelResponse(
            content=comp.response_b_content,
            model_provider=comp.response_b_provider,
            model_name=comp.response_b_model,
        ),
    )


@router.post("/{comparison_id}/analyze", response_model=AnalyzeResponse, summary="두 답변 유사점/차이점 분석")
def analyze(comparison_id: str, db: Session = Depends(get_db)):
    """두 모델의 답변에서 유사한 부분과 다른 부분을 LLM으로 분석합니다.

    - 분석은 `openai/gpt-4o-mini`로 수행됩니다.
    - `select` / `merge` 없이 여러 번 호출 가능합니다.
    """
    similarities, differences = comparison_service.analyze_comparison(db, comparison_id)
    return AnalyzeResponse(
        comparison_id=comparison_id,
        similarities=similarities,
        differences=differences,
    )


@router.post("/{comparison_id}/select", response_model=SelectResponse, summary="한 모델 답변 선택하여 대화 이어가기")
def select(comparison_id: str, req: SelectRequest, db: Session = Depends(get_db)):
    """두 답변 중 원하는 쪽을 선택해 브랜치에 저장하고 대화를 이어갑니다.

    - `selected`: `"a"` 또는 `"b"`
    - 선택 후에는 해당 comparison을 다시 사용할 수 없습니다.
    """
    bot_msg = comparison_service.select_response(db, comparison_id, req.selected)
    return SelectResponse(
        message_id=bot_msg.id,
        branch_id=bot_msg.branch_id,
        content=bot_msg.content,
    )


@router.post("/{comparison_id}/merge", response_model=MergeResponse, summary="두 답변 융합하여 대화 이어가기")
def merge(comparison_id: str, req: MergeRequest, db: Session = Depends(get_db)):
    """사용자의 지시에 따라 두 답변을 융합한 결과를 브랜치에 저장하고 대화를 이어갑니다.

    - `instruction` 예시: "A 답변의 예시 코드 부분이 좋고, B 답변의 결론 부분이 좋아"
    - 융합 후에는 해당 comparison을 다시 사용할 수 없습니다.
    """
    merged_content, message_id = comparison_service.merge_responses(
        db, comparison_id, req.instruction, req.model_provider, req.model_name
    )
    return MergeResponse(
        comparison_id=comparison_id,
        merged_content=merged_content,
        message_id=message_id,
    )
