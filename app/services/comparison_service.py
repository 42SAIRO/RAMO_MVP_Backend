from fastapi import HTTPException

from app.repositories import repository
from app.services import context_builder, token_budget
from app.services.providers import get_provider

SYSTEM_PROMPT = "너는 친절한 한국어 챗봇이야. 질문에 간결하고 정확하게 답해줘."


def compare_chat(db, branch_id: str, message: str,
                 model_a_provider: str, model_a_name: str,
                 model_b_provider: str, model_b_name: str):
    branch = repository.get_branch(db, branch_id)
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    if branch.status != "active":
        raise HTTPException(status_code=409, detail=f"branch가 '{branch.status}' 상태입니다")

    # 두 모델이 같은 context를 나눠 쓰므로, context window가 더 작은 쪽 기준으로 압축 여부를 판단한다.
    stricter_model = min(model_a_name, model_b_name, key=token_budget.get_context_window)
    context = context_builder.build_context(db, branch_id, message, stricter_model)

    try:
        content_a, _, _ = get_provider(model_a_provider).generate(context, model_a_name, SYSTEM_PROMPT)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"모델 A 오류: {e}")

    try:
        content_b, _, _ = get_provider(model_b_provider).generate(context, model_b_name, SYSTEM_PROMPT)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"모델 B 오류: {e}")

    return repository.create_comparison(
        db,
        branch_id=branch_id,
        user_message=message,
        response_a_content=content_a,
        response_a_provider=model_a_provider,
        response_a_model=model_a_name,
        response_b_content=content_b,
        response_b_provider=model_b_provider,
        response_b_model=model_b_name,
    )


def analyze_comparison(db, comparison_id: str, analyzer_provider: str = "openai", analyzer_model: str = "gpt-4o-mini"):
    comparison = repository.get_comparison(db, comparison_id)
    if comparison is None:
        raise HTTPException(status_code=404, detail="comparison을 찾을 수 없습니다")

    prompt = f"""두 AI 모델의 답변을 비교분석해줘.

[질문]
{comparison.user_message}

[모델 A 답변 ({comparison.response_a_provider} / {comparison.response_a_model})]
{comparison.response_a_content}

[모델 B 답변 ({comparison.response_b_provider} / {comparison.response_b_model})]
{comparison.response_b_content}

반드시 아래 두 섹션 형식으로 답해줘:

## 유사한 부분
두 답변에서 공통적으로 다루는 내용이나 관점을 서술해.

## 다른 부분
두 답변이 서로 다르게 접근하거나 한 쪽만 다루는 내용을 서술해."""

    context = [{"role": "user", "content": prompt}]

    try:
        analysis, _, _ = get_provider(analyzer_provider).generate(
            context, analyzer_model, "너는 두 텍스트를 비교분석하는 전문가야."
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"분석 LLM 오류: {e}")

    similarities, differences = _parse_analysis(analysis)
    return similarities, differences


def _parse_analysis(analysis: str) -> tuple[str, str]:
    if "## 유사한 부분" in analysis and "## 다른 부분" in analysis:
        parts = analysis.split("## 다른 부분")
        similarities = parts[0].replace("## 유사한 부분", "").strip()
        differences = parts[1].strip()
        return similarities, differences
    return analysis.strip(), ""


def select_response(db, comparison_id: str, selected: str):
    if selected not in ("a", "b"):
        raise HTTPException(status_code=422, detail="selected는 'a' 또는 'b'여야 합니다")

    comparison = repository.get_comparison(db, comparison_id)
    if comparison is None:
        raise HTTPException(status_code=404, detail="comparison을 찾을 수 없습니다")
    if comparison.status == "done":
        raise HTTPException(status_code=409, detail="이미 완료된 comparison입니다")

    branch = repository.get_branch(db, comparison.branch_id)

    if selected == "a":
        content, provider, model = comparison.response_a_content, comparison.response_a_provider, comparison.response_a_model
    else:
        content, provider, model = comparison.response_b_content, comparison.response_b_provider, comparison.response_b_model

    user_msg = repository.save_message(
        db,
        session_id=branch.session_id,
        branch_id=comparison.branch_id,
        role="user",
        content=comparison.user_message,
        parent_id=branch.head_id,
    )
    bot_msg = repository.save_message(
        db,
        session_id=branch.session_id,
        branch_id=comparison.branch_id,
        role="assistant",
        content=content,
        parent_id=user_msg.id,
        model_provider=provider,
        model_name=model,
    )

    branch.head_id = bot_msg.id
    repository.mark_comparison_done(db, comparison_id)
    db.commit()
    return bot_msg


def merge_responses(db, comparison_id: str, instruction: str,
                    merger_provider: str = "openai", merger_model: str = "gpt-4o-mini"):
    comparison = repository.get_comparison(db, comparison_id)
    if comparison is None:
        raise HTTPException(status_code=404, detail="comparison을 찾을 수 없습니다")
    if comparison.status == "done":
        raise HTTPException(status_code=409, detail="이미 완료된 comparison입니다")

    branch = repository.get_branch(db, comparison.branch_id)

    prompt = f"""사용자의 지시에 따라 두 AI 답변을 융합해서 최적의 답변을 만들어줘.

[원래 질문]
{comparison.user_message}

[모델 A 답변]
{comparison.response_a_content}

[모델 B 답변]
{comparison.response_b_content}

[사용자 지시]
{instruction}

위 지시에 맞게 두 답변의 원하는 부분을 조합해서 자연스럽고 완성도 높은 답변을 만들어줘."""

    context = [{"role": "user", "content": prompt}]

    try:
        merged_content, _, _ = get_provider(merger_provider).generate(
            context, merger_model, "너는 두 답변을 융합해 최적의 답변을 만드는 전문가야."
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"융합 LLM 오류: {e}")

    user_msg = repository.save_message(
        db,
        session_id=branch.session_id,
        branch_id=comparison.branch_id,
        role="user",
        content=comparison.user_message,
        parent_id=branch.head_id,
    )
    bot_msg = repository.save_message(
        db,
        session_id=branch.session_id,
        branch_id=comparison.branch_id,
        role="assistant",
        content=merged_content,
        parent_id=user_msg.id,
        model_provider=merger_provider,
        model_name=merger_model,
    )

    branch.head_id = bot_msg.id
    repository.mark_comparison_done(db, comparison_id)
    db.commit()
    return merged_content, bot_msg.id
