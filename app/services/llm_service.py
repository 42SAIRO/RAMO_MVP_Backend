import traceback

from dotenv import load_dotenv

from app.database import SessionLocal
from app.repositories import repository
from app.services import context_builder, auto_tagger, embedding_service
from app.services.persona_seeds import PERSONA_COMMON_RULES
from app.services.providers import get_provider

load_dotenv()

ROLE_QUESTION_KEYWORDS = ("역할", "파트너")


def _is_role_question(user_message: str) -> bool:
    return any(keyword in user_message for keyword in ROLE_QUESTION_KEYWORDS)


def _build_system_prompt(
    model_provider: str,
    model_name: str,
    persona_prompt: str | None = None,
    persona_name: str | None = None,
    user_message: str = "",
) -> str:
    base = (
        "너는 RAMO(Route-Aware Memory Organizer)라는 AI 챗봇 서비스야. RAMO의 주요 기능은 다음과 같아. "
        "(1) 브랜치: 대화가 한 줄로만 이어지지 않고, 특정 메시지(노드)에서 새 브랜치로 분기해서 여러 방향으로 대화를 이어갈 수 있어. "
        "각 브랜치는 자동으로 이름과 설명이 붙고, 관련 정보는 브랜치를 넘나들며 필요할 때 연결해서 기억해. "
        "(2) 브랜치 시각화: 세션의 브랜치 구조를 그래프로 보여주고, 노드를 클릭해 그 대화로 이동하거나 스플릿뷰로 옆에 나란히 열어볼 수 있어. "
        "(3) 노드/브랜치 병합(merge): 서로 다른 브랜치를 하나로 합쳐 새 브랜치를 만들 수 있고, 합칠 만한 브랜치를 추천받을 수도 있어. "
        "(4) 모델 선택/비교: 대화마다 사용할 LLM 모델을 고를 수 있고(model_provider, model_name), 두 모델의 답변을 비교해서 "
        "원하는 답변을 선택하거나 사용자가 지정한 방식으로 두 답변을 합칠(융합) 수 있고, 두 답변의 공통점/차이점 분석도 볼 수 있어. "
        "(5) 도움 받기: 브랜치(노드)마다 '도움 받기'로 기획 컨설턴트/번역 전문가/글쓰기 코치/학습 멘토/개발 파트너/어학 코치 같은 "
        "특정 역할(파트너)을 지정해서 그 역할에 맞게 대화를 이어갈 수 있어. 필요 없어지면 '도움 해제'로 그 역할을 해제할 수 있어. "
        "(6) 세션/휴지통: 세션이나 브랜치를 삭제하면 휴지통으로 이동하고, 거기서 복원하거나 영구 삭제할 수 있어. "
        "사용자가 'RAMO가 뭐야', 'RAMO 기능이 뭐야', '도움 받기가 뭐야', '도움 해제가 뭐야' 처럼 RAMO 서비스나 기능에 대해 물으면 "
        "위 설명을 바탕으로 정확하게 답해. 모르는 기능을 지어내지 말고, 위에서 설명한 기능 범위 안에서만 답해. "
        "사용자가 지금 답변하는 모델이 뭔지 물어보면, 지금 사용 중인 모델은 "
        f"{model_provider} / {model_name}이라고 정확히 답해. "
        f"지금 사용 중인 모델에 대해 설명해달라고 한다면, {model_provider} / {model_name}이라고 정확히 답한 후 모델에 대한 설명을 덧붙여. "
        "수학 문제, 공식, 방정식 등 수식이 필요한 답변은 일반 텍스트로 풀어쓰지 말고 반드시 LaTeX 문법으로 써. "
        "문장 중간에 들어가는 인라인 수식은 $...$ 로 감싸고, 독립된 수식이나 여러 줄짜리 수식은 $$...$$ 로 감싸서 써. "
        "그 외 질문에는 간결하고 정확하게 답해줘."
    )
    is_role_question = _is_role_question(user_message)

    if persona_prompt:
        if is_role_question:
            identity_note = (
                f"지금 이 브랜치는 '도움 받기'로 '{persona_name}' 파트너(역할)가 지정되어 있어. "
                f"사용자가 지금 맡은 역할을 물었으니, 반드시 '{persona_name}'라고 정확히 답하고 그 역할에 맞게 행동해."
            )
        else:
            identity_note = (
                "지금 사용자 질문은 역할을 묻는 게 아니라 너의 정체성이나 서비스 자체를 묻는 질문이야(예: '너 누구야', 'RAMO가 뭐야'). "
                "이런 질문에는 지금 어떤 역할이 지정되어 있는지와 상관없이 페르소나 이름이 아니라 항상 위 RAMO 설명을 기준으로만 답해."
            )
        return (
            f"{base}\n\n{identity_note}\n\n{PERSONA_COMMON_RULES}\n\n"
            f"지금 이 대화에서는 다음 역할로 응답해:\n{persona_prompt}"
        )

    if is_role_question:
        identity_note = (
            "사용자가 지금 맡은 역할을 물었어. 지금은 '도움 받기'로 지정된 역할(파트너)이 없는 기본 상태이니 그대로 답해."
        )
        return f"{base}\n\n{identity_note}"

    return base

_FILE_TOOL = {
    "name": "get_file_content",
    "description": "파일의 전체 내용을 가져옵니다. 요약만으로 정확한 답변이 어려울 때만 사용하세요.",
    "parameters": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "전체 내용을 가져올 파일 이름",
            }
        },
        "required": ["filename"],
    },
}


def _make_file_tool_handler(file_lookup: dict[str, str]):
    def handler(tool_name: str, args: dict) -> str:
        if tool_name == "get_file_content":
            filename = args.get("filename", "")
            return file_lookup.get(filename, f"파일 '{filename}'을 찾을 수 없습니다.")
        return "알 수 없는 도구입니다."
    return handler


def _generate(context, model_provider, model_name, file_lookup, persona_prompt=None, persona_name=None, user_message=""):
    provider = get_provider(model_provider)
    system_prompt = _build_system_prompt(model_provider, model_name, persona_prompt, persona_name, user_message)
    if file_lookup:
        return provider.generate_with_tools(
            context, model_name, system_prompt, [_FILE_TOOL], _make_file_tool_handler(file_lookup)
        )
    return provider.generate(context, model_name, system_prompt)


def handle_chat(db, req, background_tasks=None):
    from fastapi import HTTPException
    branch = repository.get_branch(db, req.branch_id)
    if branch is None:
        raise HTTPException(status_code=404, detail=f"branch_id '{req.branch_id}' 를 찾을 수 없습니다. POST /sessions 로 새 세션을 만들고 응답의 main_branch_id 를 사용하세요.")

    if branch.status != "active":
        raise HTTPException(status_code=409, detail=f"branch가 '{branch.status}' 상태입니다. 채팅은 active 브랜치에서만 가능합니다.")

    is_first_message = branch.head_id is None
    is_first_branch_message = repository.count_branch_chat_messages(db, req.branch_id) == 0
    session_id = branch.session_id
    parent_id = branch.head_id
    attachment_file_ids = list(dict.fromkeys(req.file_ids or []))

    try:
        repository.validate_message_file_ids(db, attachment_file_ids, session_id, req.branch_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    context = context_builder.build_context(db, req.branch_id, req.message, req.model_name)

    # 파일 도구 핸들러용 full-text 조회 (요약은 context_builder에서 이미 주입됨)
    all_files = (
        repository.get_session_files(db, session_id)
        + repository.get_branch_files(db, req.branch_id)
    )
    file_lookup = {
        f.filename: (f.extracted_text if f.file_type != "image" else "이 파일은 이미지이며 이미 대화에 첨부되어 있습니다.")
        for f in all_files
    } if all_files else {}

    persona_prompt = None
    persona_name = None
    if branch.persona_id:
        persona = repository.get_persona(db, branch.persona_id)
        if persona:
            persona_prompt = persona.system_prompt
            persona_name = persona.name

    db.rollback()

    try:
        answer, input_tokens, output_tokens = _generate(
            context, req.model_provider, req.model_name, file_lookup, persona_prompt, persona_name, req.message
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM API 오류: {e}")

    branch = repository.get_branch(db, req.branch_id)
    if branch is None:
        raise HTTPException(status_code=404, detail=f"branch_id '{req.branch_id}' 를 찾을 수 없습니다.")
    if branch.status != "active":
        raise HTTPException(status_code=409, detail=f"branch가 '{branch.status}' 상태입니다. 채팅은 active 브랜치에서만 가능합니다.")
    if branch.head_id != parent_id:
        raise HTTPException(status_code=409, detail="대화가 다른 요청으로 먼저 갱신되었습니다. 최신 메시지를 불러온 뒤 다시 전송하세요.")

    user_msg = repository.save_message(
        db,
        session_id=session_id,
        branch_id=req.branch_id,
        role="user",
        content=req.message,
        parent_id=parent_id,
        model_provider=req.model_provider,
        model_name=req.model_name,
    )
    repository.attach_files_to_message(
        db,
        attachment_file_ids,
        user_msg.id,
        session_id,
        req.branch_id,
    )

    bot_msg = repository.save_message(
        db,
        session_id=session_id,
        branch_id=req.branch_id,
        role="assistant",
        content=answer,
        parent_id=user_msg.id,
        model_provider=req.model_provider,
        model_name=req.model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    branch.head_id = bot_msg.id
    db.commit()

    postprocess_args = {
        "session_id": session_id,
        "branch_id": req.branch_id,
        "user_message_id": user_msg.id,
        "user_message": req.message,
        "assistant_message_id": bot_msg.id,
        "assistant_message": answer,
        "should_update_session_title": is_first_message,
        "should_update_branch_name": is_first_branch_message and (branch.parent_branch_id is not None or branch.is_merge),
    }
    if background_tasks is not None:
        background_tasks.add_task(_run_chat_postprocess, **postprocess_args)
    else:
        _run_chat_postprocess(**postprocess_args)

    return answer, user_msg, bot_msg


def _run_chat_postprocess(
    session_id: str,
    branch_id: str,
    user_message_id: int,
    user_message: str,
    assistant_message_id: int,
    assistant_message: str,
    should_update_session_title: bool,
    should_update_branch_name: bool,
) -> None:
    db = SessionLocal()
    try:
        try:
            embedding_service.save_message_embedding(db, user_message_id, user_message)
            embedding_service.save_message_embedding(db, assistant_message_id, assistant_message)
        except Exception:
            db.rollback()
            print(f"[chat_postprocess] embedding 저장 실패 branch_id={branch_id}", flush=True)
            traceback.print_exc()

        if should_update_session_title:
            try:
                title = auto_tagger.generate_session_name(user_message, assistant_message)
                repository.update_session_title(db, session_id, title)
            except Exception:
                db.rollback()
                print(f"[chat_postprocess] session title 갱신 실패 session_id={session_id}", flush=True)
                traceback.print_exc()

        if should_update_branch_name:
            try:
                branch_name = auto_tagger.generate_branch_name_from_qa(user_message, assistant_message)
                repository.update_branch_name(db, branch_id, branch_name)
            except Exception:
                db.rollback()
                print(f"[chat_postprocess] branch name 갱신 실패 branch_id={branch_id}", flush=True)
                traceback.print_exc()

        try:
            auto_tagger.auto_describe_branch(db, branch_id)
        except Exception:
            db.rollback()
            print(f"[chat_postprocess] auto_describe_branch 실패 branch_id={branch_id}", flush=True)
            traceback.print_exc()
    finally:
        db.close()
