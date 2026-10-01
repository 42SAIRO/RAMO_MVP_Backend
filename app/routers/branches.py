import traceback

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.schemas.schemas import (
    ChatRequest, ChatResponse, MessageOut, BranchOut, CreateBranchRequest,
    UpdateBranchNameRequest, PatchBranchRequest, MergeBranchRequest,
    SelectMainBranchResponse, SetBranchPersonaRequest,
)
from app.repositories import repository
from app.services import llm_service, auto_tagger, context_builder, merge_recommender

router = APIRouter(tags=["Branches & Chat"])

DEFAULT_PERSONA_NAME = "Ramo"


def _resolve_persona(db, *, persona_id: str | None = None, persona_slug: str | None = None):
    if persona_id and persona_slug:
        persona = repository.get_persona(db, persona_id)
        if persona is None or persona.slug != persona_slug:
            raise HTTPException(status_code=422, detail="persona_id와 persona_slug가 서로 일치하지 않습니다")
        return persona

    if persona_id:
        return repository.get_persona(db, persona_id)

    if persona_slug:
        return repository.get_persona_by_slug(db, persona_slug)

    return None


def _with_branch_metadata(db, branch):
    branch.merge_parent_ids = repository.get_merge_parent_ids(db, branch.id)
    persona = repository.get_persona(db, branch.persona_id) if branch.persona_id else None
    branch.persona_slug = persona.slug if persona else None
    branch.persona_key = persona.slug if persona else None
    branch.persona_name = persona.name if persona else None
    return branch


def _with_file_content_url(request: Request, file):
    file.content_url = (
        str(request.url_for("get_file_content", file_id=file.id))
        if file.content_data or (file.file_type == "image" and file.image_data)
        else None
    )
    return file


def _with_message_metadata(db, messages, request: Request):
    persona_name_by_branch_id = {}
    files_by_message_id = repository.get_files_by_message_ids(db, [message.id for message in messages])

    for message in messages:
        branch_id = message.branch_id

        if branch_id not in persona_name_by_branch_id:
            persona_name_by_branch_id[branch_id] = _get_branch_persona_name(db, branch_id)

        message.persona_name = persona_name_by_branch_id[branch_id]
        message.attachments = [
            _with_file_content_url(request, file)
            for file in files_by_message_id.get(message.id, [])
        ]

    return messages


def _get_branch_persona_name(db, branch_id):
    if not branch_id:
        return DEFAULT_PERSONA_NAME

    branch = repository.get_branch(db, branch_id)

    if branch is None or not branch.persona_id:
        return DEFAULT_PERSONA_NAME

    persona = repository.get_persona(db, branch.persona_id)
    return persona.name if persona else DEFAULT_PERSONA_NAME


def _auto_tag_branch_background(session_id: str, branch_id: str) -> None:
    db = SessionLocal()
    try:
        try:
            auto_tagger.auto_tag_branch(db, session_id, branch_id)
        except Exception:
            db.rollback()
            print(f"[auto_tag_branch_background] auto_tag_branch 실패 branch_id={branch_id}", flush=True)
            traceback.print_exc()

        try:
            auto_tagger.auto_summarize_branch(db, branch_id)
        except Exception:
            db.rollback()
            print(f"[auto_tag_branch_background] auto_summarize_branch 실패 branch_id={branch_id}", flush=True)
            traceback.print_exc()

        try:
            auto_tagger.auto_describe_branch(db, branch_id)
        except Exception:
            db.rollback()
            print(f"[auto_tag_branch_background] auto_describe_branch 실패 branch_id={branch_id}", flush=True)
            traceback.print_exc()
    finally:
        db.close()


@router.post("/chat", response_model=ChatResponse, summary="메시지 전송 및 AI 응답")
def chat(req: ChatRequest, request: Request, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """특정 브랜치에서 메시지를 보내고 AI 응답을 받습니다.

    - `branch_id`에 해당하는 브랜치의 대화 흐름 위에 메시지가 추가됩니다.
    - 분기 이전 부모 브랜치의 맥락(context)은 자동으로 포함됩니다.
    - `model_provider`와 `model_name`으로 사용할 LLM을 지정할 수 있습니다.
    """
    answer, user_message, assistant_message = llm_service.handle_chat(db, req, background_tasks)
    messages = _with_message_metadata(db, [user_message, assistant_message], request)
    return ChatResponse(reply=answer, user_message=messages[0], assistant_message=messages[1])



@router.post("/branches", response_model=BranchOut, summary="브랜치 생성")
def make_branch(req: CreateBranchRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """특정 메시지를 분기점으로 새 브랜치를 생성합니다.

    - `parent_branch_id`: 분기할 기존 브랜치 ID
    - `fork_from_message_id`: 분기 기준 메시지 ID (이 메시지까지의 맥락을 공유함)
    - 새 브랜치가 생성되면 부모 브랜치 대화 전체를 분석해 태그, 누적 요약, 시각화용 짧은 설명을 자동으로 생성합니다 (백그라운드 실행).
    - `name`을 비워두면 일단 "새 가지"로 생성되고, 이 브랜치에서 첫 질문/답변이 오가면
      그 내용을 기준으로 이름이 자동으로 다시 지어집니다.
    - `persona_id` 또는 `persona_slug`를 지정하면 이 브랜치는 그 역할(페르소나)로 대화합니다 (`GET /personas` 참고).
      지정하지 않으면(둘 다 비워두면) 부모 브랜치의 파트너를 그대로 물려받습니다.

    **검증 조건**
    - `parent_branch_id`가 해당 session에 속해야 합니다.
    - `fork_from_message_id`가 parent branch의 메시지여야 합니다.
    - `persona_id` 또는 `persona_slug`를 지정했다면 존재하는 페르소나여야 합니다.
    """
    name = req.name or "새 가지"

    persona = _resolve_persona(db, persona_id=req.persona_id, persona_slug=req.persona_slug)
    if (req.persona_id is not None or req.persona_slug is not None) and persona is None:
        raise HTTPException(status_code=404, detail="persona를 찾을 수 없습니다")

    try:
        branch = repository.create_branch(
            db, req.session_id, req.parent_branch_id, req.fork_from_message_id, name, persona.id if persona else None
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    background_tasks.add_task(_auto_tag_branch_background, req.session_id, req.parent_branch_id)
    return _with_branch_metadata(db, branch)



@router.post("/branches/merge", response_model=BranchOut, summary="브랜치 머지")
def merge_branches(req: MergeBranchRequest, db: Session = Depends(get_db)):
    """선택한 여러 브랜치를 부모로 갖는 새 브랜치를 만듭니다.

    - `parent_branch_ids`: 머지할 브랜치 2개 이상.
    - 각 부모 브랜치는 분기 이전 조상 맥락까지 포함한 전체 대화를 AI로 요약하고,
      그 요약들을 새 브랜치의 초기 context로 사용합니다 (원문 전체를 그대로 이어붙이지 않음).
    - `name`을 비워두면 "병합 브랜치"로 생성되고, 첫 질문/답변이 오가면 자동으로 이름이 다시 지어집니다.
    """
    if len(set(req.parent_branch_ids)) < 2:
        raise HTTPException(status_code=422, detail="parent_branch_ids는 서로 다른 브랜치 2개 이상이어야 합니다.")

    summaries = {}
    for parent_id in req.parent_branch_ids:
        parent = repository.get_branch(db, parent_id)
        if parent is None or parent.session_id != req.session_id:
            raise HTTPException(status_code=400, detail=f"branch '{parent_id}'가 해당 session에 속하지 않습니다.")
        messages = context_builder.get_full_branch_messages(db, parent_id)
        summaries[parent_id] = auto_tagger.summarize_branch_for_merge(messages)

    name = req.name or "병합 브랜치"
    try:
        branch = repository.create_merge_branch(db, req.session_id, summaries, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _with_branch_metadata(db, branch)


@router.post("/branches/{branch_id}/auto-name", summary="브랜치 이름 자동 생성")
def auto_name_branch(branch_id: str, db: Session = Depends(get_db)):
    """브랜치 전체 대화를 분석해 이름을 자동 생성하고 저장합니다.

    - 브랜치 생성 시에는 fork 메시지 기준으로 이름이 자동 생성됩니다.
    - 대화가 더 쌓인 후 이름을 다시 생성하고 싶을 때 이 API를 사용하세요.
    """
    branch = repository.get_branch(db, branch_id)
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    name = auto_tagger.auto_name_branch(db, branch_id)
    return {"branch_id": branch_id, "name": name}


@router.post("/branches/{branch_id}/auto-tag", summary="브랜치 자동 태그")
def auto_tag_branch(branch_id: str, db: Session = Depends(get_db)):
    """브랜치 전체 대화를 분석해 내용 태그와 역할 태그를 함께 자동 생성합니다.

    - 내용 태그(`type=normal`, 핵심 주제)와 역할 태그(`type=role`, 번역/요약/피드백 등)를 한 번에 부여합니다.
    - 새 브랜치 생성 시 부모 브랜치에는 자동으로 실행됩니다.
    - 현재 대화 중인 마지막 브랜치에는 이 API로 수동 트리거하세요.
    - 이미 달린 태그는 유지되고, 새 태그만 추가됩니다.
    """
    branch = repository.get_branch(db, branch_id)
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    result = auto_tagger.auto_tag_branch(db, branch.session_id, branch_id)
    return {"branch_id": branch_id, **result}


@router.post("/branches/{branch_id}/summarize", summary="브랜치 대화 자동 요약")
def summarize_branch(branch_id: str, db: Session = Depends(get_db)):
    """브랜치 전체 대화를 분석해 중요한 결과·결론 위주로 요약을 생성하고 저장합니다.

    - 브랜치가 fork될 때 부모 브랜치에는 자동으로 실행됩니다.
    - 현재 대화 중인 마지막 브랜치는 이 API로 수동 트리거하세요.
    - 실행할 때마다 이전 요약을 새 요약으로 덮어씁니다.
    - 메시지가 없는 브랜치는 `summary`가 `null`로 반환됩니다.
    """
    branch = repository.get_branch(db, branch_id)
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    summary = auto_tagger.auto_summarize_branch(db, branch_id)
    return {"branch_id": branch_id, "summary": summary}


@router.post("/branches/{branch_id}/describe", summary="브랜치 시각화용 짧은 설명 생성")
def describe_branch(branch_id: str, db: Session = Depends(get_db)):
    """이 브랜치 자신의 대화만으로 그래프 시각화용 한두 문장 설명을 생성하고 저장합니다.

    - `summary`(누적 요약)와 달리 조상 브랜치의 내용은 포함하지 않고, 이 브랜치 자신의
      메시지만 바탕으로 만듭니다.
    - 브랜치가 fork될 때 부모 브랜치에는 자동으로 실행됩니다.
    - 현재 대화 중인 마지막 브랜치는 이 API로 수동 트리거하세요.
    - 실행할 때마다 이전 설명을 새 설명으로 덮어씁니다.
    - 메시지가 없는 브랜치는 `description`이 `null`로 반환됩니다.
    """
    branch = repository.get_branch(db, branch_id)
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    description = auto_tagger.auto_describe_branch(db, branch_id)
    return {"branch_id": branch_id, "description": description}


@router.get("/branches/{branch_id}/merge-candidates", summary="브랜치 머지 후보 추천")
def merge_candidates(branch_id: str, db: Session = Depends(get_db)):
    """이 브랜치와 머지할 만한 다른 브랜치를 추천합니다. 머지 모달에서 사용합니다.

    - 직계 조상/자손 브랜치는 이미 부모-자식으로 연결되어 있어 후보에서 제외됩니다.
    - 내용 유사도(`type=content`)가 일정 기준 이상인 브랜치만 후보가 됩니다.
    - 같은 역할 태그를 공유하면(`type=role`) 추가 이유로 붙지만, 역할만 겹쳐서는 후보가 되지 않습니다.
    """
    branch = repository.get_branch(db, branch_id)
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    candidates = merge_recommender.get_merge_candidates(db, branch_id)
    return {"branch_id": branch_id, "candidates": candidates}


@router.post("/branches/{branch_id}/select-main", response_model=SelectMainBranchResponse, summary="main 경로 선택")
def select_main_branch(branch_id: str, db: Session = Depends(get_db)):
    """선택한 브랜치와 그 모든 조상 브랜치를 main 경로로 표시합니다.

    - 선택한 브랜치부터 루트까지의 부모 체인 전체에 `is_main=true`를 설정합니다.
    - 같은 세션의 다른 브랜치는 모두 `is_main=false`로 초기화됩니다.
    - `main_branch_ids`: 루트부터 선택 브랜치 순서로 정렬된 is_main 브랜치 ID 목록입니다.
    """
    try:
        main_ids = repository.select_main_branch(db, branch_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return SelectMainBranchResponse(branch_id=branch_id, main_branch_ids=main_ids)


@router.post("/branches/{branch_id}/restore", response_model=BranchOut, summary="브랜치 복원 (하위 포함)")
def restore_branch(branch_id: str, db: Session = Depends(get_db)):
    """휴지통의 브랜치를 하위 브랜치까지 함께 복원합니다."""
    branch = repository.restore_branch_cascade(db, branch_id)
    if branch is None:
        raise HTTPException(status_code=404, detail="휴지통에서 branch를 찾을 수 없습니다")
    return _with_branch_metadata(db, branch)


@router.delete("/branches/{branch_id}", summary="브랜치 영구 삭제 (하위 포함)")
def purge_branch(branch_id: str, db: Session = Depends(get_db)):
    """브랜치와 하위 브랜치를 완전히 삭제합니다. 복원할 수 없습니다."""
    purged = repository.purge_branch_cascade(db, branch_id)
    if not purged:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    return {"branch_id": branch_id, "purged": True}


@router.patch("/branches/{branch_id}", response_model=BranchOut, summary="브랜치 상태/접힘 수정")
def patch_branch(branch_id: str, req: PatchBranchRequest, db: Session = Depends(get_db)):
    """브랜치 status 또는 is_collapsed를 수정합니다.

    - `status`: `inactive`(비활성화) / `deleted`(소프트 삭제) / `active`(복구)
    - `is_collapsed`: 그래프에서 브랜치를 접을지 여부
    - root branch(main)는 inactive / deleted로 변경할 수 없습니다.
    - `deleted`로 변경하면 이 브랜치에서 fork된 모든 하위 브랜치도 함께 deleted 처리됩니다 (재귀적).
    - 변경하고 싶은 필드만 body에 포함하면 됩니다 (나머지는 유지).
    """
    if req.status is not None and req.status not in ("active", "inactive", "deleted"):
        raise HTTPException(status_code=422, detail="status는 active / inactive / deleted 중 하나여야 합니다.")
    try:
        branch = repository.update_branch_status(db, branch_id, req.status, req.is_collapsed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다.")
    return _with_branch_metadata(db, branch)


@router.put("/branches/{branch_id}/persona", response_model=BranchOut, summary="브랜치 파트너 지정/변경 (도움 받기)")
def set_branch_persona(branch_id: str, req: SetBranchPersonaRequest, db: Session = Depends(get_db)):
    """이 브랜치와 그 하위 브랜치 전체에 파트너(페르소나)를 지정하거나 다른 파트너로 바꿉니다 ("도움 받기").

    - 이후 이 브랜치와 하위 브랜치의 `/chat` 응답은 지정한 파트너 역할로 생성됩니다.
    - 새로 fork되는 브랜치도 부모의 파트너를 그대로 물려받습니다.
    - 이미 파트너가 지정되어 있어도 새 `persona_id` 또는 `persona_slug`로 덮어씁니다 (이 브랜치와 하위 브랜치 전체).
    - 취소하려면 `DELETE /branches/{branch_id}/persona` ("도움 해제")를 사용하세요.
    """
    if req.persona_id is None and req.persona_slug is None:
        raise HTTPException(status_code=422, detail="persona_id 또는 persona_slug가 필요합니다")

    persona = _resolve_persona(db, persona_id=req.persona_id, persona_slug=req.persona_slug)
    if persona is None:
        raise HTTPException(status_code=404, detail="persona를 찾을 수 없습니다")
    branch = repository.update_branch_persona_cascade(db, branch_id, persona.id)
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    return _with_branch_metadata(db, branch)


@router.delete("/branches/{branch_id}/persona", response_model=BranchOut, summary="브랜치 파트너 해제 (도움 해제)")
def clear_branch_persona(branch_id: str, db: Session = Depends(get_db)):
    """이 브랜치와 그 하위 브랜치 전체에서 지정된 파트너를 해제합니다 ("도움 해제").

    - 이후 이 브랜치와 하위 브랜치의 `/chat` 응답은 기본 RAMO로 돌아갑니다.
    - 상위 브랜치에 지정된 파트너는 건드리지 않고 그대로 유지됩니다.
    - 파트너가 지정되어 있지 않아도 그대로 성공합니다 (idempotent).
    """
    branch = repository.update_branch_persona_cascade(db, branch_id, None)
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    return _with_branch_metadata(db, branch)


@router.patch("/branches/{branch_id}/name", response_model=BranchOut, summary="브랜치 이름 수정")
def update_branch_name(branch_id: str, req: UpdateBranchNameRequest, db: Session = Depends(get_db)):
    """브랜치 이름을 직접 수정합니다.

    - 자동 생성된 이름이 마음에 들지 않을 때 사용합니다.
    - `/auto-name` 엔드포인트로 AI가 다시 생성하게 할 수도 있습니다.
    """
    if not req.name.strip():
        raise HTTPException(status_code=422, detail="name은 빈 문자열일 수 없습니다.")
    branch = repository.update_branch_name(db, branch_id, req.name.strip())
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다.")
    return _with_branch_metadata(db, branch)


@router.get("/branches/{branch_id}/messages", response_model=list[MessageOut], summary="브랜치 메시지 조회")
def branch_messages(
    branch_id: str,
    request: Request,
    include_inherited: bool = False,
    db: Session = Depends(get_db),
):
    """브랜치 메시지를 조회합니다.

    - `include_inherited=false` (기본값): 이 브랜치에서 직접 작성된 메시지만 반환합니다.
    - `include_inherited=true`: 분기 이전 부모 브랜치 맥락까지 포함해 반환합니다. 채팅창 표시 용도입니다.

    **예시** — main에서 M2 기준으로 분기한 branch B의 경우

    ```
    include_inherited=false → [M5, M6]           (branch B에서 작성한 것만)
    include_inherited=true  → [M1, M2, M5, M6]   (분기 이전 main 맥락 포함)
    ```
    """
    branch = repository.get_branch(db, branch_id)
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    if branch.head_id is None:
        return []
    if include_inherited:
        return _with_message_metadata(db, repository.get_thread(db, branch.head_id), request)
    return _with_message_metadata(db, repository.get_branch_messages(db, branch_id), request)


@router.get(
    "/branches/{branch_id}/thread",
    response_model=list[MessageOut],
    summary="브랜치 대화 조회 (구버전, /messages로 대체)",
    deprecated=True,
)
def branch_thread(branch_id: str, request: Request, db: Session = Depends(get_db)):
    """이전 버전 호환을 위해 남겨둔 브랜치 대화 조회 API입니다.

    - 현재 권장 API는 `GET /branches/{branch_id}/messages`입니다.
    - 이 API는 항상 부모 브랜치의 분기 이전 맥락까지 포함합니다.
    - 동작은 `GET /branches/{branch_id}/messages?include_inherited=true`와 같습니다.
    - 새 기능에서는 `/messages`를 사용하세요.

    **왜 구버전인가요?**

    처음에는 화면에 보여줄 "현재 브랜치의 전체 대화 줄기"만 필요해서 `/thread`를 만들었습니다.
    이후 사용자가 "현재 브랜치에서 직접 작성한 메시지만 보기"와
    "부모 브랜치에서 물려받은 맥락까지 같이 보기"를 선택할 필요가 생겨
    `/messages`에 `include_inherited` 옵션을 추가했습니다.

    **차이점**

    - `/thread`: 상속 맥락 포함만 가능
    - `/messages?include_inherited=true`: `/thread`와 동일
    - `/messages?include_inherited=false`: 현재 브랜치에서 작성한 메시지만 반환
    """
    branch = repository.get_branch(db, branch_id)
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다")
    if branch.head_id is None:
        return []
    return _with_message_metadata(db, repository.get_thread(db, branch.head_id), request)
