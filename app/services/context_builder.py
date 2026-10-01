from app.repositories import repository
from app.services import token_budget

OWN_BRANCH_KEEP_RECENT_MESSAGES = 6


def build_context(db, branch_id: str, new_user_message: str, model_name: str) -> list[dict]:
    """LLM에 전달할 messages 배열을 구성한다.

    1. 세션 메모리(사용자 정보)를 context 맨 앞에 주입한다.
    2. 조상 체인 중 직계 부모는 fork_from_message_id 이전 메시지까지 원문 그대로 포함하고,
       그보다 이전 조상들은 조부모의 누적 요약(Branch.summary) 하나로 대체한다 (토큰 절약).
       머지 브랜치는 부모 브랜치 요약이 (user, assistant) 메시지 쌍으로 이미 본인 메시지에
       저장되어 있으므로 그대로 포함된다.
    3. 현재 브랜치 자신의 메시지도 너무 길면(live_summary_cutoff_id) 오래된 부분은
       압축된 live_summary로, 최근 부분만 원문으로 포함한다.
    4. 다른 브랜치에서 유사 메시지를 벡터 검색해 참고 context로 추가한다.
    5. 조립한 전체가 모델의 토큰 임계치에 가까우면, 현재 브랜치 자신의 오래된 메시지를
       그 자리에서 압축하고 다시 조립한다 (이번 요청에도 즉시 반영).
    6. 마지막에 새 사용자 메시지를 추가해 반환한다.
    """
    from app.services import embedding_service, auto_tagger

    branch = repository.get_branch(db, branch_id)
    memory = repository.get_session_memory(db, branch.session_id)

    if branch.is_merge:
        ancestor_ids = [branch_id] + repository.get_merge_parent_ids(db, branch_id)
        ancestor_blocks = []
    else:
        branch_chain = get_branch_ancestor_chain(db, branch_id)
        ancestor_ids = [b.id for b in branch_chain]
        ancestor_blocks = _build_ancestor_blocks(db, branch_chain)

    own_blocks = _build_own_branch_blocks(db, branch)

    # 다른 브랜치에서 유사 메시지 검색
    try:
        similar = embedding_service.search_similar_messages(
            db, branch.session_id, new_user_message, exclude_branch_ids=ancestor_ids
        )
    except Exception:
        similar = []

    session_files = repository.get_session_files(db, branch.session_id)
    branch_files = repository.get_branch_files(db, branch_id)
    all_files = session_files + branch_files

    base: list[dict] = []

    if memory:
        base.append({"role": "user", "content": f"[사용자 정보]\n{memory}"})
        base.append({"role": "assistant", "content": "알겠습니다. 해당 정보를 기억하겠습니다."})

    text_files = [f for f in all_files if f.file_type != "image"]
    image_files = [f for f in all_files if f.file_type == "image"]

    if text_files:
        # 요약만 주입. 전체 텍스트는 llm_service의 tool handler가 필요 시 제공한다.
        lines = []
        for f in text_files:
            scope = "세션 공유" if f.branch_id is None else "브랜치"
            desc = f.summary or "(요약 없음)"
            lines.append(f"- [{scope}] {f.filename}: {desc}")
        base.append({"role": "user", "content": "[첨부 파일 목록]\n" + "\n".join(lines)})
        base.append({"role": "assistant", "content": "파일 목록을 확인했습니다. 전체 내용이 필요하면 도구를 사용하겠습니다."})

    if image_files:
        # 매 요청마다 이미지 원본을 다시 첨부해 모델이 항상 직접 볼 수 있게 한다.
        content: list[dict] = [{"type": "text", "text": "[첨부 이미지]"}]
        for f in image_files:
            content.append({"type": "text", "text": f.filename})
            content.append({"type": "image", "media_type": f.mime_type, "data": f.image_data})
        base.append({"role": "user", "content": content})
        base.append({"role": "assistant", "content": "첨부된 이미지를 확인했습니다."})

    if similar:
        refs = "\n".join([f"- {m.content}" for m in similar])
        base.append({"role": "user", "content": f"[다른 대화에서 관련 내용]\n{refs}"})
        base.append({"role": "assistant", "content": "참고하겠습니다."})

    base.extend(ancestor_blocks)

    result = base + own_blocks + [{"role": "user", "content": new_user_message}]

    if token_budget.count_context_tokens(result, model_name) >= token_budget.get_compression_threshold(model_name):
        auto_tagger.compact_branch_live_messages(db, branch_id, keep_recent=OWN_BRANCH_KEEP_RECENT_MESSAGES)
        branch = repository.get_branch(db, branch_id)
        own_blocks = _build_own_branch_blocks(db, branch)
        result = base + own_blocks + [{"role": "user", "content": new_user_message}]

    return result


def _build_ancestor_blocks(db, branch_chain: list) -> list[dict]:
    """조상 체인(root..current) 중 직계 부모 이전은 조부모 누적 요약으로, 직계 부모는 원문으로 반환한다."""
    if len(branch_chain) < 2:
        return []

    blocks: list[dict] = []
    parent = branch_chain[-2]
    child = branch_chain[-1]

    if len(branch_chain) >= 3:
        grandparent = branch_chain[-3]
        if grandparent.summary:
            blocks.append({"role": "user", "content": f"[이전 대화 누적 요약]\n{grandparent.summary}"})
            blocks.append({"role": "assistant", "content": "이전 대화 맥락을 확인했습니다."})
        else:
            # 요약이 아직 없으면(과거 데이터 등) 안전하게 원문으로 폴백한다.
            for i, b in enumerate(branch_chain[:-2]):
                nxt = branch_chain[i + 1]
                for m in repository.get_messages_until(db, b.id, nxt.fork_from_message_id):
                    blocks.append({"role": m.role, "content": m.content})

    for m in repository.get_messages_until(db, parent.id, child.fork_from_message_id):
        blocks.append({"role": m.role, "content": m.content})

    return blocks


def _build_own_branch_blocks(db, branch) -> list[dict]:
    """현재 브랜치 자신의 메시지를 반환한다. live_summary_cutoff_id가 있으면 그 이전은 요약으로 대체한다."""
    blocks: list[dict] = []
    if branch.live_summary_cutoff_id:
        if branch.live_summary:
            blocks.append({"role": "user", "content": f"[이 대화의 이전 내용 요약]\n{branch.live_summary}"})
            blocks.append({"role": "assistant", "content": "확인했습니다."})
        messages = repository.get_messages_after(db, branch.id, branch.live_summary_cutoff_id)
    else:
        messages = repository.get_branch_messages(db, branch.id)

    messages = fit_to_token_budget(messages)
    blocks.extend({"role": m.role, "content": m.content} for m in messages)
    return blocks


def get_full_branch_messages(db, branch_id: str, branch_chain: list | None = None) -> list:
    """branch의 분기 이전 조상 맥락 + 본인 메시지 전체를 원문 그대로(요약 없이) 시간순으로 반환한다.

    브랜치 트리 머지(여러 부모 요약을 합치는 기능)처럼 원문 전체가 필요한 곳에서 사용한다.
    일반 채팅 컨텍스트 조립에는 build_context를 사용할 것 (조상 요약을 적용함).
    """
    if branch_chain is None:
        branch_chain = get_branch_ancestor_chain(db, branch_id)

    messages = []
    for i, b in enumerate(branch_chain):
        if b.id == branch_id:
            messages.extend(repository.get_branch_messages(db, b.id))
        else:
            child = branch_chain[i + 1]
            messages.extend(repository.get_messages_until(db, b.id, child.fork_from_message_id))
    return messages


def get_branch_ancestor_chain(db, branch_id: str) -> list:
    """현재 branch에서 root까지 역추적해 [root, ..., current] 순으로 반환한다."""
    chain = []
    branch = repository.get_branch(db, branch_id)
    while branch is not None:
        chain.append(branch)
        if branch.parent_branch_id is None:
            break
        branch = repository.get_branch(db, branch.parent_branch_id)
    chain.reverse()
    return chain


def fit_to_token_budget(messages: list, max_messages: int = 20) -> list:
    """컨텍스트가 너무 길면 오래된 메시지부터 제외한다. 실시간 압축이 실패했을 때의 안전망."""
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]
