import uuid
from datetime import datetime, timezone

from app.models.models import Message, Branch, Conversation, Tag, BranchTag, BranchMergeParent, MessageTag, Embedding, UploadedFile, Persona


# ── Message ───────────────────────────────────────────────────────────────────

def save_message(db, session_id, branch_id, role, content,
                 parent_id=None, model_provider=None, model_name=None,
                 input_tokens=None, output_tokens=None):
    msg = Message(
        session_id=session_id,
        branch_id=branch_id,
        role=role,
        content=content,
        parent_id=parent_id,
        model_provider=model_provider,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    db.add(msg)
    db.flush()
    return msg



def get_thread(db, leaf_id):
    """leaf_id에서 root까지 parent_id를 역추적해 시간순으로 반환한다."""
    chain = []
    msg = db.query(Message).filter(Message.id == leaf_id).first()
    while msg is not None:
        chain.append(msg)
        if msg.parent_id is None:
            break
        msg = db.query(Message).filter(Message.id == msg.parent_id).first()
    chain.reverse()
    return chain


def get_message(db, message_id):
    return db.query(Message).filter(Message.id == message_id).first()


def get_branch_messages(db, branch_id: str) -> list:
    """branch의 active 메시지 전체를 시간순으로 반환한다."""
    return (
        db.query(Message)
        .filter(Message.branch_id == branch_id, Message.status == "active")
        .order_by(Message.id)
        .all()
    )


def get_messages_until(db, branch_id: str, until_message_id: int) -> list:
    """branch의 메시지 중 until_message_id까지(포함) 시간순으로 반환한다."""
    return (
        db.query(Message)
        .filter(
            Message.branch_id == branch_id,
            Message.id <= until_message_id,
            Message.status == "active",
        )
        .order_by(Message.id)
        .all()
    )


def get_messages_after(db, branch_id: str, after_message_id: int) -> list:
    """branch의 메시지 중 after_message_id 이후(제외)만 시간순으로 반환한다."""
    return (
        db.query(Message)
        .filter(
            Message.branch_id == branch_id,
            Message.id > after_message_id,
            Message.status == "active",
        )
        .order_by(Message.id)
        .all()
    )


# ── Conversation (Session) ────────────────────────────────────────────────────

def list_conversations(db):
    return (
        db.query(Conversation)
        .filter(Conversation.status == "active")
        .order_by(Conversation.created_at.desc())
        .all()
    )


def get_conversation(db, session_id: str):
    return db.query(Conversation).filter(Conversation.id == session_id).first()


def list_conversation_summaries(db) -> list[dict]:
    rows = (
        db.query(Conversation, Branch.id.label("main_branch_id"))
        .outerjoin(
            Branch,
            (Branch.session_id == Conversation.id)
            & Branch.parent_branch_id.is_(None)
            & Branch.is_merge.isnot(True),
        )
        .filter(Conversation.status == "active")
        .order_by(Conversation.created_at.desc())
        .all()
    )
    return [
        {
            "id": conv.id,
            "title": conv.title,
            "status": conv.status,
            "deleted_at": conv.deleted_at,
            "created_at": conv.created_at,
            "main_branch_id": main_branch_id,
        }
        for conv, main_branch_id in rows
    ]


def _now_str() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def soft_delete_session(db, session_id: str):
    """세션을 휴지통으로 이동한다. 브랜치 상태는 건드리지 않는다."""
    conv = get_conversation(db, session_id)
    if conv is None:
        return None
    conv.status = "deleted"
    conv.deleted_at = _now_str()
    db.commit()
    return conv


def list_deleted_sessions(db):
    return (
        db.query(Conversation)
        .filter(Conversation.status == "deleted")
        .order_by(Conversation.deleted_at.desc())
        .all()
    )


def restore_session(db, session_id: str):
    conv = get_conversation(db, session_id)
    if conv is None or conv.status != "deleted":
        return None
    conv.status = "active"
    conv.deleted_at = None
    db.commit()
    return conv


def _hard_delete_branch_data(db, branch_id: str):
    """단일 브랜치에 딸린 메시지/임베딩/태그연결/비교세션/머지관계와 브랜치 자체를 완전히 삭제한다."""
    from app.models.models import Embedding, MessageTag, ComparisonSession

    message_ids = [row[0] for row in db.query(Message.id).filter(Message.branch_id == branch_id).all()]
    if message_ids:
        db.query(Embedding).filter(Embedding.message_id.in_(message_ids)).delete(synchronize_session=False)
        db.query(MessageTag).filter(MessageTag.message_id.in_(message_ids)).delete(synchronize_session=False)
        db.query(Message).filter(Message.branch_id == branch_id).delete(synchronize_session=False)
    db.query(BranchTag).filter(BranchTag.branch_id == branch_id).delete(synchronize_session=False)
    db.query(ComparisonSession).filter(ComparisonSession.branch_id == branch_id).delete(synchronize_session=False)
    db.query(BranchMergeParent).filter(
        (BranchMergeParent.branch_id == branch_id) | (BranchMergeParent.parent_branch_id == branch_id)
    ).delete(synchronize_session=False)
    db.query(Branch).filter(Branch.id == branch_id).delete(synchronize_session=False)


def purge_session(db, session_id: str) -> bool:
    """세션과 그 안의 모든 브랜치/메시지/태그를 완전히 삭제한다. 복원 불가."""
    conv = get_conversation(db, session_id)
    if conv is None:
        return False

    branch_ids = [row[0] for row in db.query(Branch.id).filter(Branch.session_id == session_id).all()]
    for bid in branch_ids:
        _hard_delete_branch_data(db, bid)

    db.query(Tag).filter(Tag.session_id == session_id).delete(synchronize_session=False)
    db.query(Conversation).filter(Conversation.id == session_id).delete(synchronize_session=False)
    db.commit()
    return True


def create_conversation(db, title="새 대화"):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    conv = Conversation(
        id=str(uuid.uuid4()),
        title=title,
        status="active",
        created_at=now,
        updated_at=now,
    )
    db.add(conv)
    db.flush()
    main = Branch(
        id=str(uuid.uuid4()),
        session_id=conv.id,
        name="main",
        parent_branch_id=None,
        fork_from_message_id=None,
        head_id=None,
        status="active",
        is_collapsed=False,
        is_merge=False,
        is_main=True,
        created_at=now,
        updated_at=now,
    )
    db.add(main)
    db.commit()
    return conv, main


# ── Persona ───────────────────────────────────────────────────────────────────

def get_persona(db, persona_id: str) -> Persona | None:
    return db.query(Persona).filter(Persona.id == persona_id).first()


def get_persona_by_slug(db, persona_slug: str) -> Persona | None:
    return db.query(Persona).filter(Persona.slug == persona_slug).first()


def list_personas(db) -> list[Persona]:
    return db.query(Persona).order_by(Persona.created_at).all()


def seed_personas(db, personas: list[dict]) -> None:
    """slug 기준으로 페르소나를 upsert한다. 코드에 정의된 프롬프트를 배포마다 반영한다."""
    for data in personas:
        existing = db.query(Persona).filter(Persona.slug == data["slug"]).first()
        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            db.add(Persona(id=str(uuid.uuid4()), **data))
    db.commit()


# ── Branch ────────────────────────────────────────────────────────────────────

def get_branch(db, branch_id):
    return db.query(Branch).filter(Branch.id == branch_id).first()


def list_branches(db, session_id):
    return db.query(Branch).filter(Branch.session_id == session_id).all()


def get_session_memory(db, session_id: str) -> str | None:
    conv = db.query(Conversation).filter(Conversation.id == session_id).first()
    return conv.memory if conv else None


def update_session_memory(db, session_id: str, memory: str) -> Conversation | None:
    conv = db.query(Conversation).filter(Conversation.id == session_id).first()
    if conv:
        conv.memory = memory
        db.commit()
    return conv


def update_session_title(db, session_id: str, title: str) -> Conversation | None:
    conv = db.query(Conversation).filter(Conversation.id == session_id).first()
    if conv:
        conv.title = title
        db.commit()
    return conv


def update_branch_name(db, branch_id: str, name: str) -> Branch | None:
    branch = get_branch(db, branch_id)
    if branch:
        branch.name = name
        db.commit()
    return branch


def update_branch_summary(db, branch_id: str, summary: str) -> Branch | None:
    branch = get_branch(db, branch_id)
    if branch:
        branch.summary = summary
        db.commit()
    return branch


def update_branch_description(db, branch_id: str, description: str) -> Branch | None:
    branch = get_branch(db, branch_id)
    if branch:
        branch.description = description
        db.commit()
    return branch


def update_branch_persona(db, branch_id: str, persona_id: str | None) -> Branch | None:
    branch = get_branch(db, branch_id)
    if branch:
        branch.persona_id = persona_id
        db.commit()
    return branch


def update_branch_persona_cascade(db, branch_id: str, persona_id: str | None) -> Branch | None:
    """branch_id와 그 모든 하위(자손) 브랜치의 persona_id를 함께 지정/해제한다.

    상위 브랜치의 persona_id는 건드리지 않는다 (해제 시 상위 파트너는 유지됨).
    """
    branch = get_branch(db, branch_id)
    if branch is None:
        return None
    descendant_ids = get_branch_descendant_ids(db, branch_id)
    db.query(Branch).filter(Branch.id.in_({branch_id} | descendant_ids)).update(
        {"persona_id": persona_id}, synchronize_session=False
    )
    db.commit()
    db.refresh(branch)
    return branch


def update_branch_live_summary(db, branch_id: str, live_summary: str, cutoff_id: int) -> Branch | None:
    branch = get_branch(db, branch_id)
    if branch:
        branch.live_summary = live_summary
        branch.live_summary_cutoff_id = cutoff_id
        db.commit()
    return branch


def update_branch_status(db, branch_id: str, status: str | None, is_collapsed: bool | None) -> Branch | None:
    branch = get_branch(db, branch_id)
    if branch is None:
        return None
    if branch.parent_branch_id is None and not branch.is_merge and status in ("inactive", "deleted"):
        raise ValueError("root branch는 비활성화하거나 삭제할 수 없습니다")

    if status == "deleted" and branch.status != "deleted":
        now = _now_str()
        descendant_ids = get_branch_descendant_ids(db, branch_id)
        for bid in {branch_id} | descendant_ids:
            b = get_branch(db, bid)
            b.status = "deleted"
            b.deleted_at = now
    elif status is not None:
        if branch.status == "deleted" and status != "deleted":
            branch.deleted_at = None
        branch.status = status
    if is_collapsed is not None:
        branch.is_collapsed = is_collapsed
    db.commit()
    return branch


def restore_branch_cascade(db, branch_id: str) -> Branch | None:
    """브랜치와 그 하위 브랜치를 함께 복원한다."""
    branch = get_branch(db, branch_id)
    if branch is None or branch.status != "deleted":
        return None
    descendant_ids = get_branch_descendant_ids(db, branch_id)
    for bid in {branch_id} | descendant_ids:
        b = get_branch(db, bid)
        if b is not None and b.status == "deleted":
            b.status = "active"
            b.deleted_at = None
    db.commit()
    return branch


def purge_branch_cascade(db, branch_id: str) -> bool:
    """브랜치와 그 하위 브랜치를 완전히 삭제한다. 복원 불가."""
    branch = get_branch(db, branch_id)
    if branch is None:
        return False
    descendant_ids = get_branch_descendant_ids(db, branch_id)
    for bid in descendant_ids | {branch_id}:
        _hard_delete_branch_data(db, bid)
    db.commit()
    return True


def get_message_count_by_branch(db, session_id: str) -> dict:
    """session 내 브랜치별 active 메시지 수를 {branch_id: count} 형태로 반환한다."""
    from sqlalchemy import func
    rows = (
        db.query(Message.branch_id, func.count(Message.id))
        .filter(Message.session_id == session_id, Message.status == "active")
        .group_by(Message.branch_id)
        .all()
    )
    return {branch_id: count for branch_id, count in rows}


def create_branch(db, session_id, parent_branch_id, fork_from_message_id, name="새 가지", persona_id=None):
    """새 브랜치를 생성한다.

    persona_id를 명시하지 않으면(None) 부모 브랜치의 persona_id를 그대로 물려받는다
    (fork로 만든 자식 브랜치는 '도움 받기' 파트너를 상속함). 명시적으로 값을 넘기면
    (다른 파트너로 지정하거나 빈 문자열이 아닌 값) 그 값을 그대로 사용한다.
    """
    parent = get_branch(db, parent_branch_id)
    if parent is None or parent.session_id != session_id:
        raise ValueError("parent_branch_id가 해당 session에 속하지 않습니다")

    fork_msg = db.query(Message).filter(
        Message.id == fork_from_message_id,
        Message.branch_id == parent_branch_id,
    ).first()
    if fork_msg is None:
        raise ValueError("fork_from_message_id가 parent branch에 속하지 않습니다")

    effective_persona_id = persona_id if persona_id is not None else parent.persona_id

    branch = Branch(
        id=str(uuid.uuid4()),
        session_id=session_id,
        parent_branch_id=parent_branch_id,
        fork_from_message_id=fork_from_message_id,
        name=name,
        head_id=fork_from_message_id,
        status="active",
        is_collapsed=False,
        persona_id=effective_persona_id,
    )
    db.add(branch)
    db.commit()
    return branch


def create_merge_branch(db, session_id: str, parent_summaries: dict[str, str], name="병합 브랜치"):
    """여러 브랜치를 부모로 갖는 머지 브랜치를 생성한다.

    parent_summaries: {parent_branch_id: summary} — 각 부모 브랜치 전체 맥락의 요약.
    머지 브랜치는 단일 fork 지점이 없으므로 parent_branch_id / fork_from_message_id는 비워둔다.
    각 요약은 (user, assistant) 메시지 쌍으로도 저장해서, 채팅창을 열면 합쳐지는 브랜치들의
    요약이 대화 맨 앞에 바로 보이고 LLM context에도 자연스럽게 포함되게 한다.
    """
    parents = {pid: get_branch(db, pid) for pid in parent_summaries}
    if any(p is None or p.session_id != session_id for p in parents.values()):
        raise ValueError("parent_branch_ids가 해당 session에 속하지 않습니다")

    branch = Branch(
        id=str(uuid.uuid4()),
        session_id=session_id,
        parent_branch_id=None,
        fork_from_message_id=None,
        name=name,
        head_id=None,
        status="active",
        is_collapsed=False,
        is_merge=True,
    )
    db.add(branch)
    db.flush()

    last_message_id = None
    for parent_id, summary in parent_summaries.items():
        db.add(BranchMergeParent(branch_id=branch.id, parent_branch_id=parent_id, summary=summary))

        intro = save_message(
            db, session_id=session_id, branch_id=branch.id, role="user",
            content=f"[브랜치 '{parents[parent_id].name}' 요약]\n{summary}", parent_id=last_message_id,
        )
        ack = save_message(
            db, session_id=session_id, branch_id=branch.id, role="assistant",
            content="확인했습니다.", parent_id=intro.id,
        )
        last_message_id = ack.id

    branch.head_id = last_message_id
    db.commit()
    return branch


def list_branch_trash(db, session_id: str) -> list:
    return (
        db.query(Branch)
        .filter(
            Branch.session_id == session_id,
            Branch.status == "deleted",
            Branch.deleted_at.isnot(None),
        )
        .order_by(Branch.deleted_at.desc())
        .all()
    )


def count_branch_chat_messages(db, branch_id: str) -> int:
    """머지 시 자동 삽입된 요약 메시지를 제외하고, 실제 채팅으로 만들어진 메시지 수를 센다.

    채팅으로 생성된 메시지는 항상 model_provider가 채워져 있고, 머지 요약 메시지는 비어 있다.
    """
    return (
        db.query(Message)
        .filter(
            Message.branch_id == branch_id,
            Message.status == "active",
            Message.model_provider.isnot(None),
        )
        .count()
    )


def get_branch_merge_parents(db, branch_id: str) -> list[BranchMergeParent]:
    return (
        db.query(BranchMergeParent)
        .filter(BranchMergeParent.branch_id == branch_id)
        .all()
    )


def get_merge_parent_ids(db, branch_id: str) -> list[str]:
    return [mp.parent_branch_id for mp in get_branch_merge_parents(db, branch_id)]


def list_merge_edges(db, session_id: str) -> list[BranchMergeParent]:
    """session 내 모든 머지 관계(branch_id, parent_branch_id, summary)를 반환한다."""
    return (
        db.query(BranchMergeParent)
        .join(Branch, Branch.id == BranchMergeParent.branch_id)
        .filter(Branch.session_id == session_id)
        .all()
    )


def select_main_branch(db, branch_id: str) -> list[str]:
    """branch_id부터 루트까지의 부모 체인 전체를 main 경로(is_main=True)로 표시한다.

    같은 세션의 다른 브랜치는 모두 is_main=False로 초기화한 뒤, 선택한 체인에만
    is_main=True를 설정한다. 머지 브랜치는 parent_branch_id가 없어 그 지점에서
    체인이 멈춘다 (여러 부모 중 하나를 대표로 고를 근거가 없기 때문).
    """
    branch = get_branch(db, branch_id)
    if branch is None:
        raise ValueError("branch를 찾을 수 없습니다")

    chain = []
    current = branch
    while current is not None:
        chain.append(current)
        if current.parent_branch_id is None:
            break
        current = get_branch(db, current.parent_branch_id)
    chain.reverse()
    chain_ids = [b.id for b in chain]

    db.query(Branch).filter(Branch.session_id == branch.session_id).update(
        {"is_main": False}, synchronize_session=False
    )
    db.query(Branch).filter(Branch.id.in_(chain_ids)).update(
        {"is_main": True}, synchronize_session=False
    )
    db.commit()
    return chain_ids


# ── Tag ───────────────────────────────────────────────────────────────────────

def create_tag(db, session_id: str, name: str, color: str | None, type: str) -> Tag:
    tag = Tag(id=str(uuid.uuid4()), session_id=session_id, name=name, color=color, type=type)
    db.add(tag)
    db.commit()
    return tag


def get_tag(db, tag_id: str) -> Tag | None:
    return db.query(Tag).filter(Tag.id == tag_id).first()


def get_session_tags(db, session_id: str) -> list:
    return db.query(Tag).filter(Tag.session_id == session_id).order_by(Tag.created_at).all()


def get_branch_tags(db, branch_id: str) -> list:
    """브랜치에 연결된 태그 목록을 반환한다."""
    return (
        db.query(Tag)
        .join(BranchTag, Tag.id == BranchTag.tag_id)
        .filter(BranchTag.branch_id == branch_id)
        .order_by(Tag.created_at)
        .all()
    )



def search_branches(db, session_id: str, q: str) -> list[dict]:
    """브랜치 이름 또는 태그 이름에 q가 포함된 브랜치를 반환한다."""
    like = f"%{q}%"

    by_name = (
        db.query(Branch.id)
        .filter(Branch.session_id == session_id, Branch.status != "deleted", Branch.name.ilike(like))
    )
    by_tag = (
        db.query(BranchTag.branch_id)
        .join(Tag, BranchTag.tag_id == Tag.id)
        .filter(Tag.session_id == session_id, Tag.name.ilike(like))
    )

    matched_ids = {row[0] for row in by_name} | {row[0] for row in by_tag}
    if not matched_ids:
        return []

    branches = db.query(Branch).filter(Branch.id.in_(matched_ids)).all()

    results = []
    for branch in branches:
        tag_names = [
            t.name for t in
            db.query(Tag).join(BranchTag, Tag.id == BranchTag.tag_id)
            .filter(BranchTag.branch_id == branch.id).all()
        ]
        results.append({"id": branch.id, "name": branch.name, "status": branch.status, "tags": tag_names})
    return results


def remove_branch_tag(db, branch_id: str, tag_id: str) -> bool:
    bt = db.query(BranchTag).filter(
        BranchTag.branch_id == branch_id,
        BranchTag.tag_id == tag_id,
    ).first()
    if bt is None:
        return False
    db.delete(bt)
    db.commit()
    return True


def add_branch_tag(db, branch_id: str, tag_id: str) -> BranchTag:
    existing = db.query(BranchTag).filter(
        BranchTag.branch_id == branch_id,
        BranchTag.tag_id == tag_id,
    ).first()
    if existing:
        return existing
    bt = BranchTag(branch_id=branch_id, tag_id=tag_id)
    db.add(bt)
    db.commit()
    return bt


# ── Embedding ─────────────────────────────────────────────────────────────────

def save_embedding(db, message_id: int, vector: list, model: str):
    import json
    import uuid as _uuid
    from app.models.models import Embedding
    existing = db.query(Embedding).filter(Embedding.message_id == message_id).first()
    if existing:
        return existing
    emb = Embedding(
        id=str(_uuid.uuid4()),
        message_id=message_id,
        embedding=json.dumps(vector),
        embedding_model=model,
    )
    db.add(emb)
    db.commit()
    return emb


# ── Comparison ────────────────────────────────────────────────────────────────

def create_comparison(db, branch_id, user_message, response_a_content, response_a_provider,
                      response_a_model, response_b_content, response_b_provider, response_b_model):
    from app.models.models import ComparisonSession
    comp = ComparisonSession(
        id=str(uuid.uuid4()),
        branch_id=branch_id,
        user_message=user_message,
        response_a_content=response_a_content,
        response_a_provider=response_a_provider,
        response_a_model=response_a_model,
        response_b_content=response_b_content,
        response_b_provider=response_b_provider,
        response_b_model=response_b_model,
        status="pending",
    )
    db.add(comp)
    db.commit()
    return comp


def get_comparison(db, comparison_id: str):
    from app.models.models import ComparisonSession
    return db.query(ComparisonSession).filter(ComparisonSession.id == comparison_id).first()


def mark_comparison_done(db, comparison_id: str):
    from app.models.models import ComparisonSession
    comp = db.query(ComparisonSession).filter(ComparisonSession.id == comparison_id).first()
    if comp:
        comp.status = "done"


def get_session_embeddings(db, session_id: str, exclude_branch_ids: list[str]):
    from app.models.models import Embedding
    query = (
        db.query(Embedding, Message)
        .join(Message, Embedding.message_id == Message.id)
        .filter(Message.session_id == session_id, Message.status == "active")
    )
    if exclude_branch_ids:
        query = query.filter(Message.branch_id.notin_(exclude_branch_ids))
    return query.all()


def get_branch_embeddings(db, branch_id: str) -> list:
    """특정 브랜치에 속한 메시지들의 임베딩을 반환한다."""
    from app.models.models import Embedding
    return (
        db.query(Embedding)
        .join(Message, Embedding.message_id == Message.id)
        .filter(Message.branch_id == branch_id, Message.status == "active")
        .all()
    )


def get_branch_descendant_ids(db, branch_id: str) -> set[str]:
    """branch_id의 모든 자손 브랜치 id를 반환한다 (직계 자식부터 재귀적으로)."""
    descendants: set[str] = set()
    frontier = [branch_id]
    while frontier:
        rows = db.query(Branch.id).filter(Branch.parent_branch_id.in_(frontier)).all()
        child_ids = [r[0] for r in rows if r[0] not in descendants]
        descendants.update(child_ids)
        frontier = child_ids
    return descendants


def get_session_branches(db, session_id: str, exclude_ids: set[str]) -> list:
    """세션에 속한 브랜치 중 exclude_ids를 제외한 삭제되지 않은 브랜치를 반환한다."""
    query = db.query(Branch).filter(Branch.session_id == session_id, Branch.status != "deleted")
    if exclude_ids:
        query = query.filter(~Branch.id.in_(exclude_ids))
    return query.all()


# ── UploadedFile ──────────────────────────────────────────────────────────────

def save_file(db, session_id: str, filename: str, extracted_text: str,
              summary: str | None = None, branch_id: str | None = None,
              file_type: str = "text", mime_type: str | None = None,
              content_data: str | None = None,
              image_data: str | None = None) -> UploadedFile:
    f = UploadedFile(
        id=str(uuid.uuid4()),
        session_id=session_id,
        branch_id=branch_id,
        filename=filename,
        extracted_text=extracted_text,
        summary=summary,
        file_type=file_type,
        mime_type=mime_type,
        content_data=content_data,
        image_data=image_data,
    )
    db.add(f)
    db.commit()
    return f


def get_file(db, file_id: str) -> UploadedFile | None:
    return db.query(UploadedFile).filter(UploadedFile.id == file_id).first()


def get_branch_files(db, branch_id: str) -> list[UploadedFile]:
    """해당 브랜치 전용 파일 목록을 반환한다."""
    return (
        db.query(UploadedFile)
        .filter(UploadedFile.branch_id == branch_id)
        .order_by(UploadedFile.created_at)
        .all()
    )


def get_session_files(db, session_id: str) -> list[UploadedFile]:
    """세션 전체 공유 파일 목록을 반환한다 (branch_id가 None인 것)."""
    return (
        db.query(UploadedFile)
        .filter(UploadedFile.session_id == session_id, UploadedFile.branch_id.is_(None))
        .order_by(UploadedFile.created_at)
        .all()
    )


def get_message_files(db, message_id: int) -> list[UploadedFile]:
    return (
        db.query(UploadedFile)
        .filter(UploadedFile.message_id == message_id)
        .order_by(UploadedFile.created_at)
        .all()
    )


def get_files_by_message_ids(db, message_ids: list[int]) -> dict[int, list[UploadedFile]]:
    if not message_ids:
        return {}

    rows = (
        db.query(UploadedFile)
        .filter(UploadedFile.message_id.in_(message_ids))
        .order_by(UploadedFile.created_at)
        .all()
    )
    files_by_message_id: dict[int, list[UploadedFile]] = {}

    for file in rows:
        if file.message_id is None:
            continue
        files_by_message_id.setdefault(file.message_id, []).append(file)

    return files_by_message_id


def validate_message_file_ids(
    db,
    file_ids: list[str],
    session_id: str,
    branch_id: str,
) -> list[UploadedFile]:
    unique_file_ids = list(dict.fromkeys(file_ids))

    if not unique_file_ids:
        return []

    files = db.query(UploadedFile).filter(UploadedFile.id.in_(unique_file_ids)).all()
    file_by_id = {file.id: file for file in files}
    invalid_file_ids = [
        file_id
        for file_id in unique_file_ids
        if file_id not in file_by_id
        or file_by_id[file_id].session_id != session_id
        or file_by_id[file_id].branch_id != branch_id
    ]

    if invalid_file_ids:
        raise ValueError("첨부 파일이 현재 브랜치에 속하지 않거나 존재하지 않습니다.")

    return [file_by_id[file_id] for file_id in unique_file_ids]


def attach_files_to_message(
    db,
    file_ids: list[str],
    message_id: int,
    session_id: str,
    branch_id: str,
) -> list[UploadedFile]:
    files = validate_message_file_ids(db, file_ids, session_id, branch_id)

    for file in files:
        file.message_id = message_id

    db.flush()
    return files


def delete_file(db, file_id: str) -> bool:
    f = db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
    if f is None:
        return False
    db.delete(f)
    db.commit()
    return True
