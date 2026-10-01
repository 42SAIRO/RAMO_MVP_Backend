import json

from openai import OpenAI
from dotenv import load_dotenv

from app.repositories import repository

load_dotenv()
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

ROLE_LABELS = ["번역", "요약", "피드백", "브레인스토밍", "코드작성", "질문답변"]


def generate_session_name(user_message: str, assistant_message: str) -> str:
    """첫 번째 대화를 보고 세션 제목을 생성한다."""
    prompt = (
        "다음 대화를 보고 대화 제목을 15글자 이내로 지어줘. "
        "제목만 답해줘. 다른 말은 하지 마.\n\n"
        f"사용자: {user_message}\nAI: {assistant_message}\n\n제목:"
    )
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
    )
    return response.choices[0].message.content.strip()


def generate_branch_name_from_qa(user_message: str, answer: str) -> str:
    """분기 후 해당 브랜치에서 처음 나눈 질문/답변을 보고 브랜치 이름을 생성한다."""
    prompt = (
        "다음 질문과 답변을 보고 이 대화 브랜치의 이름을 10글자 이내로 지어줘. "
        "이름만 답해줘. 다른 말은 하지 마.\n\n"
        f"질문: {user_message}\n답변: {answer}\n\n이름:"
    )
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
    )
    return response.choices[0].message.content.strip()


def generate_name_from_conversation(messages: list) -> str:
    """브랜치 전체 대화를 읽고 브랜치 이름을 생성한다."""
    conversation = "\n".join([f"{m.role}: {m.content}" for m in messages])
    prompt = (
        "다음 대화의 핵심 주제를 담은 브랜치 이름을 10글자 이내로 지어줘. "
        "이름만 답해줘. 다른 말은 하지 마.\n\n"
        f"{conversation}\n\n이름:"
    )
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
    )
    return response.choices[0].message.content.strip()


def summarize_branch_for_merge(messages: list) -> str:
    """머지 시 다른 브랜치와 합칠 수 있도록 브랜치 전체 맥락을 요약한다.

    머지는 세션당 몇 번 안 일어나는 일회성 이벤트지만, 그 결과가 새로 생기는
    병합 브랜치의 초기 컨텍스트 전체가 되어 이후 대화에 계속 영향을 주므로,
    다른 고빈도 보조 작업(제목/이름/태그 등)과 달리 상위 모델(gpt-4o)을 쓴다.
    """
    if not messages:
        return "(대화 없음)"
    conversation = "\n".join([f"{m.role}: {m.content}" for m in messages])
    prompt = (
        "다음 대화를 다른 브랜치의 대화와 합쳐서 참고할 수 있도록 핵심 내용 요약해줘. "
        "요약만 답해줘. 다른 말은 하지 마.\n\n"
        f"{conversation}\n\n요약:"
    )
    response = _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def auto_name_branch(db, branch_id: str) -> str:
    """브랜치 대화 내용을 분석해 이름을 자동 생성하고 DB에 저장한다."""
    messages = repository.get_branch_messages(db, branch_id)
    if not messages:
        return "새 가지"
    name = generate_name_from_conversation(messages)
    repository.update_branch_name(db, branch_id, name)
    return name


def summarize_branch_cascade(prior_summaries: list[str], messages: list) -> str:
    """이전까지의 누적 요약 위에 새로 오간 대화를 얹어 업데이트된 누적 요약을 만든다.

    - prior_summaries: 이 브랜치 이전까지의 누적 요약 (부모 브랜치의 summary, 이 브랜치의
      기존 live_summary 등). 0~2개. 원문이 아니라 이미 압축된 텍스트만 들어간다.
    - messages: 이번에 새로 반영해야 하는 원문 메시지 (원문이 들어가는 건 이번 한 번뿐).
    - 토큰 절약이 목적이므로, 매번 원문 전체를 다시 요약하지 않고 이 함수를 반복 호출해서
      누적한다.
    - 이 결과(Branch.summary/live_summary)는 이후 모든 자식 브랜치의 채팅 컨텍스트에
      반복 주입되므로, 캐스케이드를 거듭해도 정보 손실이 덜하도록 상위 모델(gpt-4o)을 쓴다.
    """
    priors = [p for p in prior_summaries if p]
    if not messages:
        return "\n\n".join(priors)

    conversation = "\n".join([f"{m.role}: {m.content}" for m in messages])

    anti_meta = (
        "요약에는 대화에서 실제로 오간 정보, 설명, 결론, 수치, 용어를 구체적으로 담아라. "
        "'이런 주제의 대화였다', '~에 대한 설명 흐름이다'처럼 대화 자체를 메타적으로 "
        "묘사하지 말고, 실제 내용을 그대로 요약해라."
    )

    if priors:
        prior_block = "[이전까지의 누적 요약]\n" + "\n\n".join(priors) + "\n\n"
        instructions = (
            "위 [이전까지의 누적 요약]과 [새로 오간 대화]를 반영해서 업데이트된 누적 요약을 작성해줘.\n"
            f"- {anti_meta}\n"
            "- 이전 누적 요약에서 이번 대화로 달라지거나 새로 추가된 내용을 반영해라.\n"
            "- 이전 누적 요약 중 이제 더 이상 필요 없어진 내용이 있다면 빼라 (없으면 그대로 둬라).\n"
            "- 중요한 결정사항이나 핵심 정보는 반드시 포함해라.\n"
            "다른 설명 없이 업데이트된 누적 요약만 10문장 이내로 작성해."
        )
    else:
        prior_block = ""
        instructions = (
            "위 [새로 오간 대화]를 요약해줘.\n"
            f"- {anti_meta}\n"
            "- 중요한 결정사항이나 핵심 정보는 반드시 포함해라.\n"
            "다른 설명 없이 요약만 10문장 이내로 작성해."
        )

    prompt = f"{prior_block}[새로 오간 대화]\n{conversation}\n\n{instructions}"
    response = _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


def auto_summarize_branch(db, branch_id: str) -> str | None:
    """브랜치의 누적 요약(Branch.summary)을 자동 생성/갱신하고 DB에 저장한다.

    - 부모 브랜치의 summary가 있으면 그 위에 이 브랜치의 내용을 얹어서 누적한다.
    - 이 브랜치 자신이 live_summary_cutoff_id로 이미 부분 압축되어 있으면, 압축된
      부분은 재사용하고 그 이후 메시지만 새로 반영한다 (원문 전체를 다시 요약하지 않음).
    - 메시지도 없고 prior 요약도 없으면 아무것도 하지 않고 None을 반환한다.
    """
    branch = repository.get_branch(db, branch_id)
    if branch is None:
        return None

    priors = []
    if branch.parent_branch_id:
        parent = repository.get_branch(db, branch.parent_branch_id)
        if parent and parent.summary:
            priors.append(parent.summary)

    if branch.live_summary_cutoff_id:
        priors.append(branch.live_summary)
        messages = repository.get_messages_after(db, branch_id, branch.live_summary_cutoff_id)
    else:
        messages = repository.get_branch_messages(db, branch_id)

    if not messages and not priors:
        return None

    summary = summarize_branch_cascade(priors, messages)
    repository.update_branch_summary(db, branch_id, summary)
    return summary


DESCRIPTION_TARGET_CHARS = 90   # 프롬프트가 지향하는 목표 길이 (대부분 이 안에 들어와야 함)
DESCRIPTION_HARD_MAX_CHARS = 100  # 코드에서 무조건 보장하는 절대 상한


def _clamp_description(description: str, limit: int = DESCRIPTION_HARD_MAX_CHARS) -> str:
    """limit자를 넘으면 문장/어절 경계에서 자연스럽게 잘라 절대 limit자를 넘지 않게 한다. 말줄임표는 붙이지 않는다."""
    if len(description) <= limit:
        return description

    truncated = description[:limit]
    for end_char in ("다.", "요.", ".", "!", "?"):
        idx = truncated.rfind(end_char)
        if idx != -1:
            return truncated[: idx + len(end_char)]

    idx = truncated.rfind(" ")
    if idx > 0:
        return truncated[:idx].rstrip()

    return truncated.rstrip()


def generate_branch_description(messages: list) -> str:
    """브랜치 자신의 대화만 보고 시각화용 한두 문장 설명을 만든다. 조상 브랜치 내용은 쓰지 않는다."""
    conversation = "\n".join([f"{m.role}: {m.content}" for m in messages])
    prompt = (
        "다음은 하나의 대화 브랜치에서 오간 내용이다. 이 브랜치를 그래프 시각화에서 "
        "표시할 짧은 설명을 만들어줘.\n"
        "- 실제로 다룬 주제·결론·핵심 정보(용어, 수치, 방법 등)만 압축해서 담아라. "
        "'이런 주제의 대화였다'처럼 대화 자체를 메타적으로 설명하지 마라.\n"
        "- 대화 원문(질문, 답변, 인사말, '네 알겠습니다' 같은 응답 문구 등)을 그대로 옮기거나 "
        "이어 쓰지 마라. 항상 새로 요약한 문장으로만 답해라.\n"
        "- 어조는 건조한 개조식/명사형으로 끝내라 (예: '~정리', '~분석', '~비교'). "
        "존댓말, 감탄사, 이모지, 느낌표는 쓰지 마라.\n"
        "- 아래 [대화 내용]에 실질적인 주제나 정보 없이 인사말·자기소개뿐이라도, "
        "'대화 내용이 없다'는 식으로 답하지 말고 실제 오간 내용을 그대로 짧게 설명해라 "
        "(예: '인사만 나눈 상태').\n"
        f"- 전체 설명은 반드시 공백 포함 {DESCRIPTION_TARGET_CHARS}자 이내, 최대 2문장으로 써라. "
        "글자 수를 스스로 세어보고 넘으면 문장을 줄여서 다시 써라. 각 문장은 짧게 끊어라.\n"
        "- 다른 브랜치나 이전 맥락은 가정하지 말고 아래 내용만 보고 작성해라.\n"
        "다른 설명 없이 결과 문장만 작성해.\n\n"
        f"[대화 내용]\n{conversation}"
    )
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
    )
    description = response.choices[0].message.content.strip()
    return _clamp_description(description)


def auto_describe_branch(db, branch_id: str) -> str | None:
    """브랜치 자신의 대화만으로 시각화용 짧은 설명을 생성하고 DB에 저장한다.

    메시지가 하나도 없으면 아무것도 하지 않고 None을 반환한다. 메시지가 1개라도
    있으면 (인사말뿐이라도) 항상 실제 내용을 반영한 설명을 생성해 저장한다.
    """
    messages = repository.get_branch_messages(db, branch_id)
    if not messages:
        return None
    description = generate_branch_description(messages)
    repository.update_branch_description(db, branch_id, description)
    return description


def compact_branch_live_messages(db, branch_id: str, keep_recent: int = 6) -> None:
    """브랜치 자신의 오래된 메시지를 압축해 live_summary/live_summary_cutoff_id에 저장한다.

    최근 keep_recent개는 원문으로 남기고, 그 이전 메시지만 압축 대상이 된다. 이미 압축된
    적이 있으면(live_summary_cutoff_id 존재) 그 이후 메시지 중에서만 다시 나눈다.
    압축할 만큼 메시지가 쌓이지 않았으면 아무것도 하지 않는다.
    """
    branch = repository.get_branch(db, branch_id)
    if branch is None:
        return

    if branch.live_summary_cutoff_id:
        own_messages = repository.get_messages_after(db, branch_id, branch.live_summary_cutoff_id)
    else:
        own_messages = repository.get_branch_messages(db, branch_id)

    if len(own_messages) <= keep_recent:
        return

    to_compact = own_messages[:-keep_recent]
    prior = [branch.live_summary] if branch.live_summary else []
    new_summary = summarize_branch_cascade(prior, to_compact)
    repository.update_branch_live_summary(db, branch_id, new_summary, to_compact[-1].id)


def extract_user_memory(messages: list) -> str:
    """대화 전체를 읽고 사용자에 대한 핵심 정보를 추출한다."""
    conversation = "\n".join([f"{m.role}: {m.content}" for m in messages])
    prompt = (
        "다음 대화에서 사용자에 대한 중요한 정보(이름, 직업, 관심사, 선호도 등)를 추출해줘. "
        "없으면 '없음'이라고 답해줘. 3문장 이내로 간결하게 작성해줘. 다른 말은 하지 마.\n\n"
        f"{conversation}\n\n정보:"
    )
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


def extract_branch_topics_and_roles(messages: list) -> dict:
    """브랜치 전체 대화를 한 번에 분석해 내용 키워드와 역할 라벨을 함께 추출한다.

    - keywords: 핵심 주제 2~3개 (자유 주제어)
    - roles: 정해진 라벨셋(ROLE_LABELS)에서 해당하는 역할 0개 이상 (여러 개 섞여 있으면 전부)
    """
    conversation = "\n".join([f"{m.role}: {m.content}" for m in messages])
    labels = ", ".join(ROLE_LABELS)
    prompt = (
        "다음 대화를 분석해서 아래 두 가지를 JSON으로만 답해줘. 다른 말은 하지 마.\n"
        "1. keywords: 대화의 핵심 주제 2~3개 (단어나 짧은 구문)\n"
        f"2. roles: 다음 라벨셋에서 대화에 해당하는 역할을 0개 이상 모두 골라줘: {labels}. "
        "여러 역할이 섞여 있으면 전부 고르고, 라벨셋에 없는 단어는 절대 만들어내지 마. "
        "해당하는 게 없으면 빈 배열로 답해.\n\n"
        f"{conversation}\n\n"
        '형식: {"keywords": ["...", "..."], "roles": ["...", "..."]}'
    )
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"keywords": [], "roles": []}

    keywords = [k.strip() for k in data.get("keywords", []) if isinstance(k, str) and k.strip()][:3]
    # 라벨셋에 없는 값(모델의 환각)은 걸러낸다.
    roles = [r.strip() for r in data.get("roles", []) if isinstance(r, str) and r.strip() in ROLE_LABELS]
    return {"keywords": keywords, "roles": roles}


def auto_tag_branch(db, session_id: str, branch_id: str) -> dict:
    """브랜치 전체 대화를 분석해 내용 태그와 역할 태그를 함께 자동 생성하고 브랜치에 부여한다.

    - 내용 태그는 type=normal, 역할 태그는 type=role로 별도 축에 저장한다.
    - 세션에 같은 이름·같은 type의 태그가 이미 있으면 재사용한다.
    - 메시지가 없으면 빈 결과를 반환한다.
    """
    messages = repository.get_branch_messages(db, branch_id)
    if not messages:
        return {"auto_tags": [], "auto_roles": []}

    extracted = extract_branch_topics_and_roles(messages)

    existing_tags = repository.get_session_tags(db, session_id)
    existing_content = {tag.name: tag for tag in existing_tags if tag.type == "normal"}
    existing_roles = {tag.name: tag for tag in existing_tags if tag.type == "role"}

    assigned_tags = []
    for name in extracted["keywords"]:
        if name not in existing_content:
            existing_content[name] = repository.create_tag(db, session_id, name, color=None, type="normal")
        repository.add_branch_tag(db, branch_id, existing_content[name].id)
        assigned_tags.append(name)

    assigned_roles = []
    for name in extracted["roles"]:
        if name not in existing_roles:
            existing_roles[name] = repository.create_tag(db, session_id, name, color=None, type="role")
        repository.add_branch_tag(db, branch_id, existing_roles[name].id)
        assigned_roles.append(name)

    return {"auto_tags": assigned_tags, "auto_roles": assigned_roles}
