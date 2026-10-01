from app.repositories import repository
from app.services import embedding_service, context_builder

CONTENT_SIMILARITY_THRESHOLD = 0.65


def _topics_and_roles(db, branch_id: str) -> tuple[list[str], set[str]]:
    """브랜치의 콘텐츠 태그(주제 키워드)와 역할 태그를 함께 가져온다.

    콘텐츠 태그는 auto_tagger가 대화 전체를 분석해 뽑아둔 핵심 주제어라,
    브랜치 내용을 한 줄로 보여줄 요약이 없는 지금은 이걸로 대신 쓴다.
    """
    tags = repository.get_branch_tags(db, branch_id)
    topics = [t.name for t in tags if t.type == "normal"]
    roles = {t.name for t in tags if t.type == "role"}
    return topics, roles


def _content_reason_text(my_topics: list[str], other_topics: list[str], shared_topics: list[str], score: float) -> str:
    pct = round(score * 100)
    if shared_topics:
        return f"두 브랜치 모두 '{', '.join(shared_topics)}' 주제를 다루고 있어 유사합니다 (유사도 {pct}%)"
    if my_topics and other_topics:
        return (
            f"현재 브랜치는 '{', '.join(my_topics)}', 추천 브랜치는 '{', '.join(other_topics)}' "
            f"주제를 다루고 있어 대화 내용이 서로 비슷합니다 (유사도 {pct}%)"
        )
    return f"대화 내용이 서로 비슷합니다 (유사도 {pct}%). 아직 태그가 없어 주제를 특정하지 못했습니다."


def get_merge_candidates(db, branch_id: str) -> list[dict]:
    """branch_id와 머지할 만한 다른 브랜치를 추천한다.

    - 직계 조상/자손 브랜치는 이미 부모-자식으로 연결되어 있어 후보에서 제외한다.
    - 내용 유사도(임베딩 centroid 코사인 유사도)가 threshold 이상인 브랜치만 후보가 된다.
      역할 태그 일치는 그 위에 얹는 보조 이유일 뿐, 역할만 겹쳐서는 후보가 되지 않는다.
      (예: "질문답변"처럼 흔한 역할은 그것만으로 관련성을 보장하지 않는다)
    - content 이유에는 점수만이 아니라 양쪽 브랜치의 주제 키워드(콘텐츠 태그)를 함께 담아,
      "왜" 유사한지 구체적으로 설명한다. 태그가 없는 브랜치는 점수만 표시된다
      (아직 auto-tag가 실행되지 않은 브랜치일 수 있음).
    """
    branch = repository.get_branch(db, branch_id)
    if branch is None:
        return []

    ancestor_ids = {b.id for b in context_builder.get_branch_ancestor_chain(db, branch_id)}
    descendant_ids = repository.get_branch_descendant_ids(db, branch_id)
    excluded_ids = ancestor_ids | descendant_ids

    other_branches = repository.get_session_branches(db, branch.session_id, excluded_ids)
    if not other_branches:
        return []

    my_centroid = embedding_service.branch_centroid(db, branch_id)
    if my_centroid is None:
        return []
    my_topics, my_roles = _topics_and_roles(db, branch_id)

    candidates = []
    for other in other_branches:
        other_centroid = embedding_service.branch_centroid(db, other.id)
        if other_centroid is None:
            continue

        score = embedding_service.cosine_similarity(my_centroid, other_centroid)
        if score < CONTENT_SIMILARITY_THRESHOLD:
            continue

        other_topics, other_roles = _topics_and_roles(db, other.id)
        shared_topics = [t for t in my_topics if t in other_topics]

        reasons = [{
            "type": "content",
            "score": round(score, 3),
            "my_topics": my_topics,
            "other_topics": other_topics,
            "shared_topics": shared_topics,
            "text": _content_reason_text(my_topics, other_topics, shared_topics, score),
        }]

        matched_roles = sorted(my_roles & other_roles)
        if matched_roles:
            reasons.append({
                "type": "role",
                "matched": matched_roles,
                "text": f"같은 역할({', '.join(matched_roles)})을 공유합니다",
            })

        candidates.append({
            "branch_id": other.id,
            "name": other.name,
            "reasons": reasons,
        })

    candidates.sort(key=lambda c: (-len(c["reasons"]), -c["reasons"][0]["score"]))
    return candidates
