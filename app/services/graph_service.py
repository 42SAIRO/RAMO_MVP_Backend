def build_branch_graph(
    branches: list,
    message_counts: dict,
    include_inactive: bool = False,
    merge_edges: list | None = None,
    personas_by_id: dict | None = None,
) -> dict:
    """branches 목록을 nodes / edges 형태로 변환한다.

    - nodes: 각 브랜치 하나
    - edges: parent_branch_id 가 있는 브랜치마다 parent → child 관계 하나 (type="fork"),
      머지 브랜치는 merge_edges에 담긴 부모 관계마다 하나씩 (type="merge")
    - include_inactive=False 이면 deleted 브랜치는 제외한다.
    - is_collapsed=True 인 브랜치의 모든 하위 브랜치는 nodes/edges에서 제외한다 (머지 부모 관계로 연결된 브랜치는 접힘 대상에서 제외).
    """
    merge_edges = merge_edges or []
    personas_by_id = personas_by_id or {}
    visible = [b for b in branches if include_inactive or b.status != "deleted"]
    visible_ids = {b.id for b in visible}

    # parent_id → 자식 id 목록 맵 (fork 관계만; 트리 구조 기준 접힘 전파용)
    children_map: dict[str, list[str]] = {}
    for b in visible:
        if b.parent_branch_id:
            children_map.setdefault(b.parent_branch_id, []).append(b.id)

    # 접힌 브랜치의 모든 하위 브랜치 id 수집 (BFS)
    hidden_ids: set[str] = set()
    for b in visible:
        if b.is_collapsed:
            queue = list(children_map.get(b.id, []))
            while queue:
                child_id = queue.pop()
                hidden_ids.add(child_id)
                queue.extend(children_map.get(child_id, []))

    nodes = []
    edges = []

    for branch in visible:
        if branch.id in hidden_ids:
            continue

        persona = personas_by_id.get(branch.persona_id) if branch.persona_id else None
        nodes.append({
            "id": branch.id,
            "type": "branch",
            "label": branch.name,
            "description": branch.description,
            "summary": branch.summary,
            "status": branch.status,
            "is_collapsed": branch.is_collapsed,
            "is_merge": branch.is_merge,
            "message_count": message_counts.get(branch.id, 0),
            "persona_id": branch.persona_id,
            "persona_slug": persona.slug if persona else None,
            "persona_key": persona.slug if persona else None,
            "persona_name": persona.name if persona else None,
        })

        if branch.parent_branch_id is not None:
            edges.append({
                "id": f"edge-{branch.id}",
                "source": branch.parent_branch_id,
                "target": branch.id,
                "type": "fork",
                "fork_from_message_id": branch.fork_from_message_id,
            })

    for mp in merge_edges:
        if mp.branch_id in hidden_ids or mp.branch_id not in visible_ids:
            continue
        if mp.parent_branch_id not in visible_ids:
            continue
        edges.append({
            "id": f"merge-{mp.branch_id}-{mp.parent_branch_id}",
            "source": mp.parent_branch_id,
            "target": mp.branch_id,
            "type": "merge",
            "fork_from_message_id": None,
        })

    return {"nodes": nodes, "edges": edges}
