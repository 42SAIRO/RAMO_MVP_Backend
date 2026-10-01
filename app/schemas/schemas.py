from pydantic import BaseModel, Field


# ── Session ──────────────────────────────────────────────────────────────────

class homemessage(BaseModel):
    message: str


class eggmessage(BaseModel):
    message: str


class CreateSessionRequest(BaseModel):
    title: str = "새 대화"


class UpdateSessionTitleRequest(BaseModel):
    title: str


class UpdateMemoryRequest(BaseModel):
    memory: str


class MemoryOut(BaseModel):
    session_id: str
    memory: str | None


class SessionOut(BaseModel):
    id: str
    title: str
    main_branch_id: str


class ConversationOut(BaseModel):
    id: str
    title: str
    status: str
    deleted_at: str | None
    created_at: str
    main_branch_id: str | None = None

    model_config = {"from_attributes": True}


# ── Branch ────────────────────────────────────────────────────────────────────

class CreateBranchRequest(BaseModel):
    session_id: str
    parent_branch_id: str
    fork_from_message_id: int
    name: str | None = None  # None이면 "새 가지"로 생성 후 첫 대화 기준으로 자동 갱신
    persona_id: str | None = None  # 이 브랜치가 수행할 역할 (GET /personas 참고, 없으면 기본 RAMO)
    persona_slug: str | None = None  # persona_id 대신 slug로 지정할 수 있음


class MergeBranchRequest(BaseModel):
    session_id: str
    parent_branch_ids: list[str]   # 합칠 브랜치들 (2개 이상)
    name: str | None = None        # None이면 "병합 브랜치"로 생성 후 첫 대화 기준으로 자동 갱신


class UpdateBranchNameRequest(BaseModel):
    name: str


class PatchBranchRequest(BaseModel):
    status: str | None = None        # active / inactive / deleted
    is_collapsed: bool | None = None


class SetBranchPersonaRequest(BaseModel):
    persona_id: str | None = None  # GET /personas 참고 ("도움받기": 이 브랜치에 페르소나를 지정/변경)
    persona_slug: str | None = None  # 프론트의 안정적인 페르소나 key/slug 지정용


class BranchOut(BaseModel):
    id: str
    session_id: str
    name: str
    parent_branch_id: str | None
    fork_from_message_id: int | None
    head_id: int | None
    status: str
    deleted_at: str | None
    is_collapsed: bool
    is_merge: bool
    is_main: bool
    merge_parent_ids: list[str] = []
    summary: str | None
    description: str | None
    persona_id: str | None
    persona_slug: str | None = None
    persona_key: str | None = None
    persona_name: str | None = None
    created_at: str

    model_config = {"from_attributes": True}


class SelectMainBranchResponse(BaseModel):
    branch_id: str
    main_branch_ids: list[str]


# ── Persona ───────────────────────────────────────────────────────────────────

class PersonaOut(BaseModel):
    id: str
    slug: str
    name: str
    description: str | None
    default_model_provider: str
    default_model_name: str

    model_config = {"from_attributes": True}


# ── Message / Chat ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    branch_id: str
    message: str
    model_provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    file_ids: list[str] = Field(default_factory=list)


class FileOut(BaseModel):
    id: str
    session_id: str
    branch_id: str | None   # None = 세션 전체 공유 파일
    message_id: int | None = None
    filename: str
    summary: str | None
    file_type: str
    mime_type: str | None = None
    content_url: str | None = None
    created_at: str

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    branch_id: str | None
    model_provider: str | None = None
    model_name: str | None = None
    persona_name: str = "Ramo"
    attachments: list[FileOut] = Field(default_factory=list)
    created_at: str

    model_config = {"from_attributes": True}


class ChatResponse(BaseModel):
    reply: str
    user_message: MessageOut | None = None
    assistant_message: MessageOut | None = None


# ── Tag ──────────────────────────────────────────────────────────────────────

class CreateTagRequest(BaseModel):
    session_id: str
    name: str
    color: str | None = None
    type: str = "normal"        # normal / highlight


class AddTagRequest(BaseModel):
    tag_id: str


class TagOut(BaseModel):
    id: str
    session_id: str | None
    name: str
    color: str | None
    type: str
    created_at: str

    model_config = {"from_attributes": True}


# ── Search ────────────────────────────────────────────────────────────────────

class BranchSearchResult(BaseModel):
    id: str
    name: str
    status: str
    tags: list[str]


# ── File ─────────────────────────────────────────────────────────────────────


# ── Comparison ───────────────────────────────────────────────────────────────

class ModelSpec(BaseModel):
    provider: str
    name: str


class CompareRequest(BaseModel):
    branch_id: str
    message: str
    model_a: ModelSpec
    model_b: ModelSpec


class ModelResponse(BaseModel):
    content: str
    model_provider: str
    model_name: str


class CompareResponse(BaseModel):
    comparison_id: str
    response_a: ModelResponse
    response_b: ModelResponse


class AnalyzeResponse(BaseModel):
    comparison_id: str
    similarities: str
    differences: str


class SelectRequest(BaseModel):
    selected: str  # "a" or "b"


class SelectResponse(BaseModel):
    message_id: int
    branch_id: str
    content: str


class MergeRequest(BaseModel):
    instruction: str
    model_provider: str = "openai"
    model_name: str = "gpt-4o-mini"


class MergeResponse(BaseModel):
    comparison_id: str
    merged_content: str
    message_id: int


# ── Graph ─────────────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    type: str                  # "branch"
    label: str                 # 브랜치 이름
    description: str | None    # 시각화용 한두 문장 설명 (이 브랜치 자신의 내용만)
    summary: str | None        # 그래프 hover 설명에 사용할 브랜치 요약
    status: str                # active / inactive / deleted
    is_collapsed: bool
    is_merge: bool             # 여러 브랜치를 합친 머지 브랜치인지 여부
    message_count: int
    persona_id: str | None = None
    persona_slug: str | None = None
    persona_key: str | None = None
    persona_name: str | None = None


class GraphEdge(BaseModel):
    id: str
    source: str                # parent branch id
    target: str                # child branch id
    type: str                  # "fork" / "merge"
    fork_from_message_id: int | None


class GraphOut(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
