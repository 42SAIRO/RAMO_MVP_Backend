from datetime import datetime, timezone

from sqlalchemy import Column, Integer, Text, Boolean, ForeignKey, Index
from app.database import Base


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Text, primary_key=True)
    title = Column(Text, nullable=False, server_default="새 대화")
    memory = Column(Text, nullable=True)  # 브랜치 무관 공유 메모리
    status = Column(Text, nullable=False, server_default="active")  # active / deleted
    deleted_at = Column(Text, nullable=True)  # 휴지통 이동 시각 (ISO datetime)
    created_at = Column(Text, nullable=False, default=utc_now_text)
    updated_at = Column(Text, nullable=False, default=utc_now_text)


class Persona(Base):
    """브랜치 단위로 선택하는 역할 프롬프트 (기획 컨설턴트, 번역 전문가 등)."""
    __tablename__ = "personas"

    id = Column(Text, primary_key=True)
    slug = Column(Text, nullable=False, unique=True)
    name = Column(Text, nullable=False)              # "기획 컨설턴트" 등 표시용 이름
    description = Column(Text, nullable=True)         # 목록 화면용 짧은 설명
    system_prompt = Column(Text, nullable=False)
    default_model_provider = Column(Text, nullable=False)
    default_model_name = Column(Text, nullable=False)
    created_at = Column(Text, nullable=False, default=utc_now_text)


class Branch(Base):
    __tablename__ = "branches"

    id = Column(Text, primary_key=True)
    session_id = Column(Text, ForeignKey("conversations.id"), nullable=False)
    parent_branch_id = Column(Text, ForeignKey("branches.id"), nullable=True)
    fork_from_message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)
    name = Column(Text, nullable=False, server_default="main")
    head_id = Column(Integer, nullable=True)
    status = Column(Text, nullable=False, server_default="active")   # active / inactive / deleted
    deleted_at = Column(Text, nullable=True)  # 휴지통 이동 시각 (ISO datetime)
    summary = Column(Text, nullable=True)  # 대화 핵심 결과 요약 (fork 시 부모 브랜치에 자동 생성)
    description = Column(Text, nullable=True)  # 시각화용 한두 문장 설명 (이 브랜치 자신의 내용만, 조상 내용 제외)
    live_summary = Column(Text, nullable=True)  # 이 브랜치 자신의 오래된 메시지를 압축한 요약 (토큰 절약용)
    live_summary_cutoff_id = Column(Integer, nullable=True)  # 이 메시지 id까지는 live_summary로 대체됨
    persona_id = Column(Text, ForeignKey("personas.id"), nullable=True)  # 이 브랜치가 수행할 역할 (없으면 기본 RAMO)
    is_collapsed = Column(Boolean, nullable=False, server_default="0")
    is_merge = Column(Boolean, nullable=False, server_default="0")    # 여러 브랜치를 합친 머지 브랜치인지 여부
    is_main = Column(Boolean, nullable=False, server_default="0")     # 선택된 main 경로 여부
    created_at = Column(Text, nullable=False, default=utc_now_text)
    updated_at = Column(Text, nullable=False, default=utc_now_text)


class BranchMergeParent(Base):
    """머지 브랜치 하나가 가질 수 있는 여러 부모 브랜치 관계.

    일반 분기는 Branch.parent_branch_id(단일)로 표현하지만,
    머지 브랜치는 부모가 여럿이라 별도 테이블로 관리한다.
    """
    __tablename__ = "branch_merge_parents"

    branch_id = Column(Text, ForeignKey("branches.id"), primary_key=True)
    parent_branch_id = Column(Text, ForeignKey("branches.id"), primary_key=True)
    summary = Column(Text, nullable=False)   # 머지 시점에 parent 브랜치 전체 맥락을 요약한 내용
    created_at = Column(Text, nullable=False, default=utc_now_text)


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Text, ForeignKey("conversations.id"), nullable=False)
    branch_id = Column(Text, ForeignKey("branches.id"), nullable=False)
    parent_id = Column(Integer, ForeignKey("messages.id"), nullable=True)
    role = Column(Text, nullable=False)                              # user / assistant / system
    content = Column(Text, nullable=False)
    model_provider = Column(Text, nullable=True)                    # openai, anthropic 등
    model_name = Column(Text, nullable=True)                        # gpt-4o-mini 등
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    status = Column(Text, nullable=False, server_default="active")  # active / deleted
    created_at = Column(Text, nullable=False, default=utc_now_text)


class Tag(Base):
    __tablename__ = "tags"

    id = Column(Text, primary_key=True)
    session_id = Column(Text, ForeignKey("conversations.id"), nullable=True)
    name = Column(Text, nullable=False)
    color = Column(Text, nullable=True)
    type = Column(Text, nullable=False, server_default="normal")    # normal / highlight
    created_at = Column(Text, nullable=False, default=utc_now_text)


class MessageTag(Base):
    __tablename__ = "message_tags"

    message_id = Column(Integer, ForeignKey("messages.id"), primary_key=True)
    tag_id = Column(Text, ForeignKey("tags.id"), primary_key=True)
    created_at = Column(Text, nullable=False, default=utc_now_text)


class BranchTag(Base):
    __tablename__ = "branch_tags"

    branch_id = Column(Text, ForeignKey("branches.id"), primary_key=True)
    tag_id = Column(Text, ForeignKey("tags.id"), primary_key=True)
    created_at = Column(Text, nullable=False, default=utc_now_text)


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Text, primary_key=True)
    session_id = Column(Text, ForeignKey("conversations.id"), nullable=False)
    branch_id = Column(Text, ForeignKey("branches.id"), nullable=True)   # None = 세션 전체 공유
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)  # 전송된 사용자 메시지에 귀속된 첨부
    filename = Column(Text, nullable=False)
    extracted_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    file_type = Column(Text, nullable=False, server_default="text")  # text / image
    mime_type = Column(Text, nullable=True)   # image/png, application/pdf 등 원본 MIME
    content_data = Column(Text, nullable=True)  # 원본 파일 base64 인코딩 데이터
    image_data = Column(Text, nullable=True)  # 이미지일 때만, base64 인코딩 원본
    created_at = Column(Text, nullable=False, default=utc_now_text)


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Text, primary_key=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    embedding = Column(Text, nullable=False)    # JSON 직렬화 문자열 (SQLite용)
    embedding_model = Column(Text, nullable=False)
    created_at = Column(Text, nullable=False, default=utc_now_text)


class ComparisonSession(Base):
    __tablename__ = "comparison_sessions"

    id = Column(Text, primary_key=True)
    branch_id = Column(Text, ForeignKey("branches.id"), nullable=False)
    user_message = Column(Text, nullable=False)
    response_a_content = Column(Text, nullable=False)
    response_a_provider = Column(Text, nullable=False)
    response_a_model = Column(Text, nullable=False)
    response_b_content = Column(Text, nullable=False)
    response_b_provider = Column(Text, nullable=False)
    response_b_model = Column(Text, nullable=False)
    status = Column(Text, nullable=False, server_default="pending")  # pending / done
    created_at = Column(Text, nullable=False, default=utc_now_text)


Index("ix_conversations_status_created_at", Conversation.status, Conversation.created_at)
Index("ix_conversations_status_deleted_at", Conversation.status, Conversation.deleted_at)
Index("ix_branches_session_id", Branch.session_id)
Index("ix_branches_session_status_deleted_at", Branch.session_id, Branch.status, Branch.deleted_at)
Index("ix_branches_parent_branch_id", Branch.parent_branch_id)
Index("ix_messages_session_status", Message.session_id, Message.status)
Index("ix_messages_branch_status_id", Message.branch_id, Message.status, Message.id)
Index("ix_uploaded_files_session_branch", UploadedFile.session_id, UploadedFile.branch_id)
Index("ix_embeddings_message_id", Embedding.message_id)
Index("ix_comparison_sessions_branch_status", ComparisonSession.branch_id, ComparisonSession.status)
