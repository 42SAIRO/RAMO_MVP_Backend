# Chatbot Branch Project

FastAPI + SQLAlchemy + SQLite로 만든 브랜치형 LLM 챗봇 서버입니다.

일반 챗봇처럼 대화가 한 줄로만 이어지는 것이 아니라, 특정 메시지에서 새 브랜치를 만들어 다른 방향의 대화를 이어갈 수 있습니다. 세션 안에는 여러 브랜치가 있고, 각 브랜치는 메시지 흐름과 태그를 따로 가질 수 있습니다.

## 현재 확인 상태

- 세션 생성 / 조회
- 브랜치 생성 / 조회 / 상태 변경 / 접기
- 브랜치별 채팅
- 브랜치 그래프 조회
- 세션 메모리 조회 / 수정 / 자동 추출
- 자동 브랜치 이름 생성
- 자동 태그 생성 / 수동 태그 부여 / 브랜치별 태그 조회
- OpenAI 기본 채팅
- LLM provider 변경 기능은 코드에 있지만 아직 최종 확인 전

## 기술 스택

| 구분 | 내용 |
|---|---|
| Backend | FastAPI |
| DB | SQLite |
| ORM | SQLAlchemy |
| LLM | OpenAI API |
| Embedding | `text-embedding-3-small` |
| API 문서 | Swagger UI |

## 폴더 구조

```text
app/
  main.py                  FastAPI 앱 시작점
  database.py              DB 연결 설정
  models/models.py         SQLAlchemy 테이블 정의
  schemas/schemas.py       요청/응답 스키마
  routers/                 API 엔드포인트
  repositories/            DB CRUD 함수
  services/                LLM, context, graph, tag 로직
tests/
README.md
requirements.txt
```

## 실행 방법

### 1. 가상환경 생성 및 활성화

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 2. 패키지 설치

```cmd
pip install -r requirements.txt
```

### 3. `.env` 파일 생성

프로젝트 루트에 `.env` 파일을 만들고 OpenAI API key를 넣습니다.

```env
OPENAI_API_KEY=sk-...
```

### 4. 서버 실행

```cmd
uvicorn app.main:app --reload
```

서버가 켜지면 아래 주소로 Swagger UI에 접속합니다.

```text
http://127.0.0.1:8000/docs
```

## 기본 사용 흐름

### 1. 세션 생성

```http
POST /sessions
```

예시 요청:

```json
{
  "title": "구현 확인"
}
```

예시 응답:

```json
{
  "id": "session-id",
  "title": "구현 확인",
  "main_branch_id": "main-branch-id"
}
```

여기서 `id`는 세션 ID이고, `main_branch_id`는 처음 채팅할 브랜치 ID입니다.

### 2. main 브랜치에서 채팅

```http
POST /chat
```

예시 요청:

```json
{
  "branch_id": "main-branch-id",
  "message": "LLM이 뭐야?",
  "model_provider": "openai",
  "model_name": "gpt-4o-mini"
}
```

첫 메시지를 보내면 세션 제목이 LLM에 의해 자동으로 바뀔 수 있습니다.

### 3. 메시지 목록 확인

```http
GET /branches/{branch_id}/messages
```

부모 브랜치에서 분기 이전 메시지까지 같이 보고 싶으면:

```http
GET /branches/{branch_id}/messages?include_inherited=true
```

### 4. 특정 메시지에서 새 브랜치 생성

```http
POST /branches
```

예시 요청:

```json
{
  "session_id": "session-id",
  "parent_branch_id": "main-branch-id",
  "fork_from_message_id": 4,
  "name": "트랜스포머 질문"
}
```

주의할 점:

- `session_id`는 세션 ID입니다.
- `parent_branch_id`는 분기하려는 기존 브랜치 ID입니다.
- `fork_from_message_id`는 반드시 `parent_branch_id` 브랜치 안에 있는 메시지 ID여야 합니다.

### 5. 새 브랜치에서 이어서 채팅

```http
POST /chat
```

예시 요청:

```json
{
  "branch_id": "new-branch-id",
  "message": "인코더와 디코더 설명해줘",
  "model_provider": "openai",
  "model_name": "gpt-4o-mini"
}
```

새 브랜치에서는 분기 지점 이전의 부모 대화만 context에 포함됩니다.

## 주요 API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 서버 상태 확인 |
| POST | `/sessions` | 세션 생성 + main 브랜치 생성 |
| GET | `/sessions` | 세션 목록 조회 |
| PATCH | `/sessions/{session_id}/title` | 세션 제목 수정 |
| GET | `/sessions/{session_id}/branches` | 세션 내 브랜치 목록 조회 |
| GET | `/sessions/{session_id}/graph` | 브랜치 그래프 조회 |
| GET | `/sessions/{session_id}/search?q=검색어` | 브랜치 이름/태그 검색 |
| GET | `/sessions/{session_id}/memory` | 세션 메모리 조회 |
| PATCH | `/sessions/{session_id}/memory` | 세션 메모리 수정 |
| POST | `/sessions/{session_id}/memory/extract` | 대화에서 사용자 메모리 자동 추출 |
| POST | `/chat` | 브랜치에 메시지 전송 + LLM 응답 저장 |
| POST | `/branches` | 특정 메시지 기준 새 브랜치 생성 |
| PATCH | `/branches/{branch_id}` | 브랜치 상태 또는 접힘 여부 변경 |
| GET | `/branches/{branch_id}/messages` | 브랜치 메시지 조회 |
| PATCH | `/branches/{branch_id}/name` | 브랜치 이름 수정 |
| POST | `/branches/{branch_id}/auto-name` | 브랜치 이름 자동 생성 |
| POST | `/branches/{branch_id}/auto-tag` | 브랜치 태그 자동 생성 |
| GET | `/branches/{branch_id}/tags` | 특정 브랜치에 연결된 태그 조회 |
| POST | `/branches/{branch_id}/tags` | 브랜치에 태그 수동 부여 |
| DELETE | `/branches/{branch_id}/tags/{tag_id}` | 브랜치에서 태그 제거 |
| POST | `/tags` | 태그 수동 생성 |
| GET | `/sessions/{session_id}/tags` | 세션 전체 태그 목록 조회 |

## 브랜치 상태

브랜치 상태는 `PATCH /branches/{branch_id}`로 바꿉니다.

```json
{
  "status": "inactive",
  "is_collapsed": true
}
```

| 값 | 동작 |
|---|---|
| `active` | 채팅 가능 |
| `inactive` | 채팅 불가, 그래프에는 표시 |
| `deleted` | 채팅 불가, 기본 그래프에서 제외 |
| `is_collapsed: true` | 해당 브랜치의 하위 브랜치를 그래프에서 숨김 |

JSON에서는 `true`, `false`를 소문자로 써야 합니다.

## 태그 확인 방법

세션 안에 만들어진 전체 태그 목록:

```http
GET /sessions/{session_id}/tags
```

특정 브랜치에 실제로 붙은 태그 목록:

```http
GET /branches/{branch_id}/tags
```

브랜치 검색 결과에서 태그까지 같이 보고 싶으면:

```http
GET /sessions/{session_id}/search?q=트랜스포머
```

## 실행

서버 실행:

```cmd
uvicorn app.main:app --reload
```

Swagger 접속:

```text
http://127.0.0.1:8000/docs
```
