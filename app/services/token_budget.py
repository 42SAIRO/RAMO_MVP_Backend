import tiktoken

MODEL_CONTEXT_WINDOWS = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
}
DEFAULT_CONTEXT_WINDOW = 128_000

COMPRESSION_THRESHOLD_RATIO = 0.95
RESPONSE_RESERVE_TOKENS = 2000  # 응답 + 시스템 프롬프트용 여유분

_FALLBACK_ENCODING = "cl100k_base"
_CHARS_PER_TOKEN_APPROX = 4  # tiktoken 자체를 못 쓸 때(네트워크 차단 등)의 최후 근사치
IMAGE_TOKEN_ESTIMATE = 1500  # vision 모델의 고해상도 이미지 1장당 토큰 비용 근사치
_encoding_cache: dict[str, "tiktoken.Encoding | None"] = {}


def _get_encoding(model_name: str) -> "tiktoken.Encoding | None":
    """모델에 맞는 인코딩을 반환한다. tiktoken이 인코딩 파일을 못 받아오면(네트워크 차단 등)
    None을 반환해 문자수 기반 근사치로 폴백하게 한다."""
    if model_name not in _encoding_cache:
        try:
            _encoding_cache[model_name] = tiktoken.encoding_for_model(model_name)
        except KeyError:
            try:
                _encoding_cache[model_name] = tiktoken.get_encoding(_FALLBACK_ENCODING)
            except Exception:
                _encoding_cache[model_name] = None
        except Exception:
            _encoding_cache[model_name] = None
    return _encoding_cache[model_name]


def count_tokens(text: str, model_name: str) -> int:
    """텍스트의 토큰 수를 센다. openai 모델이 아니면 cl100k_base로 근사하고,
    tiktoken을 아예 쓸 수 없으면 문자수/4로 근사한다."""
    if not text:
        return 0
    encoding = _get_encoding(model_name)
    if encoding is None:
        return len(text) // _CHARS_PER_TOKEN_APPROX
    return len(encoding.encode(text))


def count_context_tokens(context: list[dict], model_name: str) -> int:
    """LLM에 보낼 messages 배열(role/content dict 목록)의 총 토큰 수를 센다.

    content는 문자열이거나(텍스트 메시지), 이미지가 첨부된 경우 text/image 블록 리스트일 수 있다.
    """
    total = 0
    for m in context:
        content = m.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content, model_name)
        else:
            for block in content:
                if block.get("type") == "text":
                    total += count_tokens(block.get("text", ""), model_name)
                elif block.get("type") == "image":
                    total += IMAGE_TOKEN_ESTIMATE
    return total


def get_context_window(model_name: str) -> int:
    return MODEL_CONTEXT_WINDOWS.get(model_name, DEFAULT_CONTEXT_WINDOW)


def get_compression_threshold(model_name: str) -> int:
    """이 토큰 수 이상이면 브랜치 자체 압축을 트리거한다 (context window의 95% - 응답 여유분)."""
    return int(get_context_window(model_name) * COMPRESSION_THRESHOLD_RATIO) - RESPONSE_RESERVE_TOKENS
