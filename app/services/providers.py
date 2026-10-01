import json
from abc import ABC, abstractmethod
import os

from dotenv import load_dotenv

load_dotenv()


def _to_openai_content(content):
    """context_builder의 provider-neutral content(str 또는 text/image 블록 리스트)를 OpenAI 포맷으로 변환한다."""
    if isinstance(content, str):
        return content
    blocks = []
    for b in content:
        if b["type"] == "text":
            blocks.append({"type": "text", "text": b["text"]})
        elif b["type"] == "image":
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{b['media_type']};base64,{b['data']}"},
            })
    return blocks


def _to_anthropic_content(content):
    """context_builder의 provider-neutral content(str 또는 text/image 블록 리스트)를 Anthropic 포맷으로 변환한다."""
    if isinstance(content, str):
        return content
    blocks = []
    for b in content:
        if b["type"] == "text":
            blocks.append({"type": "text", "text": b["text"]})
        elif b["type"] == "image":
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": b["media_type"], "data": b["data"]},
            })
    return blocks


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, context: list[dict], model_name: str, system_prompt: str) -> tuple[str, int, int]:
        """context를 받아 (답변, input_tokens, output_tokens)를 반환한다."""

    def generate_with_tools(
        self,
        context: list[dict],
        model_name: str,
        system_prompt: str,
        tools: list[dict],
        tool_handler,
    ) -> tuple[str, int, int]:
        """도구를 지원하지 않는 provider의 폴백: 그냥 generate 호출."""
        return self.generate(context, model_name, system_prompt)


class _OpenAICompatibleProvider(LLMProvider):
    """OpenAI / OpenAI-호환 엔드포인트용 공통 구현."""

    _client = None  # 서브클래스에서 설정

    def generate(self, context, model_name, system_prompt):
        context = [{**m, "content": _to_openai_content(m["content"])} for m in context]
        messages = [{"role": "system", "content": system_prompt}] + context
        response = self._client.chat.completions.create(model=model_name, messages=messages)
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    def generate_with_tools(self, context, model_name, system_prompt, tools, tool_handler):
        context = [{**m, "content": _to_openai_content(m["content"])} for m in context]
        messages = [{"role": "system", "content": system_prompt}] + context
        oai_tools = [{"type": "function", "function": t} for t in tools]
        total_in = total_out = 0

        while True:
            response = self._client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=oai_tools,
                tool_choice="auto",
            )
            total_in += response.usage.prompt_tokens
            total_out += response.usage.completion_tokens
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    result = tool_handler(tc.function.name, json.loads(tc.function.arguments))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })
            else:
                return choice.message.content, total_in, total_out


class OpenAIProvider(_OpenAICompatibleProvider):
    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI()


class AnthropicProvider(LLMProvider):
    def __init__(self):
        import anthropic
        self._client = anthropic.Anthropic()

    def generate(self, context, model_name, system_prompt):
        context = [{**m, "content": _to_anthropic_content(m["content"])} for m in context]
        response = self._client.messages.create(
            model=model_name,
            system=system_prompt,
            messages=context,
            max_tokens=1024,
        )
        return response.content[0].text, response.usage.input_tokens, response.usage.output_tokens

    def generate_with_tools(self, context, model_name, system_prompt, tools, tool_handler):
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]
        messages = [{**m, "content": _to_anthropic_content(m["content"])} for m in context]
        total_in = total_out = 0

        while True:
            response = self._client.messages.create(
                model=model_name,
                system=system_prompt,
                messages=messages,
                tools=anthropic_tools,
                max_tokens=1024,
            )
            total_in += response.usage.input_tokens
            total_out += response.usage.output_tokens

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = tool_handler(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                text = next((b.text for b in response.content if hasattr(b, "text")), "")
                return text, total_in, total_out


_CHATKHU_BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"
_PROVIDERS: dict[str, LLMProvider] = {}


class ChatKHUProvider(_OpenAICompatibleProvider):
    """경희대 AI Gateway — OpenAI 호환 엔드포인트 사용."""

    def __init__(self):
        from openai import OpenAI
        api_key = os.environ.get("CHATKHU_API_KEY")
        if not api_key:
            raise ValueError("CHATKHU_API_KEY가 .env에 설정되지 않았습니다.")
        self._client = OpenAI(api_key=api_key, base_url=_CHATKHU_BASE_URL)


def get_provider(model_provider: str) -> LLMProvider:
    if model_provider not in _PROVIDERS:
        if model_provider == "openai":
            _PROVIDERS["openai"] = OpenAIProvider()
        elif model_provider == "anthropic":
            _PROVIDERS["anthropic"] = AnthropicProvider()
        elif model_provider == "chatkhu":
            _PROVIDERS["chatkhu"] = ChatKHUProvider()
        else:
            raise ValueError(
                f"지원하지 않는 model_provider: '{model_provider}'. openai / anthropic / chatkhu 중 하나를 사용하세요."
            )
    return _PROVIDERS[model_provider]
