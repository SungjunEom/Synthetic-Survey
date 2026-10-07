"""Small OpenAI adapter; request retries are owned by the experiment runner."""
from dataclasses import asdict, dataclass


@dataclass
class Reply:
    content: str | None
    finish_reason: str
    refusal: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    system_fingerprint: str | None = None
    usage: dict | None = None

    def to_dict(self):
        return asdict(self)


class ProviderError(Exception):
    def __init__(self, kind: str, retryable: bool, fatal: bool = False, request_id: str | None = None):
        super().__init__(kind)
        self.kind, self.retryable, self.fatal, self.request_id = kind, retryable, fatal, request_id


class OpenAIProvider:
    def __init__(self, api_key: str, timeout: float):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    def complete(self, messages: list[dict], schema: dict, settings: dict, seed: int) -> Reply:
        from openai import APIConnectionError, APIStatusError
        kwargs = {"model": settings["model"], "messages": messages,
                  "max_completion_tokens": settings["max_tokens"],
                  "response_format": {"type": "json_schema", "json_schema": {
                      "name": "survey_answers", "strict": True, "schema": schema}}}
        if settings["temperature"] is not None:
            kwargs["temperature"] = settings["temperature"]
        if settings["send_seed"]:
            kwargs["seed"] = seed
        try:
            result = self.client.chat.completions.create(**kwargs)
        except APIConnectionError as exc:
            raise ProviderError(type(exc).__name__, retryable=True) from exc
        except APIStatusError as exc:
            retryable = exc.status_code in {408, 409, 429} or exc.status_code >= 500
            # Never persist the exception body: it can contain echoed request data.
            raise ProviderError(f"HTTP_{exc.status_code}", retryable=retryable,
                                fatal=not retryable, request_id=exc.request_id) from exc
        if not result.choices:
            raise ProviderError("EmptyCompletion", retryable=True)
        choice = result.choices[0]
        return Reply(content=choice.message.content, finish_reason=choice.finish_reason,
                     refusal=choice.message.refusal, response_id=result.id,
                     request_id=getattr(result, "_request_id", None), model=result.model,
                     system_fingerprint=result.system_fingerprint,
                     usage=result.usage.model_dump() if result.usage else None)
