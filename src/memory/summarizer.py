from __future__ import annotations

from typing import Protocol

from src.agent.protocol import ModelOutputParseError, extract_first_json_object


class SummaryLLM(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


class ConversationSummarizer:
    def __init__(self, llm_client: SummaryLLM | None = None) -> None:
        self.llm_client = llm_client

    def should_summarize(self, messages: list[dict[str, str]], threshold: int) -> bool:
        return len(messages) > threshold

    def summarize(
        self,
        *,
        existing_summary: str,
        messages: list[dict[str, str]],
    ) -> str:
        if self.llm_client is None:
            return self.fallback_summary(existing_summary=existing_summary, messages=messages)

        prompt = (
            "Create a concise structured conversation summary. Preserve only:\n"
            "- current goal\n- completed work\n- explicit user requirements\n"
            "- important decisions\n- unresolved issues\n"
            "Do not include ordinary small talk.\n"
            "Return JSON only: {\"summary\":\"...\"}\n"
            f"Existing summary:\n{existing_summary or '(none)'}\n"
            f"Recent messages:\n{self._format_messages(messages)}"
        )
        raw = self.llm_client.chat([{"role": "user", "content": prompt}])
        json_text = extract_first_json_object(raw)
        if json_text is None:
            raise ModelOutputParseError("Summary response returned no JSON object")
        import json

        payload = json.loads(json_text)
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            raise ModelOutputParseError("Summary response contained empty summary")
        return summary

    def fallback_summary(self, *, existing_summary: str, messages: list[dict[str, str]]) -> str:
        important = []
        for message in messages:
            content = message["content"].strip()
            if any(marker in content for marker in ("要求", "必须", "不要", "项目", "记住", "实现", "决定")):
                important.append(f"{message['role']}: {content}")
        joined = "\n".join(important[-8:])
        if existing_summary and joined:
            return f"{existing_summary}\n{joined}"
        return joined or existing_summary

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

