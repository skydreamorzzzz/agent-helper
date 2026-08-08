from __future__ import annotations

from src.tools.registry import ToolRegistry


def build_system_prompt(
    registry: ToolRegistry,
    *,
    memory_context: str = "No relevant long-term memories retrieved.",
    conversation_summary: str = "",
    execution_policy: str = "",
) -> str:
    return (
        "System Instructions\n"
        "===================\n"
        "You are a local personal assistant agent.\n"
        "You must respond with exactly one JSON object and no extra text.\n"
        "Do not include markdown, explanations, or <think> blocks.\n"
        "Choose one of these schemas:\n"
        '{"type":"tool_call","tool":"tool_name","arguments":{}}\n'
        '{"type":"final_answer","content":"answer text"}\n\n'
        "Rules:\n"
        "- Use tools when needed to compute, search web, or read/write workspace files.\n"
        "- Do not invent tool results.\n"
        "- Do not request shell, Python, OS commands, or non-registered tools.\n"
        "- After receiving tool results, continue with another tool_call if needed or final_answer.\n\n"
        "- If the current user request conflicts with old memories, follow the current user request.\n\n"
        "Relevant Long-term Memories\n"
        "===========================\n"
        f"{memory_context}\n\n"
        "Conversation Summary\n"
        "====================\n"
        f"{conversation_summary or 'No conversation summary yet.'}\n\n"
        "Execution Policy\n"
        "================\n"
        f"{execution_policy or 'No additional execution policy.'}\n\n"
        "Available Tools\n"
        "===============\n"
        f"{registry.describe_tools()}"
    )


def build_repair_prompt(invalid_output: str, error: str) -> str:
    return (
        "Your previous response was invalid for the agent JSON protocol.\n"
        "Return only one valid JSON object with one of these forms:\n"
        "Do not include markdown, explanations, or <think> blocks.\n"
        '{"type":"tool_call","tool":"tool_name","arguments":{}}\n'
        '{"type":"final_answer","content":"answer text"}\n'
        f"Validation error: {error}\n"
        f"Invalid response:\n{invalid_output}"
    )
