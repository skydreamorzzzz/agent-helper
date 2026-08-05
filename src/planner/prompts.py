from __future__ import annotations

from src.tools.registry import ToolRegistry


def build_router_prompt(user_input: str, memory_context: str = "") -> str:
    return (
        "Classify the user request for a local assistant.\n"
        "Return JSON only: {\"route\":\"direct_answer|single_tool|planned_task|clarification\","
        "\"reason\":\"...\",\"missing_information\":[]}\n"
        "Rules: ordinary chat is direct_answer; exactly one obvious tool action is single_tool; "
        "multiple dependent steps is planned_task; missing required parameters is clarification.\n"
        f"Memory context:\n{memory_context}\n"
        f"User request:\n{user_input}"
    )


def build_planner_prompt(
    *,
    user_input: str,
    registry: ToolRegistry,
    memory_context: str,
    max_steps: int,
) -> str:
    return (
        "Create a structured executable plan for the local assistant.\n"
        "Return JSON only matching this shape:\n"
        "{"
        "\"goal\":\"...\","
        "\"steps\":[{\"id\":\"step_1\",\"description\":\"...\",\"tool_name\":\"read_text_file\","
        "\"arguments\":{},\"depends_on\":[],\"expected_output\":\"...\"}],"
        "\"assumptions\":[],\"unresolved_questions\":[],\"final_output_requirement\":\"...\""
        "}\n"
        f"Maximum steps: {max_steps}.\n"
        "Use only real tools from the registry. Do not invent results. If a later tool argument depends "
        "on a previous result, use a clear placeholder string like ${step_1.result}.\n"
        "For tasks that need reading text, extracting or summarizing it, and then writing a file, use this pattern: "
        "read_text_file -> transform_text -> write_text_file.\n"
        "Do not ask unresolved questions when a reasonable assumption is enough to proceed; put those in assumptions. "
        "Use unresolved_questions only for missing information that makes execution impossible, such as missing file path or output path.\n"
        "Do not set overwrite=true unless the user explicitly asked to overwrite an existing file.\n\n"
        f"Relevant memory context:\n{memory_context}\n\n"
        f"Available tools:\n{registry.describe_tools()}\n\n"
        f"User request:\n{user_input}"
    )


def build_argument_resolution_prompt(
    *,
    step_id: str,
    tool_name: str,
    arguments: dict,
    observations: dict[str, object],
    tool_schema: dict,
) -> str:
    return (
        "Resolve tool arguments for the next plan step using prior observations.\n"
        "Return JSON only with the final arguments object. Do not include explanations.\n"
        f"Step id: {step_id}\n"
        f"Tool: {tool_name}\n"
        f"Current arguments: {arguments}\n"
        f"Prior observations: {observations}\n"
        f"Tool argument schema: {tool_schema}"
    )


def build_final_answer_prompt(goal: str, observations: dict[str, object], requirement: str) -> str:
    return (
        "Generate the final answer for a completed plan.\n"
        "Return JSON only: {\"type\":\"final_answer\",\"content\":\"...\"}\n"
        f"Goal: {goal}\n"
        f"Final output requirement: {requirement}\n"
        f"Observations: {observations}"
    )
