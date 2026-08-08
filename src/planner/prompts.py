from __future__ import annotations

from src.tools.registry import ToolRegistry


def build_router_prompt(user_input: str, memory_context: str = "") -> str:
    return (
        "Classify the user request for a local assistant.\n"
        "Return JSON only: {\"route\":\"direct_answer|single_tool|web_lookup|planned_task|deep_research|clarification\","
        "\"reason\":\"...\",\"missing_information\":[]}\n"
        "Rules:\n"
        "- Greeting, casual chat, or a question answerable from general knowledge: direct_answer.\n"
        "- A single obvious tool action (calculate, read/write a file): single_tool.\n"
        "- A simple question needing current, latest, or external facts but only 1-2 searches: web_lookup.\n"
        "- Multiple dependent steps: planned_task.\n"
        "- A request needing systematic decomposition, multiple sources, comparison, or a saved research report: deep_research.\n"
        "- Do not classify every factual question as deep_research. Use direct_answer for stable general knowledge.\n"
        "- Missing required parameters: clarification.\n"
        "Examples:\n"
        "- '你好' -> direct_answer\n"
        "- '23.5 * 17' -> single_tool\n"
        "- 'What is the latest Tavily API pricing?' -> web_lookup\n"
        "- 'Who is the current CEO of OpenAI?' -> web_lookup\n"
        "- 'Transformer 的残差连接是什么？' -> direct_answer\n"
        "- '调研主流 Agent Memory 方法并比较优缺点' -> deep_research\n"
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


def build_research_plan_prompt(
    *,
    user_input: str,
    registry: ToolRegistry,
    memory_context: str,
    max_steps: int,
) -> str:
    max_subtopics = max(2, (max_steps - 1) // 2)
    return (
        "You are planning an in-depth web research task for a local assistant.\n"
        "Break the research topic into focused sub-topics and return JSON only:\n"
        "{"
        "\"topic\":\"...\","
        "\"sub_topics\":[{\"question\":\"...\",\"search_queries\":[\"...\"]}],"
        "\"report_file\":\"reports/xxx.md\","
        "\"assumptions\":[]"
        "}\n"
        f"Maximum total search queries: {max_steps - 1}. Use no more than {max_subtopics} sub-topics.\n"
        "Each sub-topic should have 1-2 concrete search queries in the same language as the user request.\n"
        "report_file must be a path under reports/ ending in .md.\n"
        "Do not include unresolved_questions.\n\n"
        f"Available tools:\n{registry.describe_tools()}\n\n"
        f"Relevant memory context:\n{memory_context}\n\n"
        f"User request:\n{user_input}"
    )


def build_cited_report_prompt(*, topic: str, sub_topics: list[str], materials: str, report_file: str) -> str:
    return (
        "You are writing a deep-research report in Chinese Markdown based only on the provided search materials.\n"
        f"Topic: {topic}\n"
        f"Sub-topics: {sub_topics}\n"
        f"Target file: {report_file}\n\n"
        "Requirements:\n"
        "- Write in Chinese, well-structured Markdown with a title, a short intro, one section per sub-topic, and a conclusion.\n"
        "- Cite sources inline as [来源](url). Only cite URLs that appear in the provided materials. Never fabricate sources or facts.\n"
        "- End with a numbered \"参考来源\" list of the URLs actually cited.\n"
        "- Output raw Markdown only. Do not wrap it in JSON or code fences.\n\n"
        f"Search materials:\n{materials}"
    )
