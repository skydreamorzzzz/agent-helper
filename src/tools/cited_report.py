from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src.planner.models import RiskLevel
from src.tools.base import Tool


class WriteCitedReportArguments(BaseModel):
    topic: str = Field(min_length=1, max_length=300)
    report_file: str = Field(min_length=1, max_length=300)
    sub_topics: list[str] = Field(default_factory=list)

    @field_validator("report_file")
    @classmethod
    def report_file_must_be_under_reports(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if normalized.startswith("/") or ".." in normalized.split("/"):
            raise ValueError("report_file must be a relative path under reports/")
        if not normalized.startswith("reports/") or not normalized.endswith(".md"):
            raise ValueError("report_file must be under reports/ and end with .md")
        return normalized


class WriteCitedReportTool(Tool):
    name = "write_cited_report"
    description = (
        "Synthesize all prior search_web results into a cited Chinese Markdown research report and save it "
        "under workspace/reports/. Handled by the executor with LLM synthesis."
    )
    argument_schema = WriteCitedReportArguments
    risk_level = RiskLevel.WRITE

    def execute(self, arguments: WriteCitedReportArguments) -> str:
        raise RuntimeError("write_cited_report requires an LLM-backed executor implementation")
