from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.config import WORKSPACE_DIR
from src.planner.models import RiskLevel
from src.tools.base import Tool


class PolicyAction(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyDecision:
    action: PolicyAction
    reason: str
    risk_level: RiskLevel


class ToolExecutionPolicy:
    def evaluate(
        self,
        *,
        tool: Tool,
        arguments: dict[str, Any],
        confirm_write_actions: bool,
    ) -> PolicyDecision:
        risk = RiskLevel(tool.risk_level)

        if risk == RiskLevel.READ_ONLY:
            return PolicyDecision(PolicyAction.ALLOW, "read-only tool is allowed", risk)

        argument_reason = self._argument_level_reason(tool.name, arguments)
        if argument_reason is not None:
            return PolicyDecision(PolicyAction.CONFIRM, argument_reason, risk)

        if risk == RiskLevel.WRITE:
            if confirm_write_actions:
                return PolicyDecision(PolicyAction.CONFIRM, "write action requires confirmation", risk)
            return PolicyDecision(PolicyAction.ALLOW, "write confirmations are disabled", risk)

        if risk == RiskLevel.DESTRUCTIVE:
            return PolicyDecision(PolicyAction.CONFIRM, "destructive action requires confirmation", risk)

        if risk == RiskLevel.EXTERNAL:
            return PolicyDecision(PolicyAction.CONFIRM, "external action requires confirmation", risk)

        return PolicyDecision(PolicyAction.DENY, f"unsupported risk level: {risk}", risk)

    def _argument_level_reason(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        if tool_name == "write_text_file" and bool(arguments.get("overwrite", False)):
            return "write_text_file overwrite=true requires confirmation"
        if tool_name == "write_cited_report":
            path = arguments.get("report_file")
            if path and (WORKSPACE_DIR / str(path)).exists():
                return "overwriting an existing report requires confirmation"
        return None
