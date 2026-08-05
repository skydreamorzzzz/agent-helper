from __future__ import annotations

from src.planner.models import Plan, PlanValidationResult
from src.tools.registry import ToolArgumentError, ToolRegistry, UnknownToolError


class PlanValidator:
    def __init__(self, registry: ToolRegistry, *, max_steps: int = 8) -> None:
        self.registry = registry
        self.max_steps = max_steps

    def validate(self, plan: Plan) -> PlanValidationResult:
        errors: list[str] = []
        if len(plan.steps) > self.max_steps:
            errors.append(f"Plan has {len(plan.steps)} steps, maximum is {self.max_steps}")
        ids = [step.id for step in plan.steps]
        if len(ids) != len(set(ids)):
            errors.append("Plan contains duplicate step ids")
        id_set = set(ids)
        for step in plan.steps:
            try:
                tool = self.registry.get(step.tool_name)
                tool.argument_schema.model_validate(step.arguments)
            except UnknownToolError:
                errors.append(f"Unknown tool in step {step.id}: {step.tool_name}")
            except Exception as exc:
                errors.append(f"Invalid arguments in step {step.id}: {exc}")
            for dependency in step.depends_on:
                if dependency not in id_set:
                    errors.append(f"Step {step.id} depends on missing step {dependency}")
        if self._has_cycle(plan):
            errors.append("Plan dependencies contain a cycle")
        if plan.unresolved_questions:
            errors.append("Plan has unresolved questions and cannot execute")
        return PlanValidationResult(ok=not errors, errors=errors)

    def _has_cycle(self, plan: Plan) -> bool:
        graph = {step.id: step.depends_on for step in plan.steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for dependency in graph.get(node, []):
                if visit(dependency):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(step_id) for step_id in graph)

