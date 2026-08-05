from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from src.config import WORKSPACE_DIR
from src.planner.models import RiskLevel
from src.tools.base import Tool


class ReadTextFileArguments(BaseModel):
    path: str = Field(min_length=1, max_length=300)


class WriteTextFileArguments(BaseModel):
    path: str = Field(min_length=1, max_length=300)
    content: str
    overwrite: bool = False


def resolve_workspace_path(path: str, workspace_dir: Path = WORKSPACE_DIR) -> Path:
    root = workspace_dir.resolve()
    candidate = (root / path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path traversal outside workspace is not allowed")
    if candidate == root:
        raise ValueError("Path must refer to a file inside workspace")
    return candidate


class ReadTextFileTool(Tool):
    name = "read_text_file"
    description = "Read a UTF-8 text file from the workspace directory. Argument: path relative to workspace."
    argument_schema = ReadTextFileArguments
    risk_level = RiskLevel.READ_ONLY

    def execute(self, arguments: ReadTextFileArguments) -> str:
        path = resolve_workspace_path(arguments.path)
        if not path.is_file():
            raise FileNotFoundError(arguments.path)
        return path.read_text(encoding="utf-8")


class WriteTextFileTool(Tool):
    name = "write_text_file"
    description = "Write UTF-8 text to a file in the workspace directory. Does not overwrite unless overwrite is true."
    argument_schema = WriteTextFileArguments
    risk_level = RiskLevel.WRITE

    def execute(self, arguments: WriteTextFileArguments) -> str:
        path = resolve_workspace_path(arguments.path)
        if path.exists() and not arguments.overwrite:
            raise FileExistsError(f"{arguments.path} already exists; set overwrite=true to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments.content, encoding="utf-8")
        return f"Wrote {len(arguments.content)} characters to {arguments.path}"
