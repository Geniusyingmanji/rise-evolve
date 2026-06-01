from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


REQUIRED_EDIT_PROGRAM_FIELDS = [
    "source_scene_graph",
    "task_family",
    "knowledge_facts",
    "target_scene_description",
    "edit_operations",
    "preservation_constraints",
    "negative_constraints",
    "atomic_checklist",
    "editor_prompt",
    "failure_modes_to_watch",
]

OPTIONAL_EDIT_PROGRAM_FIELDS = [
    "task_id",
    "version",
    "split",
    "created_at",
    "reference_images",
]


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def flatten_checklist(checklist: Any) -> List[Dict[str, Any]]:
    if isinstance(checklist, list):
        return [x for x in checklist if isinstance(x, dict)]
    if not isinstance(checklist, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for category, items in checklist.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                row = dict(item)
                row.setdefault("category", category)
                rows.append(row)
    return rows


def validate_edit_program(program: Any, require_task_id: bool = False) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(program, dict):
        return ValidationResult(False, ["edit_program is not an object"], [])

    required = list(REQUIRED_EDIT_PROGRAM_FIELDS)
    if require_task_id:
        required.append("task_id")
    for field_name in required:
        if field_name not in program:
            errors.append(f"missing required field: {field_name}")

    if "edit_operations" in program and not isinstance(program["edit_operations"], list):
        errors.append("edit_operations must be a list")
    elif not program.get("edit_operations"):
        warnings.append("edit_operations is empty")

    if "knowledge_facts" in program and not isinstance(program["knowledge_facts"], list):
        errors.append("knowledge_facts must be a list")

    if "reference_images" in program and not isinstance(program.get("reference_images", []), list):
        errors.append("reference_images must be a list")

    checklist = flatten_checklist(program.get("atomic_checklist"))
    if not checklist:
        errors.append("atomic_checklist has no valid checklist items")
    else:
        total_weight = 0.0
        for item in checklist:
            if not item.get("question"):
                warnings.append(f"checklist item {item.get('id', '<unknown>')} has no question")
            try:
                total_weight += float(item.get("weight", 0))
            except (TypeError, ValueError):
                warnings.append(f"checklist item {item.get('id', '<unknown>')} has non-numeric weight")
        if total_weight and abs(total_weight - 1.0) > 1e-3:
            warnings.append(f"checklist weights sum to {total_weight:.4f}, expected 1.0")

    editor_prompt = program.get("editor_prompt")
    if editor_prompt is not None and (not isinstance(editor_prompt, str) or len(editor_prompt.strip()) < 20):
        warnings.append("editor_prompt is short or not a string")

    return ValidationResult(not errors, errors, warnings)


def extract_source_image_and_instruction(row: Dict[str, Any]) -> tuple[Optional[str], str]:
    if row.get("source_image") and row.get("instruction"):
        return str(row["source_image"]), str(row["instruction"])
    for message in row.get("messages", []):
        if message.get("role") != "user":
            continue
        image_path: Optional[str] = None
        text_parts: List[str] = []
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image":
                    image_path = part.get("path") or part.get("image") or image_path
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
        if image_path or text_parts:
            return image_path, " ".join(x.strip() for x in text_parts if x.strip())
    return None, ""


def extract_tool_trace(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    trace: List[Dict[str, Any]] = []
    for message in row.get("messages", []):
        for call in message.get("tool_calls", []) if isinstance(message, dict) else []:
            if not isinstance(call, dict):
                continue
            trace.append(
                {
                    "name": call.get("name"),
                    "arguments": call.get("arguments") or {},
                    "result": call.get("result"),
                }
            )
    return trace


def render_tool_trace_for_sft(row: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for message in row.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            chunks.append(f"<think>{content.strip()}</think>")
        for call in message.get("tool_calls", []) if isinstance(message, dict) else []:
            if not isinstance(call, dict):
                continue
            tool_call = {"name": call.get("name"), "arguments": call.get("arguments") or {}}
            chunks.append("<tool_call>" + json.dumps(tool_call, ensure_ascii=False, sort_keys=True) + "</tool_call>")
            if "result" in call:
                response = {"name": call.get("name"), "result": call.get("result")}
                chunks.append(
                    "<tool_response>" + json.dumps(response, ensure_ascii=False, sort_keys=True) + "</tool_response>"
                )
    return "\n".join(chunks)


def build_answer_text(program: Dict[str, Any]) -> str:
    return "<answer>" + json.dumps(program, ensure_ascii=False, sort_keys=True) + "</answer>"


def require_fields(row: Dict[str, Any], fields: Iterable[str]) -> List[str]:
    return [field_name for field_name in fields if field_name not in row]
