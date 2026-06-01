from __future__ import annotations

from typing import Any, Dict, List, Optional

from rise_evolve.agent.schemas import extract_tool_trace, flatten_checklist, validate_edit_program
from rise_evolve.reward.scoring import score_item, stable_item_id


def expected_diff_from_program(program: Optional[Dict[str, Any]], item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    item = item or {}
    program = program or {}
    operations = program.get("edit_operations") or item.get("edit_operations") or []
    allowed_changes: List[str] = []
    for op in operations:
        if isinstance(op, dict):
            change = op.get("change") or op.get("target") or op.get("region_hint")
            if change:
                allowed_changes.append(str(change))
    if not allowed_changes and item.get("target_description"):
        allowed_changes.append(str(item["target_description"]))

    protected = list(program.get("preservation_constraints") or item.get("preservation_constraints") or [])
    cognitive_targets = [
        fact.get("claim") for fact in program.get("knowledge_facts", []) if isinstance(fact, dict) and fact.get("claim")
    ]
    if item.get("target_description"):
        cognitive_targets.append(str(item["target_description"]))
    return {
        "allowed_changes": allowed_changes,
        "protected_content": protected,
        "cognitive_targets": cognitive_targets,
    }


def checklist_results(item: Dict[str, Any], heads: Dict[str, float]) -> List[Dict[str, Any]]:
    checklist = flatten_checklist(item.get("atomic_checklist") or item.get("edit_program", {}).get("atomic_checklist"))
    results: List[Dict[str, Any]] = []
    for entry in checklist:
        question = str(entry.get("question", "")).lower()
        if "reason" in question or "knowledge" in question or "discipline" in question:
            score = heads.get("cognitive", 0.0)
            category = "cognitive"
        elif "unrelated" in question or "preserve" in question or "background" in question:
            score = heads.get("preservation", 0.0)
            category = "preservation"
        elif "clear" in question or "artifact" in question:
            score = heads.get("quality", 0.0)
            category = "quality"
        else:
            score = heads.get("execution", 0.0)
            category = "execution"
        results.append(
            {
                "id": entry.get("id"),
                "category": entry.get("category", category),
                "question": entry.get("question"),
                "weight": entry.get("weight", 0.0),
                "score": round(float(score), 4),
                "pass": score >= 0.65,
                "evidence": "rule/programmatic scorer; replace with VLM difference-first evidence for production RL",
            }
        )
    return results


def score_verifier_item(item: Dict[str, Any], use_programmatic_priors: bool = True) -> Dict[str, Any]:
    score = score_item(item, use_programmatic_priors=use_programmatic_priors)
    expected_diff = expected_diff_from_program(item.get("edit_program") or item.get("final_edit_program"), item)
    return {
        "item_id": item.get("item_id") or stable_item_id(item),
        "task_id": item.get("task_id"),
        "label": item.get("label"),
        "score": score["score"],
        "heads": score["heads"],
        "attribution": score["attribution"],
        "expected_diff": expected_diff,
        "difference_report": {
            "intended": expected_diff["allowed_changes"] if score["heads"].get("execution", 0) >= 0.65 else [],
            "missing": expected_diff["allowed_changes"] if score["heads"].get("execution", 0) < 0.65 else [],
            "unintended": [] if score["heads"].get("preservation", 0) >= 0.65 else ["possible non-target drift"],
            "implied": [],
        },
        "checklist_results": checklist_results(item, score["heads"]),
        "image_delta": score["image_delta"],
        "evidence_modes": score["evidence_modes"],
    }


def score_agent_result(row: Dict[str, Any], use_programmatic_priors: bool = False) -> Dict[str, Any]:
    program = row.get("edit_program") or row.get("final_edit_program")
    validation = validate_edit_program(program)
    item = {
        "task_id": row.get("task_id") or row.get("sample_id"),
        "source_image": row.get("source_image"),
        "candidate_image": row.get("candidate_image") or row.get("edited_image"),
        "atomic_checklist": row.get("atomic_checklist") or (program or {}).get("atomic_checklist"),
        "target_description": row.get("target_description") or (program or {}).get("target_scene_description"),
    }
    trace = row.get("tool_trace") or extract_tool_trace(row)
    score = score_item(item, program, trace, use_programmatic_priors=use_programmatic_priors)
    expected_diff = expected_diff_from_program(program, item)
    return {
        "task_id": item["task_id"],
        "score": score["score"],
        "heads": score["heads"],
        "attribution": score["attribution"],
        "expected_diff": expected_diff,
        "difference_report": {
            "intended": expected_diff["allowed_changes"] if score["heads"].get("execution", 0) >= 0.65 else [],
            "missing": expected_diff["allowed_changes"] if score["heads"].get("execution", 0) < 0.65 else [],
            "unintended": [] if score["heads"].get("preservation", 0) >= 0.65 else ["possible non-target drift"],
            "implied": [],
        },
        "checklist_results": checklist_results(item, score["heads"]),
        "validation": validation.to_dict(),
        "image_delta": score["image_delta"],
        "evidence_modes": score["evidence_modes"],
    }
