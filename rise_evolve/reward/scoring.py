from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from rise_evolve.agent.schemas import extract_tool_trace, flatten_checklist, validate_edit_program
from rise_evolve.io import resolve_repo_path


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def average_hash(path: Path, hash_size: int = 8) -> Optional[str]:
    if not path.exists():
        return None
    img = Image.open(path).convert("L").resize((hash_size, hash_size))
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= avg else "0" for p in pixels)
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def hamming_hex(left: Optional[str], right: Optional[str]) -> Optional[int]:
    if not left or not right:
        return None
    width = max(len(left), len(right)) * 4
    return bin(int(left, 16) ^ int(right, 16)).count("1") if width else 0


def image_change_score(source_image: Optional[str], candidate_image: Optional[str]) -> Dict[str, Any]:
    if not source_image or not candidate_image:
        return {"available": False, "reason": "missing image path"}
    source_path = resolve_repo_path(source_image)
    candidate_path = resolve_repo_path(candidate_image)
    if not source_path.exists() or not candidate_path.exists():
        return {
            "available": False,
            "reason": "image file missing",
            "source_exists": source_path.exists(),
            "candidate_exists": candidate_path.exists(),
        }
    source_hash = average_hash(source_path)
    candidate_hash = average_hash(candidate_path)
    distance = hamming_hex(source_hash, candidate_hash)
    if distance is None:
        return {"available": False, "reason": "hash failed"}
    # 0 means unchanged; 64 is very different. Editing reward wants some change but not a global drift.
    normalized = distance / 64.0
    enough_change = clamp01(normalized / 0.08)
    not_global_drift = 1.0 - clamp01(max(0.0, normalized - 0.45) / 0.35)
    return {
        "available": True,
        "source_ahash": source_hash,
        "candidate_ahash": candidate_hash,
        "hamming": distance,
        "normalized": normalized,
        "enough_change": enough_change,
        "not_global_drift": not_global_drift,
    }


def render_prior(candidate_image: Optional[str]) -> Optional[str]:
    if not candidate_image:
        return None
    name = Path(candidate_image).name.lower()
    if "_teacher" in name or "teacher_render" in name:
        return "teacher"
    if "_negative" in name or "negative_render" in name:
        return "negative"
    return None


def score_program(program: Optional[Dict[str, Any]], tool_trace: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
    if not isinstance(program, dict):
        return {"program": 0.0, "format": 0.0, "tool": 0.0}
    validation = validate_edit_program(program)
    format_score = 1.0 if validation.ok else max(0.0, 0.6 - 0.1 * len(validation.errors))
    if validation.warnings:
        format_score = max(0.0, format_score - min(0.25, 0.05 * len(validation.warnings)))

    checklist = flatten_checklist(program.get("atomic_checklist"))
    has_target = bool(str(program.get("target_scene_description", "")).strip())
    has_editor_prompt = len(str(program.get("editor_prompt", "")).strip()) >= 20
    has_ops = bool(program.get("edit_operations"))
    has_preserve = bool(program.get("preservation_constraints"))
    has_negative = bool(program.get("negative_constraints"))
    has_knowledge = bool(program.get("knowledge_facts"))
    program_score = sum([has_target, has_editor_prompt, has_ops, has_preserve, has_negative, bool(checklist)]) / 6.0
    if has_knowledge:
        program_score = min(1.0, program_score + 0.08)

    trace = tool_trace or []
    tool_names = [x.get("name") for x in trace if isinstance(x, dict)]
    has_analyze = "analyze_image" in tool_names
    has_knowledge_tool = any(name in tool_names for name in ("search", "query_edit_knowledge", "solve_symbolic"))
    no_tool_loop = len(tool_names) <= 8 and len(tool_names) == len([x for x in tool_names if x])
    tool_score = 0.5 + 0.2 * has_analyze + 0.2 * has_knowledge_tool + 0.1 * no_tool_loop if trace else 0.5
    return {"program": clamp01(program_score), "format": clamp01(format_score), "tool": clamp01(tool_score)}


def fuse_reward(heads: Dict[str, float]) -> float:
    r_exec = clamp01(heads.get("execution", 0.0))
    r_cog = clamp01(heads.get("cognitive", 1.0))
    r_preserve = clamp01(heads.get("preservation", 0.0))
    r_region = clamp01(heads.get("region", 0.0))
    r_quality = clamp01(heads.get("quality", 0.0))
    r_readability = clamp01(heads.get("readability", r_quality))
    g_task = min(r_exec, r_cog)
    r_image = g_task * (0.45 + 0.20 * r_preserve + 0.15 * r_region + 0.10 * r_quality + 0.10 * r_readability)
    r_agent = (
        0.45 * clamp01(heads.get("program", 0.0))
        + 0.45 * r_image
        + 0.05 * clamp01(heads.get("tool", 0.0))
        + 0.05 * clamp01(heads.get("format", 0.0))
    )
    if heads.get("editor_fail"):
        r_agent = max(r_agent, 0.6 * clamp01(heads.get("program", 0.0)))
    return clamp01(r_agent)


def infer_failure_type(heads: Dict[str, float]) -> str:
    if heads.get("format", 1.0) < 0.5:
        return "format_fail"
    if heads.get("program", 1.0) < 0.55 or heads.get("cognitive", 1.0) < 0.45:
        return "planner_fail"
    if heads.get("region", 1.0) < 0.45:
        return "region_fail"
    if heads.get("execution", 1.0) < 0.45 and heads.get("program", 0.0) >= 0.65:
        return "editor_fail"
    if heads.get("preservation", 1.0) < 0.45:
        return "over_edit"
    if heads.get("execution", 1.0) < 0.45:
        return "under_edit"
    return "none"


def score_item(
    item: Dict[str, Any],
    edit_program: Optional[Dict[str, Any]] = None,
    tool_trace: Optional[List[Dict[str, Any]]] = None,
    use_programmatic_priors: bool = True,
) -> Dict[str, Any]:
    program = edit_program or item.get("edit_program") or item.get("final_edit_program")
    trace = tool_trace or extract_tool_trace(item)
    heads = {
        "program": 0.55,
        "cognitive": 0.5,
        "execution": 0.5,
        "preservation": 0.65,
        "region": 0.55,
        "quality": 0.65,
        "readability": 0.65,
        "tool": 0.5,
        "format": 0.5,
    }
    if isinstance(program, dict):
        heads.update(score_program(program, trace))
    else:
        required_item_fields = ["source_image", "candidate_image", "instruction", "atomic_checklist"]
        present = sum(1 for field in required_item_fields if item.get(field))
        heads["format"] = max(heads["format"], present / len(required_item_fields))
        heads["program"] = max(heads["program"], 0.60)

    source_image = item.get("source_image")
    candidate_image = item.get("candidate_image") or item.get("edited_image") or item.get("image")
    image_delta = image_change_score(source_image, candidate_image)
    if image_delta.get("available"):
        heads["execution"] = max(heads["execution"], 0.35 + 0.45 * image_delta["enough_change"])
        heads["preservation"] = min(heads["preservation"], image_delta["not_global_drift"])
        heads["region"] = min(0.85, 0.45 + 0.45 * image_delta["not_global_drift"])

    evidence_modes = ["schema_rules"]
    prior = render_prior(candidate_image)
    if use_programmatic_priors and prior:
        evidence_modes.append(f"programmatic_render_prior:{prior}")
        if prior == "teacher":
            heads.update(
                {
                    "cognitive": max(heads["cognitive"], 0.95),
                    "execution": max(heads["execution"], 0.95),
                    "preservation": max(heads["preservation"], 0.9),
                    "region": max(heads["region"], 0.9),
                    "quality": max(heads["quality"], 0.9),
                    "readability": max(heads["readability"], 0.9),
                }
            )
        if prior == "negative":
            failure_type = item.get("failure_type") or "reasoning_or_region_error"
            heads["cognitive"] = min(heads["cognitive"], 0.25)
            heads["execution"] = min(heads["execution"], 0.25)
            if "region" in failure_type:
                heads["region"] = min(heads["region"], 0.25)
            heads["quality"] = min(heads["quality"], 0.65)

    heads = {key: clamp01(value) for key, value in heads.items()}
    primary_failure = infer_failure_type(heads)
    score = fuse_reward(heads)
    return {
        "score": score,
        "heads": heads,
        "attribution": {
            "primary_failure": primary_failure,
            "token_credit_hints": token_credit_hints(heads, primary_failure),
        },
        "image_delta": image_delta,
        "evidence_modes": evidence_modes,
    }


def token_credit_hints(heads: Dict[str, float], failure_type: str) -> List[Dict[str, str]]:
    hints: List[Dict[str, str]] = []
    if heads.get("tool", 1.0) < 0.6:
        hints.append({"token_group": "tool_call", "reason": "tool evidence is missing, redundant, or unused"})
    if heads.get("cognitive", 1.0) < 0.6 or failure_type == "planner_fail":
        hints.append({"token_group": "reasoning", "reason": "target reasoning or knowledge binding is weak"})
    if heads.get("region", 1.0) < 0.6:
        hints.append({"token_group": "region", "reason": "edit region or locality is likely wrong"})
    if heads.get("preservation", 1.0) < 0.6 or failure_type == "over_edit":
        hints.append({"token_group": "preservation", "reason": "non-target preservation is weak"})
    if heads.get("execution", 1.0) < 0.6:
        hints.append({"token_group": "editor_prompt", "reason": "requested visual change was not executed"})
    return hints


def stable_item_id(obj: Dict[str, Any]) -> str:
    text = repr(sorted(obj.items())).encode("utf-8")
    return hashlib.sha256(text).hexdigest()[:16]
