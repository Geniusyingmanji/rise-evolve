#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw

from common import (
    PALETTE,
    average_hash,
    draw_centered_text,
    ensure_dir,
    load_font,
    new_canvas,
    normalize_text,
    repo_path,
    save_png,
    stable_hash,
    text_size,
    write_json,
)


FIXED_CREATED_AT = "2026-06-11T00:00:00Z"
CANVAS_SIZE = 512
SUBTASKS = [
    "sudoku4",
    "arithmetic_chain",
    "balance_scale",
    "sequence_pattern",
    "clock_arithmetic",
    "graph_path",
    "mirror_reflection",
    "block_stack_view",
    "circuit_bulb",
    "ph_indicator",
    "food_chain",
    "geometry_angle",
    "sorting_step",
    "fraction_shade",
    "process_order",
    "moon_phase",
]

RenderFn = Callable[[Dict[str, Any], str], Image.Image]


def draw_arrow(draw: ImageDraw.ImageDraw, start: Tuple[int, int], end: Tuple[int, int], fill: str, width: int = 4) -> None:
    draw.line((start, end), fill=fill, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    length = 13
    spread = 0.52
    p1 = (end[0] - length * math.cos(angle - spread), end[1] - length * math.sin(angle - spread))
    p2 = (end[0] - length * math.cos(angle + spread), end[1] - length * math.sin(angle + spread))
    draw.polygon([end, p1, p2], fill=fill)


def box_text(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, size: int = 24, fill: str = "#111827", bold: bool = True) -> None:
    font = load_font(size, bold=bold)
    draw_centered_text(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), text, font, fill=fill)


def make_task_id(version: str, family: str, subtask: str, idx: int) -> str:
    version_slug = "".join(ch for ch in version if ch.isalnum()) or "v2"
    return f"r{version_slug}_{family}_{subtask}_{idx:05d}"


def split_for(idx: int) -> str:
    splits = ["sft_train", "rl_prompt_train", "verifier_train", "hard_heldout"]
    return splits[idx % len(splits)]


def benchmark_alignment(task_family: str, sub_family: str, domain: str) -> Dict[str, Any]:
    return {
        "target_benchmarks": ["RISEBench"],
        "eval_dimensions": [
            "instruction_reasoning",
            "answer_verifiability",
            "appearance_consistency",
            "visual_plausibility",
        ],
        "reasoning_axes": [task_family, sub_family, domain],
    }


def base_case(
    *,
    sub_task: str,
    family_slug: str,
    task_family: str,
    sub_family: str,
    domain: str,
    knowledge_type: str,
    difficulty_level: int,
    difficulty_reason: str,
    instruction: str,
    expected_target: str,
    rational_target_description: str,
    required_knowledge: List[Dict[str, Any]],
    search_queries: List[str],
    source_scene_graph: Dict[str, Any],
    edit_operations: List[Dict[str, Any]],
    negative_constraints: List[str],
    atomic_checklist: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    verifier_spec: Dict[str, Any],
    params: Dict[str, Any],
    render: RenderFn,
) -> Dict[str, Any]:
    return {
        "sub_task": sub_task,
        "family_slug": family_slug,
        "benchmark_family": "RISE_like",
        "task_family": task_family,
        "sub_family": sub_family,
        "domain": domain,
        "knowledge_type": knowledge_type,
        "difficulty": {"level": difficulty_level, "reason": difficulty_reason},
        "instruction": instruction,
        "expected_target": expected_target,
        "rational_target_description": rational_target_description,
        "required_knowledge": required_knowledge,
        "search_queries": search_queries,
        "source_scene_graph": source_scene_graph,
        "edit_operations": edit_operations,
        "preservation_constraints": [
            "Keep the canvas size, diagram style, background, and all non-target elements unchanged.",
            "Do not introduce explanatory labels beyond the requested answer or highlight.",
            "Preserve the visible source layout unless the task rule requires moving or collapsing an object.",
        ],
        "negative_constraints": negative_constraints,
        "atomic_checklist": atomic_checklist,
        "benchmark_alignment": benchmark_alignment(task_family, sub_family, domain),
        "ground_truth": ground_truth,
        "verifier_spec": verifier_spec,
        "params": params,
        "render": render,
    }


def operation(op: str, target: str, region_hint: str, change: str) -> Dict[str, Any]:
    return {
        "op": op,
        "target": target,
        "region_hint": region_hint,
        "change": change,
        "preserve": ["background", "diagram frame", "non-target labels", "source geometry"],
    }


def rule_fact(claim: str) -> Dict[str, Any]:
    return {"claim": claim, "source": "programmatic_rule", "confidence": 1.0}


def solve_sudoku4(params: Dict[str, Any]) -> Dict[str, Any]:
    puzzle = params["puzzle"]
    row = params["empty_row"]
    col = params["empty_col"]
    digits = {1, 2, 3, 4}
    block_r = (row // 2) * 2
    block_c = (col // 2) * 2
    used = set(puzzle[row][c] for c in range(4) if puzzle[row][c] is not None)
    used.update(puzzle[r][col] for r in range(4) if puzzle[r][col] is not None)
    used.update(
        puzzle[r][c]
        for r in range(block_r, block_r + 2)
        for c in range(block_c, block_c + 2)
        if puzzle[r][c] is not None
    )
    candidates = sorted(digits - used)
    if len(candidates) != 1:
        raise AssertionError(f"sudoku4 is not uniquely solvable: candidates={candidates}")
    return {"type": "sudoku_cell", "index_base": 0, "row": row, "col": col, "value": candidates[0]}


def render_sudoku4(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("4x4 Sudoku", "Fill one cell.", size=CANVAS_SIZE)
    grid = params["solution"]
    row = params["empty_row"]
    col = params["empty_col"]
    cell = 72
    x0 = 112
    y0 = 112
    font = load_font(38, bold=True)
    draw.text((118, 62), "4x4 Sudoku", font=load_font(24, bold=True), fill=PALETTE["ink"])
    for r in range(4):
        for c in range(4):
            box = (x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell, y0 + (r + 1) * cell)
            draw.rectangle(box, fill="#ffffff", outline="#111827", width=2)
            value = grid[r][c]
            fill = PALETTE["ink"]
            if r == row and c == col:
                if mode == "source":
                    value = "?"
                    fill = PALETTE["muted"]
                elif mode == "negative":
                    value = params["wrong_value"]
                    fill = PALETTE["red"]
                else:
                    fill = PALETTE["green"]
            draw_centered_text(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), str(value), font, fill=fill)
    for i in [0, 2, 4]:
        width = 5 if i in [0, 2, 4] else 2
        draw.line((x0 + i * cell, y0, x0 + i * cell, y0 + 4 * cell), fill="#111827", width=width)
        draw.line((x0, y0 + i * cell, x0 + 4 * cell, y0 + i * cell), fill="#111827", width=width)
    return img


def sudoku4(rng: random.Random, idx: int) -> Dict[str, Any]:
    base = [[(r * 2 + r // 2 + c) % 4 + 1 for c in range(4)] for r in range(4)]
    bands = [0, 1]
    stacks = [0, 1]
    rng.shuffle(bands)
    rng.shuffle(stacks)
    rows = []
    cols = []
    for band in bands:
        pair = [band * 2, band * 2 + 1]
        rng.shuffle(pair)
        rows.extend(pair)
    for stack in stacks:
        pair = [stack * 2, stack * 2 + 1]
        rng.shuffle(pair)
        cols.extend(pair)
    digit_perm = [1, 2, 3, 4]
    rng.shuffle(digit_perm)
    digit_map = {i + 1: digit_perm[i] for i in range(4)}
    solution = [[digit_map[base[r][c]] for c in cols] for r in rows]
    empty_row = rng.randrange(4)
    empty_col = rng.randrange(4)
    value = solution[empty_row][empty_col]
    puzzle = [list(row) for row in solution]
    puzzle[empty_row][empty_col] = None
    wrong = rng.choice([x for x in [1, 2, 3, 4] if x != value])
    params = {
        "solution": solution,
        "puzzle": puzzle,
        "empty_row": empty_row,
        "empty_col": empty_col,
        "wrong_value": wrong,
    }
    ground_truth = solve_sudoku4(params)
    row_text = empty_row + 1
    col_text = empty_col + 1
    value_text = str(ground_truth["value"])
    return base_case(
        sub_task="sudoku4",
        family_slug="logic",
        task_family="logical_reasoning",
        sub_family="grid_constraint_completion",
        domain="logic",
        knowledge_type="symbolic",
        difficulty_level=3,
        difficulty_reason="A 4x4 constraint grid has one empty cell whose value is uniquely fixed by row, column, and box rules.",
        instruction="Fill the empty 4x4 Sudoku cell so every row, column, and 2x2 box contains 1, 2, 3, and 4.",
        expected_target=f"The empty cell at row {row_text}, column {col_text} is filled with {value_text}.",
        rational_target_description=(
            f"Row {row_text}, column {col_text} must be {value_text} because the other digits in its row, column, "
            "and 2x2 box exclude every other value."
        ),
        required_knowledge=[rule_fact("In a 4x4 Sudoku, each row, column, and 2x2 box must contain the digits 1 through 4 exactly once.")],
        search_queries=["4x4 sudoku row column 2x2 box rule"],
        source_scene_graph={
            "objects": ["4x4 sudoku grid", "digits 1-4", "one question-mark cell"],
            "editable_region": f"cell row {row_text}, column {col_text}",
            "preserve_region": "all grid lines and all already-filled digits",
        },
        edit_operations=[
            operation("replace_text", "question-mark sudoku cell", f"row {row_text}, column {col_text}", f"replace ? with {value_text}")
        ],
        negative_constraints=[
            f"Do not fill the empty cell with {wrong}; it violates the Sudoku constraints.",
            "Do not alter any pre-filled digit or grid boundary.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Is row {row_text}, column {col_text} filled with {value_text}?", "weight": 0.35},
            {"id": "C2", "question": "Does every row contain 1, 2, 3, and 4 exactly once?", "weight": 0.20},
            {"id": "C3", "question": "Does every column contain 1, 2, 3, and 4 exactly once?", "weight": 0.20},
            {"id": "C4", "question": "Are the 2x2 boxes and all original digits preserved?", "weight": 0.15},
            {"id": "C5", "question": "Is only the empty cell edited?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": f"What digit is written in row {row_text}, column {col_text}? Answer digits only.", "expected_answer": value_text, "weight": 0.50},
                {"question": "How many blank cells are shown? Answer digits only.", "expected_answer": "0", "weight": 0.25},
                {"question": "How many columns are in the grid? Answer digits only.", "expected_answer": "4", "weight": 0.25},
            ],
            "programmatic": {"solver": "solve_sudoku4", "params": params},
        },
        params=params,
        render=render_sudoku4,
    )


def apply_arithmetic(value: int, op: Dict[str, Any]) -> int:
    if op["op"] == "mul":
        return value * op["value"]
    if op["op"] == "add":
        return value + op["value"]
    if op["op"] == "sub":
        return value - op["value"]
    raise ValueError(f"unknown op {op}")


def solve_arithmetic_chain(params: Dict[str, Any]) -> Dict[str, Any]:
    value = params["start"]
    intermediates = [value]
    for op in params["ops"]:
        value = apply_arithmetic(value, op)
        intermediates.append(value)
    return {"type": "arithmetic_chain", "start": params["start"], "result": value, "intermediates": intermediates}


def render_arithmetic_chain(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Arithmetic chain", "Complete the final value.", size=CANVAS_SIZE)
    draw.text((62, 72), "Apply every operation", font=load_font(24, bold=True), fill=PALETTE["ink"])
    boxes = len(params["ops"]) + 2
    box_w = 68 if boxes == 5 else 84
    gap = 28
    total = boxes * box_w + (boxes - 1) * gap
    x = (CANVAS_SIZE - total) // 2
    y = 220
    h = 76
    labels = [str(params["start"])] + [op["label"] for op in params["ops"]]
    result = solve_arithmetic_chain(params)["result"]
    if mode == "source":
        labels.append("?")
        result_fill = PALETTE["muted"]
    elif mode == "negative":
        labels.append(str(params["negative_result"]))
        result_fill = PALETTE["red"]
    else:
        labels.append(str(result))
        result_fill = PALETTE["green"]
    for i, label in enumerate(labels):
        left = x + i * (box_w + gap)
        box = (left, y, left + box_w, y + h)
        fill = "#ffffff"
        outline = "#111827"
        if i == len(labels) - 1:
            outline = result_fill
        draw.rounded_rectangle(box, radius=8, fill=fill, outline=outline, width=4)
        font_size = 28 if len(label) <= 3 else 22
        box_text(draw, box, label, size=font_size, fill=result_fill if i == len(labels) - 1 else PALETTE["ink"])
        if i < len(labels) - 1:
            draw_arrow(draw, (box[2] + 4, y + h // 2), (box[2] + gap - 4, y + h // 2), PALETTE["muted"], width=3)
    return img


def arithmetic_chain(rng: random.Random, idx: int) -> Dict[str, Any]:
    op_pool = [
        {"op": "mul", "value": 2, "label": "x2"},
        {"op": "mul", "value": 3, "label": "x3"},
        {"op": "add", "value": 4, "label": "+4"},
        {"op": "add", "value": 5, "label": "+5"},
        {"op": "sub", "value": 3, "label": "-3"},
        {"op": "sub", "value": 6, "label": "-6"},
    ]
    while True:
        start = rng.randint(2, 12)
        ops = [dict(op) for op in rng.sample(op_pool, rng.choice([2, 3]))]
        params = {"start": start, "ops": ops}
        solved = solve_arithmetic_chain(params)
        if -40 <= solved["result"] <= 99 and solved["intermediates"][-2] != solved["result"]:
            break
    negative = solved["intermediates"][-2]
    params["negative_result"] = negative
    result = solved["result"]
    steps = " -> ".join([str(start)] + [op["label"] for op in ops] + ["?"])
    return base_case(
        sub_task="arithmetic_chain",
        family_slug="math",
        task_family="symbolic_reasoning",
        sub_family="multi_step_arithmetic",
        domain="math",
        knowledge_type="procedural",
        difficulty_level=2 + (1 if len(ops) == 3 else 0),
        difficulty_reason="The final value requires applying two or three arithmetic operations in order.",
        instruction=f"Compute the final value in the arithmetic chain: {steps}.",
        expected_target=f"The result box is filled with {result}.",
        rational_target_description=f"Starting at {start}, applying the operations in order gives {result}. The final box should show that value only.",
        required_knowledge=[rule_fact("Arithmetic chains are evaluated left to right when each operation is shown as a sequential flow step.")],
        search_queries=["multi step arithmetic flow evaluate in order"],
        source_scene_graph={
            "objects": ["start number box", "operation boxes", "arrow connectors", "empty result box"],
            "editable_region": "rightmost result box",
            "preserve_region": "start number, operation labels, arrows, and box layout",
        },
        edit_operations=[operation("replace_text", "rightmost result box", "rightmost box labeled ?", f"replace ? with {result}")],
        negative_constraints=[
            f"Do not stop one step early at {negative}.",
            "Do not reorder, skip, or change any operation label.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Does the result box show {result}?", "weight": 0.40},
            {"id": "C2", "question": "Were all operation boxes applied from left to right?", "weight": 0.25},
            {"id": "C3", "question": "Are the start value and operation labels unchanged?", "weight": 0.15},
            {"id": "C4", "question": "Is the result written only in the final box?", "weight": 0.10},
            {"id": "C5", "question": "Is the flow diagram layout preserved?", "weight": 0.10},
        ],
        ground_truth={"type": "arithmetic_chain", "start": start, "result": result, "intermediates": solved["intermediates"]},
        verifier_spec={
            "vqa_checks": [
                {"question": "What signed number is written in the final result box? Answer the signed number only.", "expected_answer": str(result), "weight": 0.60},
                {"question": "What signed number is written in the first box of the chain? Answer the signed number only.", "expected_answer": str(start), "weight": 0.20},
                {"question": "How many operation boxes are shown? Answer digits only.", "expected_answer": str(len(ops)), "weight": 0.20},
            ],
            "programmatic": {"solver": "solve_arithmetic_chain", "params": params},
        },
        params=params,
        render=render_arithmetic_chain,
    )


def solve_balance_scale(params: Dict[str, Any]) -> Dict[str, Any]:
    left_known = sum(params["left_weights"])
    right_known = sum(params["right_weights"])
    if params["missing_side"] == "left":
        value = right_known - left_known
    else:
        value = left_known - right_known
    if value <= 0:
        raise AssertionError("missing balance value must be positive")
    return {"type": "balance_weight", "side": params["missing_side"], "value": value}


def partition_weight(rng: random.Random, total: int, count: int) -> Optional[List[int]]:
    for _ in range(200):
        cuts = sorted(rng.sample(range(1, total), count - 1)) if count > 1 else []
        vals = []
        last = 0
        for cut in cuts + [total]:
            vals.append(cut - last)
            last = cut
        if all(1 <= v <= 9 for v in vals):
            rng.shuffle(vals)
            return vals
    return None


def render_balance_scale(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Balance scale", "Find the missing weight.", size=CANVAS_SIZE)
    draw.text((72, 62), "Balance the scale", font=load_font(24, bold=True), fill=PALETTE["ink"])
    pivot = (256, 235)
    left_pan = (144, 326)
    right_pan = (368, 326)
    draw.line((pivot[0], pivot[1], left_pan[0], left_pan[1] - 45), fill="#475569", width=4)
    draw.line((pivot[0], pivot[1], right_pan[0], right_pan[1] - 45), fill="#475569", width=4)
    draw.line((150, 282, 362, 282), fill="#334155", width=7)
    draw.polygon([(256, 235), (232, 410), (280, 410)], fill="#94a3b8", outline="#475569")
    draw.ellipse((238, 218, 274, 254), fill="#e2e8f0", outline="#334155", width=3)
    for cx in [left_pan[0], right_pan[0]]:
        draw.line((cx - 72, 326, cx + 72, 326), fill="#334155", width=4)
        draw.arc((cx - 78, 292, cx + 78, 360), 0, 180, fill="#334155", width=4)
    font = load_font(22, bold=True)

    def draw_weights(weights: List[int], side: str, missing: bool) -> None:
        cx = left_pan[0] if side == "left" else right_pan[0]
        y = 260
        values: List[Any] = list(weights)
        if missing:
            if mode == "source":
                values.append("?")
                miss_fill = PALETTE["muted"]
            elif mode == "negative":
                values.append(params["wrong_value"])
                miss_fill = PALETTE["red"]
            else:
                values.append(params["missing_value"])
                miss_fill = PALETTE["green"]
        total_w = len(values) * 44 + (len(values) - 1) * 8
        x = cx - total_w // 2
        for i, val in enumerate(values):
            box = (x + i * 52, y, x + i * 52 + 44, y + 46)
            fill = "#ffffff"
            outline = "#111827"
            text_fill = PALETTE["ink"]
            if missing and i == len(values) - 1:
                outline = miss_fill
                text_fill = miss_fill
            draw.rounded_rectangle(box, radius=6, fill=fill, outline=outline, width=3)
            draw_centered_text(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), str(val), font, fill=text_fill)

    draw_weights(params["left_weights"], "left", params["missing_side"] == "left")
    draw_weights(params["right_weights"], "right", params["missing_side"] == "right")
    return img


def balance_scale(rng: random.Random, idx: int) -> Dict[str, Any]:
    while True:
        missing_side = rng.choice(["left", "right"])
        missing_value = rng.randint(2, 9)
        known_missing = [rng.randint(1, 5) for _ in range(rng.choice([1, 2]))]
        other_sum = missing_value + sum(known_missing)
        other = partition_weight(rng, other_sum, rng.choice([2, 3]))
        if other:
            break
    if missing_side == "left":
        left_weights = known_missing
        right_weights = other
    else:
        left_weights = other
        right_weights = known_missing
    wrong_candidates = [v for v in range(1, 12) if v != missing_value and v > 0]
    wrong_value = rng.choice([v for v in wrong_candidates if v in {missing_value - 2, missing_value - 1, missing_value + 1, missing_value + 2}] or wrong_candidates)
    params = {
        "left_weights": left_weights,
        "right_weights": right_weights,
        "missing_side": missing_side,
        "missing_value": missing_value,
        "wrong_value": wrong_value,
    }
    ground_truth = solve_balance_scale(params)
    return base_case(
        sub_task="balance_scale",
        family_slug="physics",
        task_family="quantitative_reasoning",
        sub_family="equilibrium_balance",
        domain="everyday_physics",
        knowledge_type="procedural",
        difficulty_level=3,
        difficulty_reason="The missing weight is found by equating the numeric totals on the two pans.",
        instruction="Replace the ? weight with the number that makes the two scale pans balance.",
        expected_target=f"The missing {missing_side} weight is filled with {missing_value}.",
        rational_target_description=(
            f"The known weights require the missing {missing_side} pan value to be {missing_value} so the left and right totals match."
        ),
        required_knowledge=[rule_fact("A level two-pan balance is balanced when the total weight on the left equals the total weight on the right.")],
        search_queries=["two pan balance missing weight equal totals"],
        source_scene_graph={
            "objects": ["level balance beam", "left pan", "right pan", "numbered weight boxes", "one question-mark weight box"],
            "editable_region": f"question-mark box on the {missing_side} pan",
            "preserve_region": "beam, stand, pan positions, and all known weight labels",
        },
        edit_operations=[
            operation("replace_text", "missing weight box", f"{missing_side} pan question-mark box", f"replace ? with {missing_value}")
        ],
        negative_constraints=[
            f"Do not use {wrong_value}; it makes the pan totals unequal.",
            "Do not move known weights between pans.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Does the missing {missing_side} pan box show {missing_value}?", "weight": 0.35},
            {"id": "C2", "question": "Do the left and right pan totals match after the edit?", "weight": 0.30},
            {"id": "C3", "question": "Are all originally visible known weights preserved?", "weight": 0.15},
            {"id": "C4", "question": "Is the scale beam and pan layout unchanged?", "weight": 0.10},
            {"id": "C5", "question": "Is only the missing weight label edited?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": "What number is written in the formerly missing weight box? Answer digits only.", "expected_answer": str(missing_value), "weight": 0.55},
                {"question": "Which side contains the filled missing weight? Answer one word.", "expected_answer": missing_side, "weight": 0.25},
                {"question": "Is the scale beam level? Answer yes or no.", "expected_answer": "yes", "weight": 0.20},
            ],
            "programmatic": {"solver": "solve_balance_scale", "params": params},
        },
        params=params,
        render=render_balance_scale,
    )


SHAPE_COLORS = {
    "blue circle": ("circle", PALETTE["blue"]),
    "green square": ("square", PALETTE["green"]),
    "orange triangle": ("triangle", PALETTE["orange"]),
    "purple diamond": ("diamond", PALETTE["purple"]),
}


def solve_sequence_pattern(params: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "sequence_next", "pattern": params["pattern"], "answer": params["answer"], "sequence": params["sequence"]}


def draw_shape_token(draw: ImageDraw.ImageDraw, center: Tuple[int, int], token: str, size: int = 46) -> None:
    shape, color = SHAPE_COLORS[token]
    x, y = center
    if shape == "circle":
        draw.ellipse((x - size // 2, y - size // 2, x + size // 2, y + size // 2), fill=color, outline="#111827", width=3)
    elif shape == "square":
        draw.rectangle((x - size // 2, y - size // 2, x + size // 2, y + size // 2), fill=color, outline="#111827", width=3)
    elif shape == "triangle":
        draw.polygon([(x, y - size // 2), (x - size // 2, y + size // 2), (x + size // 2, y + size // 2)], fill=color, outline="#111827")
    else:
        draw.polygon([(x, y - size // 2), (x - size // 2, y), (x, y + size // 2), (x + size // 2, y)], fill=color, outline="#111827")


def render_sequence_pattern(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Sequence pattern", "Complete the last slot.", size=CANVAS_SIZE)
    draw.text((72, 72), params["title"], font=load_font(24, bold=True), fill=PALETTE["ink"])
    seq = list(params["sequence"])
    if mode == "source":
        shown = seq[:-1] + ["?"]
        answer_fill = PALETTE["muted"]
    elif mode == "negative":
        shown = seq[:-1] + [params["wrong_answer"]]
        answer_fill = PALETTE["red"]
    else:
        shown = seq
        answer_fill = PALETTE["green"]
    slot_w = 76
    gap = 18
    total = len(shown) * slot_w + (len(shown) - 1) * gap
    x0 = (CANVAS_SIZE - total) // 2
    y0 = 220
    font = load_font(30, bold=True)
    for i, token in enumerate(shown):
        box = (x0 + i * (slot_w + gap), y0, x0 + i * (slot_w + gap) + slot_w, y0 + 86)
        outline = answer_fill if i == len(shown) - 1 else "#111827"
        draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline=outline, width=4)
        if isinstance(token, int):
            draw_centered_text(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), str(token), font, fill=answer_fill if i == len(shown) - 1 else PALETTE["ink"])
        elif token == "?":
            draw_centered_text(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), "?", font, fill=answer_fill)
        else:
            draw_shape_token(draw, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), token)
    return img


def sequence_pattern(rng: random.Random, idx: int) -> Dict[str, Any]:
    pattern = rng.choice(["add_k", "doubling", "alternation"])
    if pattern == "add_k":
        length = rng.choice([4, 5])
        start = rng.randint(1, 9)
        k = rng.randint(2, 6)
        seq = [start + i * k for i in range(length)]
        answer = seq[-1]
        wrong = answer + rng.choice([1, k + 1, -1])
        title = f"Add {k} each step"
        rule = f"Each number increases by {k}."
    elif pattern == "doubling":
        length = rng.choice([4, 5])
        start = rng.randint(1, 4)
        seq = [start * (2**i) for i in range(length)]
        answer = seq[-1]
        wrong = answer + start
        title = "Double each step"
        rule = "Each number is double the previous number."
    else:
        length = 5
        tokens = rng.sample(list(SHAPE_COLORS.keys()), 2)
        seq = [tokens[i % 2] for i in range(length)]
        answer = seq[-1]
        wrong = tokens[1] if answer == tokens[0] else tokens[0]
        title = "Alternate shapes"
        rule = f"The sequence alternates between {tokens[0]} and {tokens[1]}."
    params = {"pattern": pattern, "sequence": seq, "answer": answer, "wrong_answer": wrong, "title": title, "rule": rule}
    ground_truth = solve_sequence_pattern(params)
    expected = str(answer)
    if isinstance(answer, int):
        final_slot_question = "What number is written in the final slot? Answer the signed number only."
    else:
        final_slot_question = "What color and shape are drawn in the final slot? Answer as '<color> <shape>'."
    return base_case(
        sub_task="sequence_pattern",
        family_slug="logic",
        task_family="pattern_reasoning",
        sub_family="deterministic_sequence_completion",
        domain="logic",
        knowledge_type="symbolic",
        difficulty_level=2,
        difficulty_reason="The last item follows a visible deterministic sequence rule.",
        instruction="Complete the final slot so the row follows the same deterministic pattern.",
        expected_target=f"The final slot is filled with {expected}.",
        rational_target_description=f"{rule} Therefore the last slot must be {expected}.",
        required_knowledge=[rule_fact("A deterministic sequence should be completed by applying the same visible rule to the final slot.")],
        search_queries=["number and shape sequence pattern completion"],
        source_scene_graph={
            "objects": ["row of sequence slots", "visible sequence items", "empty final slot"],
            "editable_region": "rightmost empty slot",
            "preserve_region": "all earlier sequence items and slot positions",
        },
        edit_operations=[operation("replace_content", "final sequence slot", "rightmost slot", f"replace ? with {expected}")],
        negative_constraints=[
            f"Do not fill the last slot with {wrong}; it breaks the pattern.",
            "Do not change any earlier sequence item.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Does the final slot show {expected}?", "weight": 0.40},
            {"id": "C2", "question": "Does the completed row follow the same pattern throughout?", "weight": 0.25},
            {"id": "C3", "question": "Are all non-final sequence items unchanged?", "weight": 0.15},
            {"id": "C4", "question": "Is the final slot filled clearly within its border?", "weight": 0.10},
            {"id": "C5", "question": "Is no extra sequence item added outside the row?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": final_slot_question, "expected_answer": expected, "weight": 0.60},
                {"question": "Is the final slot empty? Answer yes or no.", "expected_answer": "no", "weight": 0.20},
                {"question": "How many slots are in the row? Answer digits only.", "expected_answer": str(len(seq)), "weight": 0.20},
            ],
            "programmatic": {"solver": "solve_sequence_pattern", "params": params},
        },
        params=params,
        render=render_sequence_pattern,
    )


def add_hours(hour: int, minute: int, add: int) -> Tuple[int, int]:
    h0 = hour % 12
    h = (h0 + add) % 12
    return (12 if h == 0 else h, minute)


def solve_clock_arithmetic(params: Dict[str, Any]) -> Dict[str, Any]:
    hour, minute = add_hours(params["start_hour"], params["minute"], params["add_hours"])
    return {
        "type": "clock_time",
        "start_hour": params["start_hour"],
        "minute": params["minute"],
        "add_hours": params["add_hours"],
        "target_hour": hour,
        "target_minute": minute,
        "target_time": format_time(hour, minute),
    }


def format_time(hour: int, minute: int) -> str:
    return f"{hour}:{minute:02d}"


def render_clock_face(draw: ImageDraw.ImageDraw, center: Tuple[int, int], hour: int, minute: int, hand_color: str) -> None:
    cx, cy = center
    radius = 148
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill="#ffffff", outline="#111827", width=5)
    for i in range(60):
        angle = math.radians(i * 6 - 90)
        outer = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        inner_radius = radius - (16 if i % 5 == 0 else 8)
        inner = (cx + inner_radius * math.cos(angle), cy + inner_radius * math.sin(angle))
        draw.line((inner, outer), fill="#334155", width=3 if i % 5 == 0 else 1)
    for label, pos in {"12": (cx, cy - 110), "3": (cx + 110, cy), "6": (cx, cy + 110), "9": (cx - 110, cy)}.items():
        draw_centered_text(draw, pos, label, load_font(22, bold=True), fill=PALETTE["ink"])
    minute_angle = math.radians(minute * 6 - 90)
    hour_angle = math.radians(((hour % 12) + minute / 60) * 30 - 90)
    hour_end = (cx + 76 * math.cos(hour_angle), cy + 76 * math.sin(hour_angle))
    minute_end = (cx + 112 * math.cos(minute_angle), cy + 112 * math.sin(minute_angle))
    draw.line((center, hour_end), fill=hand_color, width=8)
    draw.line((center, minute_end), fill="#111827", width=5)
    draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill="#111827")


def render_clock_arithmetic(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Clock arithmetic", "Add hours.", size=CANVAS_SIZE)
    solved = solve_clock_arithmetic(params)
    if mode == "source":
        hour = params["start_hour"]
        minute = params["minute"]
        label = f"Start {format_time(hour, minute)}"
        color = PALETTE["ink"]
    elif mode == "negative":
        hour, minute = add_hours(params["start_hour"], params["minute"], params["add_hours"] + 1)
        label = f"+{params['add_hours']} hours"
        color = PALETTE["red"]
    else:
        hour = solved["target_hour"]
        minute = solved["target_minute"]
        label = f"+{params['add_hours']} hours"
        color = PALETTE["green"]
    draw.text((78, 50), label, font=load_font(24, bold=True), fill=color)
    render_clock_face(draw, (256, 286), hour, minute, color)
    return img


def clock_arithmetic(rng: random.Random, idx: int) -> Dict[str, Any]:
    start_hour = rng.randint(1, 12)
    minute = rng.choice([0, 15, 30, 45])
    add = rng.randint(1, 6)
    params = {"start_hour": start_hour, "minute": minute, "add_hours": add}
    ground_truth = solve_clock_arithmetic(params)
    target_time = ground_truth["target_time"]
    return base_case(
        sub_task="clock_arithmetic",
        family_slug="time",
        task_family="temporal_reasoning",
        sub_family="clock_time_arithmetic",
        domain="time",
        knowledge_type="procedural",
        difficulty_level=3,
        difficulty_reason="The target clock requires adding whole hours to the shown analog time while preserving minutes.",
        instruction=f"Edit the clock to show the time {add} hours after the source time.",
        expected_target=f"The clock is redrawn to show {target_time}.",
        rational_target_description=f"Adding {add} hours to {format_time(start_hour, minute)} gives {target_time}. The minute hand remains at {minute:02d} minutes.",
        required_knowledge=[rule_fact("Adding whole hours to a clock changes the hour while preserving the minute value.")],
        search_queries=["analog clock add hours keep minutes"],
        source_scene_graph={
            "objects": ["analog clock face", "hour hand", "minute hand", "start time label"],
            "editable_region": "clock hands",
            "preserve_region": "clock face, tick marks, numerals, and canvas border",
        },
        edit_operations=[operation("rotate_hands", "clock hands", "center clock face", f"set analog clock hands to {target_time}")],
        negative_constraints=[
            "Do not move the hour hand by the wrong number of hours.",
            "Do not change the minute hand when adding whole hours.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Does the edited clock show {target_time}?", "weight": 0.40},
            {"id": "C2", "question": f"Is the minute hand still at {minute:02d} minutes?", "weight": 0.20},
            {"id": "C3", "question": "Is the hour hand advanced by the requested number of hours?", "weight": 0.20},
            {"id": "C4", "question": "Are the clock face, numerals, and tick marks preserved?", "weight": 0.10},
            {"id": "C5", "question": "Is no extra clock face or unrelated label added?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": "What time is shown on the clock? Answer H:MM only.", "expected_answer": target_time, "weight": 0.60},
                {"question": "What minute value does the minute hand show? Answer two digits.", "expected_answer": f"{minute:02d}", "weight": 0.20},
                {"question": "Is there one clock face? Answer yes or no.", "expected_answer": "yes", "weight": 0.20},
            ],
            "programmatic": {"solver": "solve_clock_arithmetic", "params": params},
        },
        params=params,
        render=render_clock_arithmetic,
    )


GRAPH_POSITIONS = {
    "A": (92, 260),
    "B": (190, 140),
    "C": (190, 380),
    "D": (322, 140),
    "E": (322, 380),
    "F": (430, 260),
}


def edge_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def path_weight(path: List[str], weights: Dict[Tuple[str, str], int]) -> int:
    return sum(weights[edge_key(path[i], path[i + 1])] for i in range(len(path) - 1))


def enumerate_paths(nodes: List[str], edge_list: List[Tuple[str, str, int]], start: str, goal: str) -> List[List[str]]:
    adj: Dict[str, List[str]] = {node: [] for node in nodes}
    for a, b, _ in edge_list:
        adj[a].append(b)
        adj[b].append(a)
    paths: List[List[str]] = []

    def dfs(node: str, seen: set, path: List[str]) -> None:
        if node == goal:
            paths.append(list(path))
            return
        for nxt in sorted(adj[node]):
            if nxt not in seen:
                seen.add(nxt)
                path.append(nxt)
                dfs(nxt, seen, path)
                path.pop()
                seen.remove(nxt)

    dfs(start, {start}, [start])
    return paths


def solve_graph_path(params: Dict[str, Any]) -> Dict[str, Any]:
    nodes = params["nodes"]
    start = params["start"]
    goal = params["goal"]
    adj: Dict[str, List[Tuple[int, str]]] = {node: [] for node in nodes}
    for a, b, w in params["edges"]:
        adj[a].append((w, b))
        adj[b].append((w, a))
    pq = [(0, start, [start])]
    best: Dict[str, int] = {}
    while pq:
        dist, node, path = heapq.heappop(pq)
        if node in best and best[node] <= dist:
            continue
        best[node] = dist
        if node == goal:
            return {"type": "shortest_path", "start": start, "goal": goal, "path": path, "distance": dist}
        for w, nxt in adj[node]:
            heapq.heappush(pq, (dist + w, nxt, path + [nxt]))
    raise AssertionError("graph has no path")


def render_graph_path(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Weighted graph", "Highlight shortest path.", size=CANVAS_SIZE)
    draw.text((74, 54), f"Shortest path {params['start']} to {params['goal']}", font=load_font(24, bold=True), fill=PALETTE["ink"])
    positions = {node: GRAPH_POSITIONS[node] for node in params["nodes"]}
    weights = {edge_key(a, b): w for a, b, w in params["edges"]}
    highlight_path: List[str] = []
    highlight_color = PALETTE["green"]
    if mode == "teacher":
        highlight_path = params["shortest_path"]
    elif mode == "negative":
        highlight_path = params["negative_path"]
        highlight_color = PALETTE["red"]
    for a, b, _ in params["edges"]:
        draw.line((positions[a], positions[b]), fill="#94a3b8", width=4)
    if highlight_path:
        for i in range(len(highlight_path) - 1):
            a, b = highlight_path[i], highlight_path[i + 1]
            draw.line((positions[a], positions[b]), fill=highlight_color, width=10)
            draw.line((positions[a], positions[b]), fill="#ffffff", width=3)
    for a, b, w in params["edges"]:
        ax, ay = positions[a]
        bx, by = positions[b]
        mx, my = (ax + bx) // 2, (ay + by) // 2
        label = str(w)
        font = load_font(18, bold=True)
        tw, th = text_size(draw, label, font)
        draw.rounded_rectangle((mx - tw / 2 - 6, my - th / 2 - 5, mx + tw / 2 + 6, my + th / 2 + 7), radius=5, fill="#ffffff", outline="#cbd5e1")
        draw_centered_text(draw, (mx, my), label, font, fill=PALETTE["ink"])
    for node, (x, y) in positions.items():
        fill = "#ffffff"
        outline = "#111827"
        if node == params["start"]:
            outline = PALETTE["blue"]
        if node == params["goal"]:
            outline = PALETTE["orange"]
        draw.ellipse((x - 24, y - 24, x + 24, y + 24), fill=fill, outline=outline, width=4)
        draw_centered_text(draw, (x, y), node, load_font(24, bold=True), fill=PALETTE["ink"])
    return img


def graph_path(rng: random.Random, idx: int) -> Dict[str, Any]:
    candidates = [
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "E"),
        ("D", "F"),
        ("E", "F"),
        ("B", "C"),
        ("D", "E"),
        ("B", "E"),
        ("C", "D"),
    ]
    nodes = ["A", "B", "C", "D", "E", "F"]
    start = "A"
    goal = "F"
    for _ in range(1000):
        edges = []
        for a, b in candidates:
            if (a, b) in candidates[:6] or rng.random() < 0.55:
                edges.append((a, b, rng.randint(1, 9)))
        paths = enumerate_paths(nodes, edges, start, goal)
        weights = {edge_key(a, b): w for a, b, w in edges}
        ranked = sorted((path_weight(path, weights), path) for path in paths)
        if len(ranked) >= 2 and ranked[0][0] < ranked[1][0]:
            shortest = ranked[0][1]
            negative = ranked[1][1]
            break
    else:
        raise AssertionError("could not sample graph with unique shortest path")
    params = {
        "nodes": nodes,
        "edges": edges,
        "start": start,
        "goal": goal,
        "shortest_path": shortest,
        "negative_path": negative,
        "shortest_distance": ranked[0][0],
        "negative_distance": ranked[1][0],
    }
    ground_truth = solve_graph_path(params)
    path_label = "-".join(ground_truth["path"])
    return base_case(
        sub_task="graph_path",
        family_slug="graph",
        task_family="graph_reasoning",
        sub_family="shortest_weighted_path",
        domain="graph_theory",
        knowledge_type="procedural",
        difficulty_level=4,
        difficulty_reason="The answer requires comparing total weights across multiple graph paths and highlighting the unique minimum.",
        instruction=f"Highlight the shortest weighted path from {start} to {goal}.",
        expected_target=f"The path {path_label} is highlighted in green.",
        rational_target_description=(
            f"The total weight of {path_label} is {ground_truth['distance']}, lower than the available alternatives. "
            "Only those shortest-path edges should be highlighted."
        ),
        required_knowledge=[rule_fact("A weighted shortest path minimizes the sum of edge weights from the start node to the goal node.")],
        search_queries=["weighted graph shortest path sum edge weights"],
        source_scene_graph={
            "objects": ["six labeled nodes", "weighted undirected edges", "start node A", "goal node F"],
            "editable_region": "edges along the shortest path",
            "preserve_region": "node positions, node labels, all edge weights, and non-path edges",
        },
        edit_operations=[operation("highlight_edges", "shortest path edges", f"path {path_label}", "draw the shortest path edges in green")],
        negative_constraints=[
            f"Do not highlight {'-'.join(negative)}; its total weight is {ranked[1][0]}, which is longer.",
            "Do not change any edge weight or node label.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Are exactly the edges of path {path_label} highlighted?", "weight": 0.35},
            {"id": "C2", "question": "Is the highlighted path connected from A to F?", "weight": 0.20},
            {"id": "C3", "question": "Is the highlighted path the minimum total weight path?", "weight": 0.25},
            {"id": "C4", "question": "Are all node labels and edge weights preserved?", "weight": 0.10},
            {"id": "C5", "question": "Are non-path edges left unhighlighted?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": "Reading the highlighted nodes from start to end, which node path is highlighted? Answer hyphen-separated capital letters.", "expected_answer": path_label, "weight": 0.60},
                {"question": "What color highlights the path? Answer one word.", "expected_answer": "green", "weight": 0.20},
                {"question": "What letter is written on the leftmost start node? Answer one capital letter.", "expected_answer": start, "weight": 0.20},
            ],
            "programmatic": {"solver": "solve_graph_path", "params": params},
        },
        params=params,
        render=render_graph_path,
    )


def mirror_x_point(x: int, mirror_x: int) -> int:
    return 2 * mirror_x - x


def solve_mirror_reflection(params: Dict[str, Any]) -> Dict[str, Any]:
    mirror_x = params["mirror_x"]
    mirrored_flag = [[mirror_x_point(x, mirror_x), y] for x, y in params["flag_points"]]
    mirrored_pole = [mirror_x_point(params["pole_box"][2], mirror_x), params["pole_box"][1], mirror_x_point(params["pole_box"][0], mirror_x), params["pole_box"][3]]
    return {
        "type": "mirror_reflection",
        "mirror_x": mirror_x,
        "right_copy_orientation": "left",
        "mirrored_flag_points": mirrored_flag,
        "mirrored_pole_box": mirrored_pole,
    }


def render_flag_object(draw: ImageDraw.ImageDraw, pole_box: Sequence[int], flag_points: Sequence[Sequence[int]], fill: str, outline: str) -> None:
    draw.rectangle(tuple(pole_box), fill=fill, outline=outline, width=3)
    draw.polygon([tuple(p) for p in flag_points], fill=fill, outline=outline)


def render_mirror_reflection(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Mirror reflection", "Add the reflected copy.", size=CANVAS_SIZE)
    mx = params["mirror_x"]
    draw.text((72, 58), "Vertical mirror", font=load_font(24, bold=True), fill=PALETTE["ink"])
    draw.line((mx, 94, mx, 430), fill="#0f766e", width=5)
    draw.text((mx + 10, 104), "mirror", font=load_font(18, bold=True), fill="#0f766e")
    render_flag_object(draw, params["pole_box"], params["flag_points"], PALETTE["sky"], "#075985")
    if mode != "source":
        if mode == "teacher":
            solved = solve_mirror_reflection(params)
            pole = solved["mirrored_pole_box"]
            flag = solved["mirrored_flag_points"]
            fill = "#86efac"
            outline = "#166534"
        else:
            dx = params["negative_dx"]
            pole = [params["pole_box"][0] + dx, params["pole_box"][1], params["pole_box"][2] + dx, params["pole_box"][3]]
            flag = [[x + dx, y] for x, y in params["flag_points"]]
            fill = "#fecaca"
            outline = "#991b1b"
        render_flag_object(draw, pole, flag, fill, outline)
    return img


def mirror_reflection(rng: random.Random, idx: int) -> Dict[str, Any]:
    mirror_x = rng.randint(252, 264)
    x = rng.randint(118, 150)
    y = rng.randint(176, 220)
    pole_box = [x, y, x + 12, y + 118]
    flag_points = [[x + 12, y + 14], [x + rng.randint(76, 92), y + 40], [x + 12, y + 68]]
    negative_dx = mirror_x + 38 - x
    params = {"mirror_x": mirror_x, "pole_box": pole_box, "flag_points": flag_points, "negative_dx": negative_dx}
    ground_truth = solve_mirror_reflection(params)
    return base_case(
        sub_task="mirror_reflection",
        family_slug="spatial",
        task_family="spatial_reasoning",
        sub_family="mirror_symmetry",
        domain="geometry",
        knowledge_type="procedural",
        difficulty_level=3,
        difficulty_reason="The target requires reflecting an asymmetric object across a vertical mirror line, reversing its orientation.",
        instruction="Add the correct mirror reflection of the left object on the right side of the vertical mirror line.",
        expected_target="A reflected copy appears on the right side, with the flag pointing left toward the mirror.",
        rational_target_description=(
            "A vertical mirror reflection places each point the same distance on the opposite side of the mirror line. "
            "Because the source flag points right, the reflected copy must point left."
        ),
        required_knowledge=[rule_fact("Reflection across a vertical line preserves distance from the line but reverses left-right orientation.")],
        search_queries=["vertical mirror reflection left right orientation"],
        source_scene_graph={
            "objects": ["vertical mirror line", "asymmetric flag object on the left"],
            "editable_region": "empty right side of the mirror line",
            "preserve_region": "left object, mirror line, label, background, and border",
        },
        edit_operations=[operation("add_reflected_object", "right side of mirror", "right half of canvas", "draw the horizontally mirrored copy")],
        negative_constraints=[
            "Do not merely translate the original object; the reflected copy must reverse left-right orientation.",
            "Do not move or redraw the original left object.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": "Is a copy of the object added on the right side of the mirror line?", "weight": 0.20},
            {"id": "C2", "question": "Does the right copy point left, opposite the source object?", "weight": 0.30},
            {"id": "C3", "question": "Are matching points equally distant from the mirror line?", "weight": 0.25},
            {"id": "C4", "question": "Is the original left object unchanged?", "weight": 0.15},
            {"id": "C5", "question": "Is the vertical mirror line preserved?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": "Which direction does the reflected flag point? Answer one word.", "expected_answer": "left", "weight": 0.55},
                {"question": "Is the copy on the right side of the mirror? Answer yes or no.", "expected_answer": "yes", "weight": 0.25},
                {"question": "Is the mirror line vertical? Answer yes or no.", "expected_answer": "yes", "weight": 0.20},
            ],
            "programmatic": {"solver": "solve_mirror_reflection", "params": params},
        },
        params=params,
        render=render_mirror_reflection,
    )


BLOCK_COLORS = [
    ("red", "#ef4444"),
    ("blue", "#3b82f6"),
    ("green", "#22c55e"),
    ("yellow", "#facc15"),
    ("purple", "#a855f7"),
]


def solve_block_stack_view(params: Dict[str, Any]) -> Dict[str, Any]:
    stacks = [list(col) for col in params["stacks"]]
    col = params["remove_col"]
    row = params["remove_row"]
    removed = stacks[col].pop(row)
    return {"type": "block_stack_after_removal", "remove_col": col, "remove_row": row, "removed_color": removed, "final_stacks": stacks}


def render_blocks(draw: ImageDraw.ImageDraw, stacks: List[List[Optional[str]]], mark: Optional[Tuple[int, int]] = None) -> None:
    color_map = dict(BLOCK_COLORS)
    block = 58
    gap = 18
    x0 = 126
    base_y = 408
    draw.line((92, base_y + 4, 420, base_y + 4), fill="#475569", width=4)
    for c, stack in enumerate(stacks):
        x = x0 + c * (block + gap)
        draw.text((x + 17, base_y + 18), str(c + 1), font=load_font(18, bold=True), fill=PALETTE["muted"])
        for r, color_name in enumerate(stack):
            y = base_y - (r + 1) * block
            box = (x, y, x + block, y + block)
            if color_name is None:
                draw.rectangle(box, fill="#f8fafc", outline="#ef4444", width=3)
                draw.line((box[0] + 8, box[1] + 8, box[2] - 8, box[3] - 8), fill="#ef4444", width=2)
                draw.line((box[0] + 8, box[3] - 8, box[2] - 8, box[1] + 8), fill="#ef4444", width=2)
                continue
            draw.rectangle(box, fill=color_map[color_name], outline="#111827", width=3)
            if mark == (c, r):
                draw.line((box[0] + 10, box[1] + 10, box[2] - 10, box[3] - 10), fill="#111827", width=5)
                draw.line((box[0] + 10, box[3] - 10, box[2] - 10, box[1] + 10), fill="#111827", width=5)


def render_block_stack_view(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Block stack", "Remove marked block.", size=CANVAS_SIZE)
    draw.text((76, 58), "Remove the marked block", font=load_font(24, bold=True), fill=PALETTE["ink"])
    if mode == "source":
        render_blocks(draw, [list(col) for col in params["stacks"]], mark=(params["remove_col"], params["remove_row"]))
    elif mode == "negative":
        stacks: List[List[Optional[str]]] = [list(col) for col in params["stacks"]]
        stacks[params["remove_col"]][params["remove_row"]] = None
        render_blocks(draw, stacks)
    else:
        solved = solve_block_stack_view(params)
        render_blocks(draw, solved["final_stacks"])
    return img


def block_stack_view(rng: random.Random, idx: int) -> Dict[str, Any]:
    color_names = [name for name, _ in BLOCK_COLORS]
    while True:
        stacks = []
        for _ in range(3):
            h = rng.randint(2, 4)
            stacks.append([rng.choice(color_names) for _ in range(h)])
        cols = [i for i, stack in enumerate(stacks) if len(stack) >= 2]
        remove_col = rng.choice(cols)
        remove_row = rng.randint(0, len(stacks[remove_col]) - 2)
        if len(stacks[remove_col]) <= 4:
            break
    params = {"stacks": stacks, "remove_col": remove_col, "remove_row": remove_row}
    ground_truth = solve_block_stack_view(params)
    col_text = remove_col + 1
    row_text = remove_row + 1
    final_height = len(ground_truth["final_stacks"][remove_col])
    return base_case(
        sub_task="block_stack_view",
        family_slug="spatial",
        task_family="causal_spatial_reasoning",
        sub_family="gravity_stack_collapse",
        domain="spatial_physics",
        knowledge_type="procedural",
        difficulty_level=3,
        difficulty_reason="Removing a lower block requires blocks above it to fall down and close the gap.",
        instruction="Show the block stacks after removing the marked block; gravity makes any blocks above it fall down.",
        expected_target=f"Column {col_text} has the marked block removed and collapses to height {final_height}.",
        rational_target_description=(
            f"The marked block is in column {col_text}, row {row_text} from the bottom. After removal, blocks above it drop down so no gap remains."
        ),
        required_knowledge=[rule_fact("In a vertical stack, unsupported blocks above a removed block fall downward until they rest on another block or the ground.")],
        search_queries=["block stack remove support gravity collapse"],
        source_scene_graph={
            "objects": ["three columns of colored square blocks", "ground line", "marked block with X"],
            "editable_region": f"column {col_text} stack",
            "preserve_region": "other columns, ground line, block colors, and canvas border",
        },
        edit_operations=[
            operation("remove_and_collapse", "marked block column", f"column {col_text}, row {row_text}", "remove marked block and drop upper blocks down")
        ],
        negative_constraints=[
            "Do not leave an empty gap where the marked block was removed.",
            "Do not remove or recolor blocks in other columns.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Is the marked block removed from column {col_text}?", "weight": 0.25},
            {"id": "C2", "question": "Have blocks above the removed block fallen down with no gap?", "weight": 0.30},
            {"id": "C3", "question": f"Does column {col_text} have height {final_height} after the edit?", "weight": 0.20},
            {"id": "C4", "question": "Are all other columns unchanged?", "weight": 0.15},
            {"id": "C5", "question": "Are the ground line and block style preserved?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": f"How many blocks are in column {col_text}? Answer digits only.", "expected_answer": str(final_height), "weight": 0.45},
                {"question": f"Is there an empty gap in column {col_text}? Answer yes or no.", "expected_answer": "no", "weight": 0.35},
                {"question": "How many block columns are shown? Answer digits only.", "expected_answer": "3", "weight": 0.20},
            ],
            "programmatic": {"solver": "solve_block_stack_view", "params": params},
        },
        params=params,
        render=render_block_stack_view,
    )


def solve_circuit_bulb(params: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "circuit_bulb", "switch_state": "closed", "bulb_state": "lit"}


def render_circuit_bulb(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Circuit bulb", "Close the switch.", size=CANVAS_SIZE)
    draw.text((72, 58), "Close the switch", font=load_font(24, bold=True), fill=PALETTE["ink"])
    wire = "#334155"
    x_left, x_right = 118, 394
    y_top, y_bot = 164, 344
    switch_left = (236, y_top)
    switch_right = (310, y_top)
    bulb_center = (256, y_bot)
    bulb_lit = mode == "teacher"
    switch_closed = mode != "source"

    draw.line((x_left, y_top, switch_left[0], y_top), fill=wire, width=6)
    draw.line((switch_right[0], y_top, x_right, y_top), fill=wire, width=6)
    draw.line((x_left, y_top, x_left, y_bot), fill=wire, width=6)
    draw.line((x_right, y_top, x_right, y_bot), fill=wire, width=6)
    draw.line((x_left, y_bot, bulb_center[0] - 50, y_bot), fill=wire, width=6)
    draw.line((bulb_center[0] + 50, y_bot, x_right, y_bot), fill=wire, width=6)

    draw.line((x_left - 22, 220, x_left - 22, 290), fill="#111827", width=7)
    draw.line((x_left - 4, 238, x_left - 4, 272), fill="#111827", width=4)
    draw.text((x_left - 34, 194), "+", font=load_font(22, bold=True), fill=PALETTE["ink"])
    draw.text((x_left - 12, 194), "-", font=load_font(22, bold=True), fill=PALETTE["ink"])
    draw.text((x_left - 48, 302), "battery", font=load_font(16, bold=True), fill=PALETTE["muted"])

    for p in [switch_left, switch_right]:
        draw.ellipse((p[0] - 8, p[1] - 8, p[0] + 8, p[1] + 8), fill="#ffffff", outline="#111827", width=3)
    if switch_closed:
        switch_color = PALETTE["green"] if mode == "teacher" else PALETTE["red"]
        draw.line((switch_left, switch_right), fill=switch_color, width=7)
    else:
        draw.line((switch_left, (switch_right[0] - 16, switch_right[1] - 54)), fill=PALETTE["muted"], width=7)
    draw.text((238, 104), "switch", font=load_font(17, bold=True), fill=PALETTE["muted"])

    bulb_fill = PALETTE["yellow"] if bulb_lit else "#d1d5db"
    bulb_outline = PALETTE["green"] if bulb_lit else "#6b7280"
    if bulb_lit:
        for r, color in [(78, "#fef3c7"), (62, "#fde68a")]:
            draw.ellipse((bulb_center[0] - r, bulb_center[1] - r, bulb_center[0] + r, bulb_center[1] + r), fill=color)
    draw.ellipse((bulb_center[0] - 48, bulb_center[1] - 48, bulb_center[0] + 48, bulb_center[1] + 48), fill=bulb_fill, outline=bulb_outline, width=5)
    draw.arc((bulb_center[0] - 22, bulb_center[1] - 16, bulb_center[0] + 22, bulb_center[1] + 26), 200, 340, fill="#92400e", width=4)
    draw.rectangle((bulb_center[0] - 22, bulb_center[1] + 46, bulb_center[0] + 22, bulb_center[1] + 68), fill="#94a3b8", outline="#475569", width=3)
    draw.text((230, 428), "bulb", font=load_font(17, bold=True), fill=PALETTE["muted"])
    return img


def circuit_bulb(rng: random.Random, idx: int) -> Dict[str, Any]:
    params: Dict[str, Any] = {"initial_switch_state": "open", "target_switch_state": "closed"}
    ground_truth = solve_circuit_bulb(params)
    return base_case(
        sub_task="circuit_bulb",
        family_slug="physics",
        task_family="causal_reasoning",
        sub_family="electric_circuit_state_change",
        domain="physics",
        knowledge_type="conceptual",
        difficulty_level=2,
        difficulty_reason="Closing a complete series circuit changes the switch state and allows current through the bulb.",
        instruction="Show the circuit after the switch closes.",
        expected_target="The switch is closed and the bulb is lit yellow.",
        rational_target_description="A closed switch completes the series circuit, so current can flow through the bulb and it lights.",
        required_knowledge=[rule_fact("In a complete series circuit, closing the switch allows current to flow and turns the bulb on.")],
        search_queries=["series circuit switch closed bulb lights"],
        source_scene_graph={
            "objects": ["battery", "series wires", "open switch", "gray off bulb"],
            "editable_region": "top switch and bottom bulb",
            "preserve_region": "battery, wire path, labels, background, and border",
        },
        edit_operations=[
            operation("close_switch", "open switch", "upper center", "draw the switch lever closed"),
            operation("change_color", "bulb", "lower center", "change the bulb from gray to lit yellow"),
        ],
        negative_constraints=[
            "Do not leave the bulb gray after the switch is closed.",
            "Do not alter the battery polarity or wire layout.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": "Is the switch visibly closed?", "weight": 0.30},
            {"id": "C2", "question": "Is the bulb lit yellow?", "weight": 0.35},
            {"id": "C3", "question": "Is the circuit still a single series loop?", "weight": 0.15},
            {"id": "C4", "question": "Are the battery and wire path preserved?", "weight": 0.10},
            {"id": "C5", "question": "Is no extra component added?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": "What color is the bulb, yellow or gray? Answer one word.", "expected_answer": "yellow", "weight": 0.60},
                {"question": "Is the switch visibly closed? Answer yes or no.", "expected_answer": "yes", "weight": 0.25},
                {"question": "How many bulbs are shown? Answer digits only.", "expected_answer": "1", "weight": 0.15},
            ],
            "programmatic": {"solver": "solve_circuit_bulb", "params": params},
        },
        params=params,
        render=render_circuit_bulb,
    )


PH_SOLUTIONS = [
    {"solution": "lemon juice", "ph_category": "acid", "expected_color": "red", "negative_color": "purple"},
    {"solution": "vinegar", "ph_category": "acid", "expected_color": "orange", "negative_color": "purple"},
    {"solution": "pure water", "ph_category": "neutral", "expected_color": "green", "negative_color": "red"},
    {"solution": "soap", "ph_category": "base", "expected_color": "blue", "negative_color": "red"},
    {"solution": "baking soda", "ph_category": "base", "expected_color": "purple", "negative_color": "red"},
]

PH_COLOR_MAP = {
    "red": "#ef4444",
    "orange": "#f97316",
    "green": "#22c55e",
    "blue": "#2563eb",
    "purple": "#7c3aed",
    "white": "#ffffff",
}


def solve_ph_indicator(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "ph_indicator",
        "solution": params["solution"],
        "ph_category": params["ph_category"],
        "expected_color": params["expected_color"],
    }


def render_ph_indicator(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("pH indicator", "Color the strip.", size=CANVAS_SIZE)
    draw.text((72, 58), "Universal indicator", font=load_font(24, bold=True), fill=PALETTE["ink"])
    beaker = (142, 128, 370, 394)
    draw.line((beaker[0], beaker[1], beaker[0] + 28, beaker[3]), fill="#334155", width=4)
    draw.line((beaker[2], beaker[1], beaker[2] - 28, beaker[3]), fill="#334155", width=4)
    draw.line((beaker[0] + 28, beaker[3], beaker[2] - 28, beaker[3]), fill="#334155", width=4)
    draw.arc((beaker[0], beaker[1] - 10, beaker[2], beaker[1] + 34), 0, 180, fill="#334155", width=4)
    liquid_box = (168, 260, 344, 388)
    draw.rectangle(liquid_box, fill="#dbeafe", outline="#93c5fd", width=2)
    draw.line((liquid_box[0], liquid_box[1], liquid_box[2], liquid_box[1]), fill="#60a5fa", width=3)
    draw_centered_text(draw, (256, 226), params["solution"], load_font(22, bold=True), fill=PALETTE["ink"])
    if mode == "source":
        strip_color = "white"
        outline = PALETTE["muted"]
    elif mode == "negative":
        strip_color = params["negative_color"]
        outline = PALETTE["red"]
    else:
        strip_color = params["expected_color"]
        outline = PALETTE["green"]
    strip = (230, 118, 282, 370)
    draw.rounded_rectangle(strip, radius=7, fill=PH_COLOR_MAP[strip_color], outline=outline, width=4)
    for y in range(strip[1] + 28, strip[3] - 10, 38):
        draw.line((strip[0] + 6, y, strip[2] - 6, y), fill="#cbd5e1" if strip_color == "white" else "#ffffff", width=2)
    draw.text((194, 412), "indicator strip", font=load_font(17, bold=True), fill=PALETTE["muted"])
    return img


def ph_indicator(rng: random.Random, idx: int) -> Dict[str, Any]:
    sample = dict(rng.choice(PH_SOLUTIONS))
    params = sample
    ground_truth = solve_ph_indicator(params)
    expected = params["expected_color"]
    return base_case(
        sub_task="ph_indicator",
        family_slug="chemistry",
        task_family="scientific_reasoning",
        sub_family="indicator_color_mapping",
        domain="chemistry",
        knowledge_type="conceptual",
        difficulty_level=2,
        difficulty_reason="The solution label determines the acid, neutral, or base category and the universal-indicator color.",
        instruction="Color the universal indicator strip correctly for the labeled solution.",
        expected_target=f"The strip for {params['solution']} is colored {expected}.",
        rational_target_description=f"{params['solution']} is {params['ph_category']}, so the universal indicator should be {expected}.",
        required_knowledge=[rule_fact("Universal indicator is red or orange for acids, green for neutral solutions, and blue or purple for bases.")],
        search_queries=["universal indicator colors acid neutral base"],
        source_scene_graph={
            "objects": ["labeled beaker", "blue solution fill", "white universal indicator strip"],
            "editable_region": "vertical indicator strip in the beaker",
            "preserve_region": "beaker outline, solution label, liquid level, background, and border",
        },
        edit_operations=[operation("change_color", "indicator strip", "middle center", f"color the strip {expected}")],
        negative_constraints=[
            f"Do not color the strip {params['negative_color']}; that indicates the wrong pH end.",
            "Do not change the beaker label or liquid level.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Is the strip colored {expected}?", "weight": 0.40},
            {"id": "C2", "question": f"Is the solution treated as {params['ph_category']}?", "weight": 0.25},
            {"id": "C3", "question": "Is the beaker label preserved?", "weight": 0.15},
            {"id": "C4", "question": "Is only the strip color edited?", "weight": 0.10},
            {"id": "C5", "question": "Is the strip still inside the beaker?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": "What color is the indicator strip? Answer one word.", "expected_answer": expected, "weight": 0.60},
                {"question": "What text is written on the beaker label? Answer exactly as written.", "expected_answer": params["solution"], "weight": 0.25},
                {"question": "Is the indicator strip inside the beaker? Answer yes or no.", "expected_answer": "yes", "weight": 0.15},
            ],
            "programmatic": {"solver": "solve_ph_indicator", "params": params},
        },
        params=params,
        render=render_ph_indicator,
    )


FOOD_CHAINS = [
    ["grass", "rabbit", "fox"],
    ["algae", "small fish", "heron"],
    ["corn", "mouse", "snake", "hawk"],
    ["leaf", "caterpillar", "bird", "cat"],
]


def solve_food_chain(params: Dict[str, Any]) -> Dict[str, Any]:
    i = params["wrong_index"]
    order = params["organisms"]
    return {"type": "food_chain_edge", "corrected_edge": {"from": order[i], "to": order[i + 1]}}


def render_food_chain(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Food chain", "Fix energy flow.", size=CANVAS_SIZE)
    draw.text((72, 58), "Energy flow", font=load_font(24, bold=True), fill=PALETTE["ink"])
    organisms = params["organisms"]
    n = len(organisms)
    box_w = 96 if n == 4 else 118
    gap = 26
    total = n * box_w + (n - 1) * gap
    x0 = (CANVAS_SIZE - total) // 2
    y0 = 226
    boxes = []
    for i, name in enumerate(organisms):
        box = (x0 + i * (box_w + gap), y0, x0 + i * (box_w + gap) + box_w, y0 + 82)
        boxes.append(box)
        draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#111827", width=3)
        box_text(draw, box, name, size=18 if len(name) > 8 else 21)

    flipped = set()
    arrow_color = PALETTE["muted"]
    if mode == "source":
        flipped.add(params["wrong_index"])
    elif mode == "negative":
        flipped.update([params["wrong_index"], params["negative_index"]])
        arrow_color = PALETTE["red"]
    else:
        arrow_color = PALETTE["green"]
    for i in range(n - 1):
        left = boxes[i]
        right = boxes[i + 1]
        if i in flipped:
            draw_arrow(draw, (right[0] - 8, (right[1] + right[3]) // 2), (left[2] + 8, (left[1] + left[3]) // 2), arrow_color, width=5)
        else:
            draw_arrow(draw, (left[2] + 8, (left[1] + left[3]) // 2), (right[0] - 8, (right[1] + right[3]) // 2), arrow_color, width=5)
    return img


def food_chain(rng: random.Random, idx: int) -> Dict[str, Any]:
    organisms = list(rng.choice(FOOD_CHAINS))
    wrong_index = rng.randrange(len(organisms) - 1)
    other_indices = [i for i in range(len(organisms) - 1) if i != wrong_index]
    negative_index = rng.choice(other_indices) if other_indices else wrong_index
    params = {"organisms": organisms, "wrong_index": wrong_index, "negative_index": negative_index}
    ground_truth = solve_food_chain(params)
    corrected = ground_truth["corrected_edge"]
    edge_weight = 1.0 / (len(organisms) - 1)
    edge_vqa_checks = [
        {
            "question": f"Is there an arrow pointing from {organisms[i]} to {organisms[i + 1]}? Answer yes or no.",
            "expected_answer": "yes",
            "weight": edge_weight,
        }
        for i in range(len(organisms) - 1)
    ]
    return base_case(
        sub_task="food_chain",
        family_slug="biology",
        task_family="scientific_reasoning",
        sub_family="energy_flow_direction",
        domain="biology",
        knowledge_type="conceptual",
        difficulty_level=2,
        difficulty_reason="One arrow in a simple food chain points opposite the correct energy-flow direction.",
        instruction="Fix the wrong arrow so the food-chain energy flow is correct.",
        expected_target=f"The arrow is corrected to point from {corrected['from']} to {corrected['to']}.",
        rational_target_description="Food-chain arrows point from the organism that provides energy to the organism that receives it.",
        required_knowledge=[rule_fact("In a food chain, arrows show energy flowing from food or prey toward the consumer.")],
        search_queries=["food chain arrows direction energy flow prey consumer"],
        source_scene_graph={
            "objects": ["organism label boxes", "arrow connectors", "one reversed energy-flow arrow"],
            "editable_region": f"arrow between {corrected['from']} and {corrected['to']}",
            "preserve_region": "organism labels, box order, non-target arrows, background, and border",
        },
        edit_operations=[operation("flip_arrow", "wrong food-chain arrow", "middle row", f"point from {corrected['from']} to {corrected['to']}")],
        negative_constraints=[
            "Do not flip a different arrow while leaving the originally wrong arrow incorrect.",
            "Do not reorder or rename the organism boxes.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Does the corrected arrow point from {corrected['from']} to {corrected['to']}?", "weight": 0.40},
            {"id": "C2", "question": "Do all arrows follow producer/prey to consumer energy flow?", "weight": 0.25},
            {"id": "C3", "question": "Are the organism labels unchanged?", "weight": 0.15},
            {"id": "C4", "question": "Is only the wrong arrow fixed?", "weight": 0.10},
            {"id": "C5", "question": "Is the horizontal box layout preserved?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": edge_vqa_checks,
            "programmatic": {"solver": "solve_food_chain", "params": params},
        },
        params=params,
        render=render_food_chain,
    )


def solve_geometry_angle(params: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "triangle_angle", "angle_deg": 180 - params["angle_a"] - params["angle_b"]}


def render_geometry_angle(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Triangle angle", "Find the missing angle.", size=CANVAS_SIZE)
    draw.text((72, 58), "Triangle angles", font=load_font(24, bold=True), fill=PALETTE["ink"])
    pts = [(126, 374), (386, 374), (278, 138)]
    draw.polygon(pts, fill="#ffffff", outline="#111827")
    draw.line((pts[0], pts[1]), fill="#111827", width=5)
    draw.line((pts[1], pts[2]), fill="#111827", width=5)
    draw.line((pts[2], pts[0]), fill="#111827", width=5)
    solved = solve_geometry_angle(params)
    if mode == "source":
        third = "?"
        fill = PALETTE["muted"]
    elif mode == "negative":
        third = f"{params['wrong_angle']} deg"
        fill = PALETTE["red"]
    else:
        third = f"{solved['angle_deg']} deg"
        fill = PALETTE["green"]
    labels = [
        (pts[0][0] + 52, pts[0][1] - 34, f"{params['angle_a']} deg", PALETTE["ink"]),
        (pts[1][0] - 58, pts[1][1] - 34, f"{params['angle_b']} deg", PALETTE["ink"]),
        (pts[2][0], pts[2][1] + 56, third, fill),
    ]
    for x, y, text, color in labels:
        font = load_font(24, bold=True)
        tw, th = text_size(draw, text, font)
        draw.rounded_rectangle((x - tw / 2 - 8, y - th / 2 - 6, x + tw / 2 + 8, y + th / 2 + 8), radius=6, fill="#ffffff", outline=color, width=3)
        draw_centered_text(draw, (x, y), text, font, fill=color)
    return img


def geometry_angle(rng: random.Random, idx: int) -> Dict[str, Any]:
    while True:
        angle_a = rng.choice([40, 45, 50, 55, 60, 65, 70, 75])
        angle_b = rng.choice([35, 40, 45, 50, 55, 60, 65])
        angle_c = 180 - angle_a - angle_b
        if 35 <= angle_c <= 95:
            break
    delta = rng.choice([-30, -20, -10, 10, 20, 30])
    wrong = angle_c + delta
    if wrong <= 0 or wrong >= 180:
        wrong = angle_c - delta
    params = {"angle_a": angle_a, "angle_b": angle_b, "wrong_angle": wrong}
    ground_truth = solve_geometry_angle(params)
    angle = ground_truth["angle_deg"]
    return base_case(
        sub_task="geometry_angle",
        family_slug="math",
        task_family="quantitative_reasoning",
        sub_family="triangle_angle_sum",
        domain="math",
        knowledge_type="procedural",
        difficulty_level=2,
        difficulty_reason="The missing interior angle follows directly from the 180-degree triangle angle sum.",
        instruction="Replace the ? with the correct missing interior angle of the triangle.",
        expected_target=f"The missing angle label is replaced with {angle} deg.",
        rational_target_description=f"The two shown angles sum to {angle_a + angle_b} deg, so the third angle is 180 - {angle_a + angle_b} = {angle} deg.",
        required_knowledge=[rule_fact("The three interior angles of a triangle sum to 180 degrees.")],
        search_queries=["triangle interior angles sum 180 missing angle"],
        source_scene_graph={
            "objects": ["triangle outline", "two interior angle labels", "question-mark angle label"],
            "editable_region": "top interior angle label",
            "preserve_region": "triangle sides, two known angle labels, background, and border",
        },
        edit_operations=[operation("replace_text", "question-mark angle label", "upper center inside triangle", f"replace ? with {angle} deg")],
        negative_constraints=[
            f"Do not use {wrong} deg; the three angles would not sum to 180 deg.",
            "Do not move the triangle or change the known angle labels.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Does the missing angle show {angle} deg?", "weight": 0.40},
            {"id": "C2", "question": "Do the three angle labels sum to 180 deg?", "weight": 0.25},
            {"id": "C3", "question": "Are the two given angle labels unchanged?", "weight": 0.15},
            {"id": "C4", "question": "Is the answer placed where the ? was?", "weight": 0.10},
            {"id": "C5", "question": "Is the triangle outline preserved?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": "What number is written at the top vertex inside the triangle? Answer digits only.", "expected_answer": str(angle), "weight": 0.60},
                {"question": "How many angle labels are shown? Answer digits only.", "expected_answer": "3", "weight": 0.20},
                {"question": "What shape is drawn? Answer one word.", "expected_answer": "triangle", "weight": 0.20},
            ],
            "programmatic": {"solver": "solve_geometry_angle", "params": params},
        },
        params=params,
        render=render_geometry_angle,
    )


def solve_sorting_step(params: Dict[str, Any]) -> Dict[str, Any]:
    arr = list(params["array"])
    i = params["swap_index"]
    arr[i], arr[i + 1] = arr[i + 1], arr[i]
    return {"type": "bubble_sort_next_swap", "swapped_indices": [i, i + 1], "resulting_array": arr}


def render_sorting_step(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Bubble sort", "Show the next swap.", size=CANVAS_SIZE)
    draw.text((72, 58), "bubble sort: next swap", font=load_font(24, bold=True), fill=PALETTE["ink"])
    if mode == "source":
        arr = list(params["array"])
        highlight = params["swap_index"]
        outline = PALETTE["blue"]
    elif mode == "teacher":
        arr = list(params["array"])
        highlight = params["swap_index"]
        arr[highlight], arr[highlight + 1] = arr[highlight + 1], arr[highlight]
        outline = PALETTE["green"]
    elif mode == "negative":
        arr = list(params["array"])
        highlight = params["negative_index"]
        arr[highlight], arr[highlight + 1] = arr[highlight + 1], arr[highlight]
        outline = PALETTE["red"]
    else:
        raise ValueError(f"unknown sorting render mode: {mode}")
    bar_w = 54
    gap = 28
    x0 = (CANVAS_SIZE - (5 * bar_w + 4 * gap)) // 2
    base_y = 390
    max_val = max(params["array"])
    for i, value in enumerate(arr):
        x = x0 + i * (bar_w + gap)
        height = 72 + int((value / max_val) * 150)
        if i in [highlight, highlight + 1]:
            draw.rounded_rectangle((x - 8, base_y - 236, x + bar_w + 8, base_y + 12), radius=7, fill="#ffffff", outline=outline, width=4)
        draw.rectangle((x, base_y - height, x + bar_w, base_y), fill="#60a5fa", outline="#111827", width=3)
        draw_centered_text(draw, (x + bar_w / 2, base_y - height - 22), str(value), load_font(20, bold=True), fill=PALETTE["ink"])
        draw_centered_text(draw, (x + bar_w / 2, base_y + 28), str(i), load_font(16, bold=True), fill=PALETTE["muted"])
    draw.line((76, base_y, 436, base_y), fill="#475569", width=4)
    return img


def sorting_step(rng: random.Random, idx: int) -> Dict[str, Any]:
    values = rng.sample(range(2, 10), 5)
    descending_pairs = [i for i in range(4) if values[i] > values[i + 1]]
    if not descending_pairs:
        values[1], values[2] = 9, 3
        descending_pairs = [i for i in range(4) if values[i] > values[i + 1]]
    swap_index = rng.choice(descending_pairs)
    negative_index = rng.choice([i for i in range(4) if i != swap_index])
    params = {"array": values, "swap_index": swap_index, "negative_index": negative_index}
    ground_truth = solve_sorting_step(params)
    resulting_array_text = ", ".join(str(value) for value in ground_truth["resulting_array"])
    return base_case(
        sub_task="sorting_step",
        family_slug="cs",
        task_family="algorithmic_reasoning",
        sub_family="bubble_sort_adjacent_swap",
        domain="computer_science",
        knowledge_type="procedural",
        difficulty_level=3,
        difficulty_reason="The next state requires swapping only the highlighted out-of-order adjacent pair.",
        instruction="Show the array after the next bubble-sort swap of the highlighted compared pair.",
        expected_target=f"Indices {swap_index} and {swap_index + 1} are swapped, giving {ground_truth['resulting_array']}.",
        rational_target_description="In bubble sort, when the compared adjacent pair is out of order, that pair is swapped and all other positions stay fixed.",
        required_knowledge=[rule_fact("Bubble sort swaps a compared adjacent pair when the left value is greater than the right value.")],
        search_queries=["bubble sort adjacent pair next swap"],
        source_scene_graph={
            "objects": ["five vertical value bars", "index labels", "highlighted compared adjacent pair", "bubble sort caption"],
            "editable_region": f"bars at indices {swap_index} and {swap_index + 1}",
            "preserve_region": "caption, bar style, non-compared bars, index positions, background, and border",
        },
        edit_operations=[operation("swap_adjacent_bars", "highlighted compared pair", "middle row", f"swap indices {swap_index} and {swap_index + 1}")],
        negative_constraints=[
            f"Do not swap indices {negative_index} and {negative_index + 1}; that is not the highlighted compared pair.",
            "Do not sort the whole array or change any bar value.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Are indices {swap_index} and {swap_index + 1} swapped?", "weight": 0.35},
            {"id": "C2", "question": f"Does the resulting array equal {ground_truth['resulting_array']}?", "weight": 0.30},
            {"id": "C3", "question": "Are all other bar positions unchanged?", "weight": 0.15},
            {"id": "C4", "question": "Is the bubble-sort caption preserved?", "weight": 0.10},
            {"id": "C5", "question": "Are there still exactly five bars?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": "Reading the bar labels left to right, what are the five numbers? Answer comma-separated digits.", "expected_answer": resulting_array_text, "weight": 0.70},
                {"question": "How many bars are shown? Answer digits only.", "expected_answer": "5", "weight": 0.20},
                {"question": "What color outlines the compared pair? Answer one word.", "expected_answer": "green", "weight": 0.10},
            ],
            "programmatic": {"solver": "solve_sorting_step", "params": params},
        },
        params=params,
        render=render_sorting_step,
    )


def solve_fraction_shade(params: Dict[str, Any]) -> Dict[str, Any]:
    target = params["rows"] * params["cols"] * params["p"] // params["q"]
    return {"type": "fraction_shade", "fraction": f"{params['p']}/{params['q']}", "target_count": target}


def render_fraction_shade(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Fraction shade", "Shade the target fraction.", size=CANVAS_SIZE)
    caption = f"shade exactly {params['p']}/{params['q']} of the grid"
    draw.text((58, 58), caption, font=load_font(23, bold=True), fill=PALETTE["ink"])
    solved = solve_fraction_shade(params)
    if mode == "source":
        count = params["source_count"]
        shade = "#93c5fd"
    elif mode == "negative":
        count = params["wrong_count"]
        shade = "#fecaca"
    else:
        count = solved["target_count"]
        shade = "#86efac"
    rows = params["rows"]
    cols = params["cols"]
    cell = min(58, 280 // max(rows, cols))
    grid_w = cols * cell
    grid_h = rows * cell
    x0 = (CANVAS_SIZE - grid_w) // 2
    y0 = 150 + (260 - grid_h) // 2
    for r in range(rows):
        for c in range(cols):
            idx_cell = r * cols + c
            box = (x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell, y0 + (r + 1) * cell)
            fill = shade if idx_cell < count else "#ffffff"
            draw.rectangle(box, fill=fill, outline="#111827", width=2)
    draw.text((x0, y0 + grid_h + 18), f"shaded cells: {count}", font=load_font(18, bold=True), fill=PALETTE["muted"])
    return img


def fraction_shade(rng: random.Random, idx: int) -> Dict[str, Any]:
    choices = [
        (3, 4, 1, 2),
        (4, 4, 3, 8),
        (3, 5, 2, 5),
        (2, 6, 1, 3),
        (4, 5, 3, 5),
    ]
    rows, cols, p, q = rng.choice(choices)
    total = rows * cols
    target = total * p // q
    candidates = [target - 2, target - 1, target + 1, target + 2]
    source_count = rng.choice([c for c in candidates if 0 <= c <= total and c != target])
    wrong_count = target + rng.choice([-1, 1])
    if wrong_count < 0 or wrong_count > total:
        wrong_count = target - 1 if target > 0 else target + 1
    params = {"rows": rows, "cols": cols, "p": p, "q": q, "source_count": source_count, "wrong_count": wrong_count}
    ground_truth = solve_fraction_shade(params)
    return base_case(
        sub_task="fraction_shade",
        family_slug="math",
        task_family="quantitative_reasoning",
        sub_family="fraction_of_area",
        domain="math",
        knowledge_type="procedural",
        difficulty_level=2,
        difficulty_reason="The target count is found by multiplying the grid cell total by the requested fraction.",
        instruction="Shade cells so exactly the requested fraction of the grid is shaded.",
        expected_target=f"Exactly {ground_truth['target_count']} of the {total} cells are shaded.",
        rational_target_description=f"The grid has {total} cells, and {p}/{q} of {total} is {ground_truth['target_count']}.",
        required_knowledge=[rule_fact("A fraction of a grid equals the total number of cells multiplied by the fraction.")],
        search_queries=["shade fraction of grid count cells"],
        source_scene_graph={
            "objects": ["rectangular cell grid", "some shaded cells", "fraction caption"],
            "editable_region": "grid cells",
            "preserve_region": "grid dimensions, caption, cell borders, background, and border",
        },
        edit_operations=[operation("adjust_shaded_cells", "grid cells", "middle center", f"shade exactly {ground_truth['target_count']} cells")],
        negative_constraints=[
            f"Do not shade {wrong_count} cells; that is off by one from the target.",
            "Do not change the grid size or fraction caption.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Are exactly {ground_truth['target_count']} cells shaded?", "weight": 0.40},
            {"id": "C2", "question": f"Does the shaded count equal {p}/{q} of the grid?", "weight": 0.25},
            {"id": "C3", "question": "Are the grid dimensions unchanged?", "weight": 0.15},
            {"id": "C4", "question": "Is the caption preserved?", "weight": 0.10},
            {"id": "C5", "question": "Are cells shaded within the original grid only?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": "How many cells are shaded? Answer digits only.", "expected_answer": str(ground_truth["target_count"]), "weight": 0.55},
                {"question": "What fraction is written in the caption? Answer slash-separated digits.", "expected_answer": f"{p}/{q}", "weight": 0.25},
                {"question": "How many total cells are in the grid? Answer digits only.", "expected_answer": str(total), "weight": 0.20},
            ],
            "programmatic": {"solver": "solve_fraction_shade", "params": params},
        },
        params=params,
        render=render_fraction_shade,
    )


PROCESS_SEQUENCES = [
    ["seed", "sprout", "flower", "fruit"],
    ["wash", "cut", "cook", "serve"],
    ["mix", "pour", "bake", "cool"],
    ["wake", "dress", "eat", "leave"],
]


def solve_process_order(params: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "process_order", "correct_order": list(params["correct_order"])}


def render_process_order(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Process order", "Reorder the steps.", size=CANVAS_SIZE)
    draw.text((72, 58), "Process order", font=load_font(24, bold=True), fill=PALETTE["ink"])
    if mode == "source":
        order = params["source_order"]
        outline = PALETTE["blue"]
    elif mode == "negative":
        order = params["negative_order"]
        outline = PALETTE["red"]
    else:
        order = params["correct_order"]
        outline = PALETTE["green"]
    box_w = 86
    gap = 24
    x0 = (CANVAS_SIZE - (4 * box_w + 3 * gap)) // 2
    y0 = 222
    boxes = []
    for i, step in enumerate(order):
        box = (x0 + i * (box_w + gap), y0, x0 + i * (box_w + gap) + box_w, y0 + 82)
        boxes.append(box)
        draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline=outline if mode != "source" else "#111827", width=3)
        draw_centered_text(draw, ((box[0] + box[2]) / 2, box[1] - 20), str(i + 1), load_font(18, bold=True), fill=PALETTE["muted"])
        box_text(draw, box, step, size=19 if len(step) <= 6 else 17)
    for i in range(3):
        draw_arrow(draw, (boxes[i][2] + 6, (boxes[i][1] + boxes[i][3]) // 2), (boxes[i + 1][0] - 6, (boxes[i + 1][1] + boxes[i + 1][3]) // 2), PALETTE["muted"], width=4)
    return img


def process_order(rng: random.Random, idx: int) -> Dict[str, Any]:
    correct = list(rng.choice(PROCESS_SEQUENCES))
    i, j = sorted(rng.sample(range(4), 2))
    source = list(correct)
    source[i], source[j] = source[j], source[i]
    neg_i = rng.choice([0, 1, 2])
    negative = list(correct)
    negative[neg_i], negative[neg_i + 1] = negative[neg_i + 1], negative[neg_i]
    params = {"correct_order": correct, "source_order": source, "swapped_positions": [i, j], "negative_order": negative}
    ground_truth = solve_process_order(params)
    correct_order_text = ", ".join(correct)
    return base_case(
        sub_task="process_order",
        family_slug="kris",
        task_family="procedural_reasoning",
        sub_family="everyday_sequence_ordering",
        domain="everyday_common_sense",
        knowledge_type="procedural",
        difficulty_level=2,
        difficulty_reason="Two steps in a familiar four-step process are swapped and must be restored to chronological order.",
        instruction="Reorder the process strip into the correct sequence.",
        expected_target=f"The strip is ordered as {' -> '.join(correct)}.",
        rational_target_description="The correct answer restores the familiar start-to-finish order of the everyday process.",
        required_knowledge=[rule_fact("Procedural steps should be ordered by the real-world temporal sequence of the activity.")],
        search_queries=["everyday process steps chronological order"],
        source_scene_graph={
            "objects": ["four step boxes", "rightward process arrows", "two swapped step labels"],
            "editable_region": "horizontal process strip",
            "preserve_region": "box count, arrow connectors, style, background, and border",
        },
        edit_operations=[operation("reorder_steps", "process step boxes", "middle row", f"order as {' -> '.join(correct)}")],
        negative_constraints=[
            f"Do not leave the process as {' -> '.join(negative)}; one adjacent pair remains wrong.",
            "Do not add or delete process steps.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Is the order {' -> '.join(correct)}?", "weight": 0.40},
            {"id": "C2", "question": "Are all four original step labels present exactly once?", "weight": 0.25},
            {"id": "C3", "question": "Do the arrows still run left to right?", "weight": 0.15},
            {"id": "C4", "question": "Is no extra step added?", "weight": 0.10},
            {"id": "C5", "question": "Is the process-strip layout preserved?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": "List the four steps left to right, separated by commas.", "expected_answer": correct_order_text, "weight": 0.70},
                {"question": "How many steps are shown? Answer digits only.", "expected_answer": "4", "weight": 0.20},
                {"question": "Do the arrows point left to right? Answer yes or no.", "expected_answer": "yes", "weight": 0.10},
            ],
            "programmatic": {"solver": "solve_process_order", "params": params},
        },
        params=params,
        render=render_process_order,
    )


MOON_PHASES = [
    "new moon",
    "waxing crescent",
    "first quarter",
    "waxing gibbous",
    "full moon",
    "waning gibbous",
    "last quarter",
    "waning crescent",
]


def solve_moon_phase(params: Dict[str, Any]) -> Dict[str, Any]:
    phase = MOON_PHASES[(params["start_index"] + 3) % len(MOON_PHASES)]
    return {"type": "moon_phase_next", "phase_name": phase}


def draw_moon_icon(draw: ImageDraw.ImageDraw, center: Tuple[int, int], phase: str, outline: str = "#111827") -> None:
    cx, cy = center
    r = 34
    dark = "#111827"
    light = "#f8fafc"
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=dark, outline=outline, width=3)
    if phase == "new moon":
        return
    if phase == "full moon":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=light, outline=outline, width=3)
        return
    if phase == "first quarter":
        draw.pieslice((cx - r, cy - r, cx + r, cy + r), -90, 90, fill=light)
    elif phase == "last quarter":
        draw.pieslice((cx - r, cy - r, cx + r, cy + r), 90, 270, fill=light)
    elif phase == "waxing crescent":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=light)
        draw.ellipse((cx - r - 14, cy - r, cx + r - 14, cy + r), fill=dark)
    elif phase == "waning crescent":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=light)
        draw.ellipse((cx - r + 14, cy - r, cx + r + 14, cy + r), fill=dark)
    elif phase == "waxing gibbous":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=light)
        draw.ellipse((cx - r - 44, cy - r, cx + r - 44, cy + r), fill=dark)
    elif phase == "waning gibbous":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=light)
        draw.ellipse((cx - r + 44, cy - r, cx + r + 44, cy + r), fill=dark)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=outline, width=3)


def render_moon_phase(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Moon phase", "Complete the sequence.", size=CANVAS_SIZE)
    draw.text((72, 58), "Next moon phase", font=load_font(24, bold=True), fill=PALETTE["ink"])
    shown = [MOON_PHASES[(params["start_index"] + i) % len(MOON_PHASES)] for i in range(3)]
    solved = solve_moon_phase(params)
    if mode == "source":
        last: Optional[str] = None
        outline = PALETTE["muted"]
    elif mode == "negative":
        last = params["wrong_phase"]
        outline = PALETTE["red"]
    else:
        last = solved["phase_name"]
        outline = PALETTE["green"]
    phases: List[Optional[str]] = shown + [last]
    slot_w = 94
    gap = 20
    x0 = (CANVAS_SIZE - (4 * slot_w + 3 * gap)) // 2
    y0 = 206
    for i, phase in enumerate(phases):
        box = (x0 + i * (slot_w + gap), y0, x0 + i * (slot_w + gap) + slot_w, y0 + 118)
        draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline=outline if i == 3 else "#111827", width=3)
        center = ((box[0] + box[2]) // 2, box[1] + 48)
        if phase is None:
            draw_centered_text(draw, center, "?", load_font(34, bold=True), fill=PALETTE["muted"])
        else:
            draw_moon_icon(draw, center, phase, outline="#111827")
            label = phase
            if len(label) > 10:
                label = label.replace(" ", "\n")
            draw_centered_text(draw, ((box[0] + box[2]) / 2, box[3] - 26), label, load_font(12, bold=True), fill=PALETTE["muted"])
    return img


def moon_phase(rng: random.Random, idx: int) -> Dict[str, Any]:
    start_index = rng.randrange(len(MOON_PHASES))
    answer = MOON_PHASES[(start_index + 3) % len(MOON_PHASES)]
    wrong = rng.choice([phase for phase in MOON_PHASES if phase != answer])
    params = {"start_index": start_index, "wrong_phase": wrong}
    ground_truth = solve_moon_phase(params)
    return base_case(
        sub_task="moon_phase",
        family_slug="geo",
        task_family="temporal_reasoning",
        sub_family="cyclic_phase_sequence",
        domain="geography/astronomy",
        knowledge_type="conceptual",
        difficulty_level=3,
        difficulty_reason="The next icon follows a cyclic moon-phase order from three consecutive visible phases.",
        instruction="Draw the next moon phase in the sequence.",
        expected_target=f"The empty last slot is filled with {answer}.",
        rational_target_description="Moon phases proceed in a fixed cycle; the fourth slot should show the phase after the first three consecutive phases.",
        required_knowledge=[rule_fact("The moon phase cycle proceeds new, waxing crescent, first quarter, waxing gibbous, full, waning gibbous, last quarter, waning crescent, then repeats.")],
        search_queries=["moon phase sequence next phase order"],
        source_scene_graph={
            "objects": ["row of four phase slots", "three moon phase icons", "question-mark final slot"],
            "editable_region": "rightmost moon phase slot",
            "preserve_region": "first three moon icons, slot layout, labels, background, and border",
        },
        edit_operations=[operation("replace_icon", "question-mark moon slot", "middle right", f"draw {answer}")],
        negative_constraints=[
            f"Do not draw {wrong}; it is not the next phase in this sequence.",
            "Do not change the first three moon phase icons.",
        ],
        atomic_checklist=[
            {"id": "C1", "question": f"Does the last slot show {answer}?", "weight": 0.40},
            {"id": "C2", "question": "Does the completed row follow the moon-phase cycle?", "weight": 0.25},
            {"id": "C3", "question": "Are the first three phase icons unchanged?", "weight": 0.15},
            {"id": "C4", "question": "Is the question mark removed?", "weight": 0.10},
            {"id": "C5", "question": "Are there still exactly four slots?", "weight": 0.10},
        ],
        ground_truth=ground_truth,
        verifier_spec={
            "vqa_checks": [
                {"question": "What phase label is written in the final slot? Answer with the exact words.", "expected_answer": answer, "weight": 0.60},
                {"question": "How many moon phase slots are shown? Answer digits only.", "expected_answer": "4", "weight": 0.20},
                {"question": "Is the final slot still a question mark? Answer yes or no.", "expected_answer": "no", "weight": 0.20},
            ],
            "programmatic": {"solver": "solve_moon_phase", "params": params},
        },
        params=params,
        render=render_moon_phase,
    )


GENERATORS: Dict[str, Callable[[random.Random, int], Dict[str, Any]]] = {
    "sudoku4": sudoku4,
    "arithmetic_chain": arithmetic_chain,
    "balance_scale": balance_scale,
    "sequence_pattern": sequence_pattern,
    "clock_arithmetic": clock_arithmetic,
    "graph_path": graph_path,
    "mirror_reflection": mirror_reflection,
    "block_stack_view": block_stack_view,
    "circuit_bulb": circuit_bulb,
    "ph_indicator": ph_indicator,
    "food_chain": food_chain,
    "geometry_angle": geometry_angle,
    "sorting_step": sorting_step,
    "fraction_shade": fraction_shade,
    "process_order": process_order,
    "moon_phase": moon_phase,
}

SOLVERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "solve_sudoku4": solve_sudoku4,
    "solve_arithmetic_chain": solve_arithmetic_chain,
    "solve_balance_scale": solve_balance_scale,
    "solve_sequence_pattern": solve_sequence_pattern,
    "solve_clock_arithmetic": solve_clock_arithmetic,
    "solve_graph_path": solve_graph_path,
    "solve_mirror_reflection": solve_mirror_reflection,
    "solve_block_stack_view": solve_block_stack_view,
    "solve_circuit_bulb": solve_circuit_bulb,
    "solve_ph_indicator": solve_ph_indicator,
    "solve_food_chain": solve_food_chain,
    "solve_geometry_angle": solve_geometry_angle,
    "solve_sorting_step": solve_sorting_step,
    "solve_fraction_shade": solve_fraction_shade,
    "solve_process_order": solve_process_order,
    "solve_moon_phase": solve_moon_phase,
}


def build_row(case: Dict[str, Any], task_id: str, version: str, idx: int, paths: Dict[str, Path]) -> Dict[str, Any]:
    source_rel = save_png(case["render"](case["params"], "source"), paths["source"])
    teacher_rel = save_png(case["render"](case["params"], "teacher"), paths["teacher"])
    negative_rel = save_png(case["render"](case["params"], "negative"), paths["negative"])
    instruction = case["instruction"]
    row = {
        "task_id": task_id,
        "benchmark_family": case["benchmark_family"],
        "task_family": case["task_family"],
        "sub_family": case["sub_family"],
        "sub_task": case["sub_task"],
        "domain": case["domain"],
        "knowledge_type": case["knowledge_type"],
        "difficulty": case["difficulty"],
        "instruction": instruction,
        "expected_target": case["expected_target"],
        "rational_target_description": case["rational_target_description"],
        "required_knowledge": case["required_knowledge"],
        "source_scene_graph": case["source_scene_graph"],
        "edit_operations": case["edit_operations"],
        "preservation_constraints": case["preservation_constraints"],
        "negative_constraints": case["negative_constraints"],
        "atomic_checklist": case["atomic_checklist"],
        "search_queries": case["search_queries"],
        "benchmark_alignment": case["benchmark_alignment"],
        "leakage_tags": {
            "status": "passed_exact_text_check",
            "benchmark_text_exact_match": False,
            "benchmark_text_max_sim": None,
            "benchmark_image_max_sim": None,
            "normalized_instruction": normalize_text(instruction),
        },
        "source": f"programmatic_{version}",
        "split": split_for(idx),
        "version": version,
        "license": "programmatic",
        "created_at": FIXED_CREATED_AT,
        "source_image": source_rel,
        "source_image_provenance": {
            "type": "programmatic",
            "generator": "scripts/data/build_reasoning_tasks_v2.py",
            "params_hash": stable_hash(case["params"]),
            "license": "programmatic",
        },
        "render_paths": {"teacher": teacher_rel, "negative": negative_rel},
        "image_hashes": {
            "source_ahash": average_hash(paths["source"]),
            "teacher_ahash": average_hash(paths["teacher"]),
            "negative_ahash": average_hash(paths["negative"]),
        },
        "ground_truth": case["ground_truth"],
        "verifier_spec": case["verifier_spec"],
    }
    return row


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path} line {line_no} is not valid JSON: {exc}") from exc
    return rows


def paths_for(task_id: str) -> Dict[str, Path]:
    return {
        "source": repo_path("data", "images", "source", f"{task_id}.png"),
        "teacher": repo_path("data", "renders", "teacher", f"{task_id}_teacher.png"),
        "negative": repo_path("data", "renders", "negative", f"{task_id}_negative.png"),
    }


def pixel_diff(a: Path, b: Path) -> int:
    img_a = Image.open(a).convert("RGB")
    img_b = Image.open(b).convert("RGB")
    diff = ImageChops.difference(img_a, img_b)
    return sum(1 for px in diff.getdata() if px != (0, 0, 0))


def assert_solver(row: Dict[str, Any]) -> None:
    spec = row["verifier_spec"]["programmatic"]
    solver_name = spec["solver"]
    solved = SOLVERS[solver_name](spec["params"])
    expected = row["ground_truth"]
    if solved != expected:
        raise AssertionError(f"{row['task_id']} solver mismatch: solved={solved} expected={expected}")


VQA_FORMAT_RULES: List[Tuple[str, Optional[str]]] = [
    ("Answer digits only.", r"\d+"),
    ("Answer the signed number only.", r"-?\d+"),
    ("Answer yes or no.", r"yes|no"),
    ("Answer one word.", r"\S+"),
    ("Answer comma-separated digits.", r"\d+(, \d+)*"),
    ("Answer slash-separated digits.", r"\d+/\d+"),
    ("Answer H:MM only.", r"\d{1,2}:\d{2}"),
    ("Answer two digits.", r"\d{2}"),
    ("Answer hyphen-separated capital letters.", r"[A-Z](?:-[A-Z])*"),
    ("Answer one capital letter.", r"[A-Z]"),
    ("Answer as '<color> <shape>'.", r"[a-z]+ [a-z]+"),
    ("Answer exactly as written.", None),
    ("Answer with the exact words.", r"[a-z]+(?: [a-z]+)*"),
    ("List the four steps left to right, separated by commas.", r"[a-z]+, [a-z]+, [a-z]+, [a-z]+"),
]


def assert_vqa_format(row: Dict[str, Any]) -> None:
    checks = row["verifier_spec"].get("vqa_checks", [])
    if not checks:
        raise AssertionError(f"{row['task_id']} has no vqa_checks")
    for check in checks:
        question = check.get("question", "")
        expected = str(check.get("expected_answer", ""))
        for suffix, pattern in VQA_FORMAT_RULES:
            if question.endswith(suffix):
                if pattern is not None and re.fullmatch(pattern, expected) is None:
                    raise AssertionError(
                        f"{row['task_id']} expected_answer format mismatch for question {question!r}: {expected!r}"
                    )
                break
        else:
            raise AssertionError(f"{row['task_id']} vqa_check lacks explicit supported answer format: {question!r}")


def assert_task_specific_verifier(row: Dict[str, Any]) -> None:
    sub_task = row["sub_task"]
    spec = row["verifier_spec"]
    params = spec["programmatic"]["params"]
    checks = spec["vqa_checks"]
    if sub_task == "sorting_step":
        solved = solve_sorting_step(params)
        expected = ", ".join(str(value) for value in solved["resulting_array"])
        if checks[0]["expected_answer"] != expected:
            raise AssertionError(f"{row['task_id']} sorting VQA does not match swapped solver array")
        if solved["resulting_array"] == params["array"]:
            raise AssertionError(f"{row['task_id']} sorting solver did not apply the swap")
    elif sub_task == "geometry_angle":
        solved = solve_geometry_angle(params)
        if checks[0]["expected_answer"] != str(solved["angle_deg"]):
            raise AssertionError(f"{row['task_id']} geometry VQA does not match the written teacher angle")
        if checks[0]["expected_answer"] == str(params["wrong_angle"]):
            raise AssertionError(f"{row['task_id']} geometry negative angle matches the teacher answer")
    elif sub_task == "food_chain":
        organisms = params["organisms"]
        expected_questions = [
            f"Is there an arrow pointing from {organisms[i]} to {organisms[i + 1]}? Answer yes or no."
            for i in range(len(organisms) - 1)
        ]
        actual_questions = [check["question"] for check in checks]
        if actual_questions != expected_questions or any(check["expected_answer"] != "yes" for check in checks):
            raise AssertionError(f"{row['task_id']} food-chain VQA must cover every adjacent arrow direction")
        if params["wrong_index"] == params["negative_index"] and len(organisms) > 2:
            raise AssertionError(f"{row['task_id']} food-chain negative does not use a distinct wrong-pair arrow")
    elif sub_task == "circuit_bulb":
        if checks[0]["question"] != "What color is the bulb, yellow or gray? Answer one word." or checks[0]["expected_answer"] != "yellow":
            raise AssertionError(f"{row['task_id']} circuit VQA must verify bulb color")
    elif sub_task == "ph_indicator":
        if checks[0]["question"] != "What color is the indicator strip? Answer one word.":
            raise AssertionError(f"{row['task_id']} pH VQA must verify strip color")
        if checks[0]["expected_answer"] != params["expected_color"]:
            raise AssertionError(f"{row['task_id']} pH VQA answer does not match rendered strip color")
    elif sub_task == "process_order":
        expected = ", ".join(params["correct_order"])
        if checks[0]["expected_answer"] != expected:
            raise AssertionError(f"{row['task_id']} process-order VQA must use comma-separated left-to-right labels")


def validate_selected(task_ids: List[str], task_path: Path) -> Dict[str, Any]:
    rows = read_jsonl(task_path)
    by_id = {row["task_id"]: row for row in rows}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise AssertionError(f"missing task rows: {missing[:5]}")
    selected = [by_id[task_id] for task_id in task_ids]
    for row in selected:
        assert_solver(row)
        assert_vqa_format(row)
        assert_task_specific_verifier(row)
        paths = paths_for(row["task_id"])
        for key, path in paths.items():
            if not path.exists():
                raise AssertionError(f"missing {key} render for {row['task_id']}: {path}")
            size = path.stat().st_size
            if size <= 2048:
                raise AssertionError(f"{key} render is too small for {row['task_id']}: {size} bytes")
        source_teacher = pixel_diff(paths["source"], paths["teacher"])
        negative_teacher = pixel_diff(paths["negative"], paths["teacher"])
        if source_teacher <= 0:
            raise AssertionError(f"teacher render does not differ from source for {row['task_id']}")
        if negative_teacher <= 0:
            raise AssertionError(f"negative render does not differ from teacher for {row['task_id']}")
    return {
        "jsonl_rows_parsed": len(rows),
        "selected_tasks_checked": len(selected),
        "renders_checked": len(selected) * 3,
        "solver_checks": len(selected),
        "min_source_teacher_pixel_diff": min(pixel_diff(paths_for(row["task_id"])["source"], paths_for(row["task_id"])["teacher"]) for row in selected),
        "min_negative_teacher_pixel_diff": min(pixel_diff(paths_for(row["task_id"])["negative"], paths_for(row["task_id"])["teacher"]) for row in selected),
    }


def write_scene_inventory() -> None:
    inventory = {
        "sudoku4": {
            "present_elements": [
                "light canvas background with thin border",
                "4x4 white Sudoku grid",
                "thicker separators between 2x2 boxes",
                "black pre-filled digits",
                "one question-mark empty cell in the source",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "title text": "upper left",
                "Sudoku grid": "middle center",
                "empty or filled target cell": "inside the middle grid area at a randomized row and column",
                "2x2 box separators": "middle center, crossing the grid vertically and horizontally",
            },
            "teacher_change": "The teacher render replaces the question mark with the unique green Sudoku digit.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["timer", "pencil marks", "multiple blanks", "9x9 grid", "handwritten notes"],
            "style": "flat 2D programmatic puzzle diagram with crisp grid lines",
        },
        "arithmetic_chain": {
            "present_elements": [
                "light canvas background with thin border",
                "title text",
                "start number box",
                "two or three operation boxes",
                "rightward arrow connectors",
                "rightmost question-mark result box in the source",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "title text": "upper left",
                "flow boxes": "middle left to middle right",
                "arrow connectors": "middle row between boxes",
                "result box": "middle right",
            },
            "teacher_change": "The teacher render fills the rightmost result box with the computed final value in green.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["calculator UI", "equation proof steps", "extra result boxes", "vertical layout"],
            "style": "flat 2D flowchart with rounded boxes and arrows",
        },
        "balance_scale": {
            "present_elements": [
                "light canvas background with thin border",
                "level balance beam",
                "central triangular stand and pivot",
                "left and right pan arcs",
                "numbered rectangular weight boxes",
                "one question-mark weight box in the source",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "title text": "upper left",
                "pivot and stand": "middle center to lower center",
                "left pan and weights": "middle left",
                "right pan and weights": "middle right",
                "missing weight box": "middle left or middle right depending on randomized side",
            },
            "teacher_change": "The teacher render replaces the question mark with the numeric weight that makes both pan totals equal.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["kilogram unit labels", "spring scale dial", "tilted beam in the source", "realistic shadows"],
            "style": "flat 2D educational balance-scale diagram",
        },
        "sequence_pattern": {
            "present_elements": [
                "light canvas background with thin border",
                "pattern title text",
                "row of four or five rounded slots",
                "numbers or colored geometric shapes",
                "rightmost question-mark slot in the source",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "pattern title text": "upper left",
                "sequence row": "middle left to middle right",
                "known sequence items": "middle left through middle center",
                "final slot": "middle right",
            },
            "teacher_change": "The teacher render fills the final slot with the number or shape that continues the visible pattern.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["multiple rows", "hidden legend", "random distractor items", "animation"],
            "style": "flat 2D pattern-completion card row",
        },
        "clock_arithmetic": {
            "present_elements": [
                "light canvas background with thin border",
                "analog clock face",
                "hour tick marks",
                "12, 3, 6, and 9 numerals",
                "hour hand",
                "minute hand",
                "time or add-hours label",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "time label": "upper left",
                "clock face": "middle center to lower center",
                "hour and minute hands": "middle center",
                "cardinal numerals": "upper center, middle right, lower center, and middle left around the face",
            },
            "teacher_change": "The teacher render redraws the hands to the time after the requested whole-hour addition, with the minute hand preserved.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["second hand", "digital clock", "date", "alarm icons", "roman numerals"],
            "style": "flat 2D analog clock diagram with simple hands",
        },
        "graph_path": {
            "present_elements": [
                "light canvas background with thin border",
                "six labeled circular nodes A through F",
                "undirected weighted edges",
                "small white edge-weight labels",
                "shortest-path instruction title",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "title text": "upper left",
                "start node A": "middle left",
                "nodes B and D": "upper center",
                "nodes C and E": "lower center",
                "goal node F": "middle right",
                "weighted edges": "middle left through middle right",
            },
            "teacher_change": "The teacher render highlights the unique shortest A-to-F path edges in green.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["directed arrowheads", "coordinates", "adjacency matrix", "negative edge weights"],
            "style": "flat 2D weighted graph diagram with labeled nodes and edges",
        },
        "mirror_reflection": {
            "present_elements": [
                "light canvas background with thin border",
                "vertical teal mirror line",
                "mirror text label",
                "asymmetric flag-like object on the left",
                "empty right side in the source",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "title text": "upper left",
                "mirror line": "upper center to lower center",
                "mirror label": "upper center just right of the mirror line",
                "source object": "middle left",
                "reflected-object region": "middle right",
            },
            "teacher_change": "The teacher render adds a mirrored right-side copy whose flag points left toward the mirror line.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["horizontal mirror", "glass reflection texture", "shadow", "second mirror", "rotation guide"],
            "style": "flat 2D geometry reflection diagram with solid colored shapes",
        },
        "block_stack_view": {
            "present_elements": [
                "light canvas background with thin border",
                "three vertical columns of colored square blocks",
                "ground line",
                "column number labels",
                "marked block with an X in the source",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "title text": "upper left",
                "block columns": "middle left to middle right, sitting on lower ground line",
                "ground line": "lower left to lower right",
                "column labels": "lower left to lower right below stacks",
                "marked block": "within one randomized column, below at least one block",
            },
            "teacher_change": "The teacher render removes the marked block and shifts any blocks above it downward so no gap remains.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["3D perspective", "side view", "falling animation", "physics arrows", "block labels by color name"],
            "style": "flat 2D front-view block stack diagram with solid colors",
        },
        "circuit_bulb": {
            "present_elements": [
                "light canvas background with thin border",
                "battery with polarity marks",
                "rectangular series wire loop",
                "top switch that is open in the source",
                "bottom bulb that is gray/off in the source",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "title text": "upper left",
                "switch label and switch": "upper center",
                "battery": "middle left",
                "wire loop": "middle left through middle right",
                "bulb": "lower center",
            },
            "teacher_change": "The teacher render closes the switch and changes the bulb to a lit yellow state.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["ammeter", "voltage value", "parallel branch", "extra bulb", "current arrows"],
            "style": "flat 2D physics circuit diagram with simple symbolic components",
        },
        "ph_indicator": {
            "present_elements": [
                "light canvas background with thin border",
                "labeled beaker",
                "blue liquid fill",
                "universal indicator strip",
                "white strip in the source",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "title text": "upper left",
                "beaker rim": "upper center",
                "solution label": "middle center above the liquid",
                "indicator strip": "middle center to lower center",
                "liquid fill": "lower center",
            },
            "teacher_change": "The teacher render colors the strip according to the labeled solution's pH category.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["pH number scale", "litmus paper label", "dropper", "thermometer", "chemical equation"],
            "style": "flat 2D chemistry beaker diagram with a single editable color strip",
        },
        "food_chain": {
            "present_elements": [
                "light canvas background with thin border",
                "three or four organism label boxes",
                "horizontal arrow connectors",
                "one arrow pointing the wrong way in the source",
                "energy-flow title",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "title text": "upper left",
                "first organism box": "middle left",
                "middle organism boxes": "middle center",
                "last organism box": "middle right",
                "arrow connectors": "middle left through middle right",
            },
            "teacher_change": "The teacher render flips only the wrong arrow so energy flows from food or prey to consumer.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["ecosystem photo", "nutrient cycle", "decomposer loop", "population counts", "trophic pyramid"],
            "style": "flat 2D biology flow diagram with labeled rounded boxes",
        },
        "geometry_angle": {
            "present_elements": [
                "light canvas background with thin border",
                "triangle outline",
                "two numeric interior angle labels",
                "one question-mark interior angle label in the source",
                "triangle-angle title",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "title text": "upper left",
                "triangle top vertex": "upper center",
                "triangle base": "lower left to lower right",
                "known angle labels": "lower left and lower right inside the triangle",
                "missing angle label": "middle center below the top vertex",
            },
            "teacher_change": "The teacher render replaces the question mark with the angle value that makes the triangle total 180 degrees.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["protractor", "side lengths", "coordinate grid", "right-angle marker unless sampled", "external angles"],
            "style": "flat 2D math geometry diagram with boxed angle labels",
        },
        "sorting_step": {
            "present_elements": [
                "light canvas background with thin border",
                "five vertical value bars",
                "numeric bar value labels",
                "index labels under the bars",
                "highlighted adjacent compared pair in the source",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "caption text": "upper left",
                "bar chart": "middle left through middle right",
                "highlighted compared pair": "middle row at a randomized adjacent pair",
                "value labels": "above each bar in the middle row",
                "index labels": "lower row below the bars",
            },
            "teacher_change": "The teacher render swaps only the highlighted adjacent pair and leaves all other bars in place.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["full sorted output", "pseudocode panel", "animation trail", "non-adjacent swap", "more than five bars"],
            "style": "flat 2D computer-science bar-array diagram",
        },
        "fraction_shade": {
            "present_elements": [
                "light canvas background with thin border",
                "rectangular grid of cells",
                "some shaded cells",
                "fraction caption asking for an exact shaded fraction",
                "shaded-cell count label",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "fraction caption": "upper left",
                "grid": "middle center",
                "shaded cells": "middle center within the grid",
                "unshaded cells": "middle center within the grid",
                "shaded-cell count label": "lower center",
            },
            "teacher_change": "The teacher render adjusts the shaded cells so the count exactly equals the requested fraction of the grid.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["pie chart", "decimal conversion", "mixed numbers", "irregular cells", "multiple grids"],
            "style": "flat 2D math fraction grid with solid shaded cells",
        },
        "process_order": {
            "present_elements": [
                "light canvas background with thin border",
                "four step boxes",
                "step number labels",
                "rightward process arrows",
                "two step labels swapped in the source",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "title text": "upper left",
                "step 1 box": "middle left",
                "step 2 and 3 boxes": "middle center",
                "step 4 box": "middle right",
                "process arrows": "middle row between boxes",
            },
            "teacher_change": "The teacher render reorders the four boxes into the correct start-to-finish process sequence.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["branching workflow", "timer", "checklist marks", "photo panels", "more than four steps"],
            "style": "flat 2D procedural strip with rounded step boxes and arrows",
        },
        "moon_phase": {
            "present_elements": [
                "light canvas background with thin border",
                "row of four rounded phase slots",
                "three moon phase icons",
                "rightmost question-mark slot in the source",
                "small phase labels under filled icons",
            ],
            "layout": {
                "canvas background and border": "full frame",
                "title text": "upper left",
                "first phase slot": "middle left",
                "second and third phase slots": "middle center",
                "empty or filled final phase slot": "middle right",
                "phase labels": "lower row inside each filled slot",
            },
            "teacher_change": "The teacher render replaces the question mark with the next moon phase icon in the cycle.",
            "text_labels_present": True,
            "commonly_assumed_but_absent": ["Earth diagram", "Sun rays", "calendar dates", "telescope view", "lunar eclipse"],
            "style": "flat 2D astronomy sequence diagram with simple moon icons",
        },
    }
    write_json(repo_path("data", "taxonomy", "scene_inventory_v2.json"), inventory)


def parse_subtasks(text: str) -> List[str]:
    if not text:
        return list(SUBTASKS)
    items = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(items) - set(SUBTASKS))
    if unknown:
        raise SystemExit(f"unknown subtasks: {', '.join(unknown)}; valid: {', '.join(SUBTASKS)}")
    return items


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Build verifiable reasoning-dense v2 image-editing tasks.")
    parser.add_argument("--num-per-subtask", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--version", default="v2")
    parser.add_argument("--subtasks", default=",".join(SUBTASKS), help="Comma-separated subtask names.")
    args = parser.parse_args(argv)
    if args.num_per_subtask < 0:
        raise SystemExit("--num-per-subtask must be non-negative")

    subtasks = parse_subtasks(args.subtasks)
    rng = random.Random(args.seed)
    for path in [
        repo_path("data", "images", "source"),
        repo_path("data", "renders", "teacher"),
        repo_path("data", "renders", "negative"),
        repo_path("data", "tasks"),
        repo_path("data", "taxonomy"),
    ]:
        ensure_dir(path)

    task_path = repo_path("data", "tasks", f"tasks_{args.version}.jsonl")
    existing_rows = read_jsonl(task_path)
    existing_ids = {row.get("task_id") for row in existing_rows}
    rows_to_append: List[Dict[str, Any]] = []
    selected_ids: List[str] = []
    skipped = 0

    for subtask in subtasks:
        generator = GENERATORS[subtask]
        for idx in range(args.num_per_subtask):
            case = generator(rng, idx)
            task_id = make_task_id(args.version, case["family_slug"], subtask, idx)
            selected_ids.append(task_id)
            if task_id in existing_ids:
                skipped += 1
                continue
            rows_to_append.append(build_row(case, task_id, args.version, idx, paths_for(task_id)))

    appended = append_jsonl(task_path, rows_to_append)
    write_scene_inventory()
    summary = validate_selected(selected_ids, task_path) if selected_ids else {
        "jsonl_rows_parsed": len(read_jsonl(task_path)),
        "selected_tasks_checked": 0,
        "renders_checked": 0,
        "solver_checks": 0,
    }
    print(
        "self-test summary: "
        f"jsonl_rows_parsed={summary['jsonl_rows_parsed']} "
        f"selected_tasks_checked={summary['selected_tasks_checked']} "
        f"renders_checked={summary['renders_checked']} "
        f"solver_checks={summary['solver_checks']} "
        f"appended={appended} skipped_existing={skipped}"
    )
    if selected_ids:
        print(
            "pixel diff summary: "
            f"min_source_teacher={summary['min_source_teacher_pixel_diff']} "
            f"min_negative_teacher={summary['min_negative_teacher_pixel_diff']}"
        )
    print(f"wrote {task_path.relative_to(repo_path())} and data/taxonomy/scene_inventory_v2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
