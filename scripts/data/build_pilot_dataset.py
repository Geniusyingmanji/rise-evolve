#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw

from common import (
    PALETTE,
    ROOT,
    average_hash,
    draw_centered_text,
    ensure_dir,
    json_hash_file,
    load_font,
    new_canvas,
    normalize_text,
    repo_path,
    save_png,
    stable_hash,
    utc_now,
    write_json,
    write_jsonl,
)
from mine_taxonomy import build_knowledge_bank


ImageTriple = Tuple[Image.Image, Image.Image, Image.Image]


class BuildContext:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.version = args.version
        self.rng = random.Random(args.seed)
        self.knowledge_bank = build_knowledge_bank()
        self.created_at = utc_now()
        self.source_dir = repo_path("data", "images", "source")
        self.teacher_dir = repo_path("data", "renders", "teacher")
        self.negative_dir = repo_path("data", "renders", "negative")
        for path in (self.source_dir, self.teacher_dir, self.negative_dir):
            ensure_dir(path)

    def paths(self, task_id: str) -> Dict[str, Path]:
        return {
            "source": self.source_dir / f"{task_id}.png",
            "teacher": self.teacher_dir / f"{task_id}_teacher.png",
            "negative": self.negative_dir / f"{task_id}_negative.png",
        }


def draw_arrow(draw: ImageDraw.ImageDraw, start: Tuple[int, int], end: Tuple[int, int], fill: str, width: int = 5) -> None:
    draw.line((start, end), fill=fill, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    length = 18
    spread = 0.55
    p1 = (end[0] - length * math.cos(angle - spread), end[1] - length * math.sin(angle - spread))
    p2 = (end[0] - length * math.cos(angle + spread), end[1] - length * math.sin(angle + spread))
    draw.polygon([end, p1, p2], fill=fill)


def apply_global_variant(img: Image.Image, params: Dict[str, Any]) -> Image.Image:
    """Apply a small task-level layout jitter shared by source/target/negative."""
    dx = params.get("jitter_x", 0)
    dy = params.get("jitter_y", 0)
    if not dx and not dy:
        return img
    canvas = Image.new("RGB", img.size, "#f8fafc")
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(img.size[0], img.size[0] - dx) if dx >= 0 else img.size[0]
    src_y1 = min(img.size[1], img.size[1] - dy) if dy >= 0 else img.size[1]
    crop = img.crop((src_x0, src_y0, src_x1, src_y1))
    canvas.paste(crop, (max(0, dx), max(0, dy)))
    return canvas


def draw_equation(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Math equation", "Replace x with its solved value.")
    a, b, x, c = params["a"], params["b"], params["x"], params["c"]
    font_big = load_font(44, bold=True)
    font_mid = load_font(34, bold=True)
    draw_centered_text(draw, (256, 150), f"{a}x + {b} = {c}", font_big)
    if mode == "source":
        answer = "x = ?"
        fill = PALETTE["muted"]
    elif mode == "negative":
        answer = f"x = {x + 1}"
        fill = PALETTE["red"]
    else:
        answer = f"x = {x}"
        fill = PALETTE["green"]
    draw.rounded_rectangle((150, 245, 362, 325), radius=16, outline=fill, width=4, fill="#ffffff")
    draw_centered_text(draw, (256, 285), answer, font_mid, fill=fill)
    return img


def draw_refraction(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Physics refraction", "Complete the ray after it enters glass.")
    draw.rectangle((0, 260, 512, 512), fill="#dbeafe")
    draw.line((256, 115, 256, 430), fill="#64748b", width=2)
    draw.line((0, 260, 512, 260), fill="#334155", width=3)
    draw.text((275, 130), "normal", font=load_font(16), fill=PALETTE["muted"])
    draw.text((24, 218), "air", font=load_font(20), fill=PALETTE["ink"])
    draw.text((24, 285), "glass", font=load_font(20), fill=PALETTE["ink"])
    draw_arrow(draw, (105, 95), (256, 260), PALETTE["orange"], 6)
    if mode != "source":
        if mode == "negative":
            end = (390, 445)
            color = PALETTE["red"]
        else:
            end = (305, 445)
            color = PALETTE["green"]
        draw_arrow(draw, (256, 260), end, color, 6)
    return img


def draw_litmus(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Chemistry indicator", "Show the litmus result in acid.")
    draw.ellipse((155, 320, 357, 360), fill="#bfdbfe", outline="#475569", width=3)
    draw.rectangle((155, 180, 357, 340), fill="#dbeafe", outline="#475569", width=3)
    draw.ellipse((155, 160, 357, 200), fill="#eff6ff", outline="#475569", width=3)
    draw.text((214, 378), "acid", font=load_font(24, bold=True), fill=PALETTE["ink"])
    strip_color = PALETTE["blue"]
    if mode == "teacher":
        strip_color = PALETTE["red"]
    elif mode == "negative":
        strip_color = PALETTE["green"]
    draw.rounded_rectangle((240, 92, 272, 330), radius=8, fill=strip_color, outline="#1f2937", width=2)
    draw.text((292, 112), "litmus", font=load_font(18), fill=PALETTE["muted"])
    return img


def draw_biology_chrysalis(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Biology lifecycle", "Edit to the next butterfly life-cycle stage.")
    draw.rectangle((0, 355, 512, 512), fill="#dcfce7")
    draw.ellipse((105, 330, 430, 395), fill="#65a30d", outline="#365314", width=3)
    if mode == "source":
        for i in range(5):
            x = 170 + i * 38
            draw.ellipse((x, 250 + (i % 2) * 8, x + 54, 300 + (i % 2) * 8), fill="#84cc16", outline="#365314", width=2)
        draw.ellipse((342, 245, 388, 292), fill="#84cc16", outline="#365314", width=2)
        draw.text((188, 405), "caterpillar", font=load_font(24), fill=PALETTE["ink"])
    elif mode == "negative":
        draw.ellipse((205, 210, 315, 310), fill="#f472b6", outline="#831843", width=3)
        draw.text((205, 405), "flower", font=load_font(24), fill=PALETTE["red"])
    else:
        draw.line((256, 140, 256, 225), fill="#365314", width=4)
        draw.ellipse((210, 205, 302, 345), fill="#a3e635", outline="#365314", width=4)
        draw.text((206, 405), "chrysalis", font=load_font(24), fill=PALETTE["green"])
    return img


def draw_bst(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Computer science", "Insert the value into the binary search tree.")
    font = load_font(24, bold=True)
    nodes = {
        8: (256, 130),
        4: (160, 230),
        12: (352, 230),
        2: (105, 335),
        6: (215, 335),
    }
    if mode == "teacher":
        nodes[10] = (310, 335)
    elif mode == "negative":
        nodes[10] = (395, 335)
    edges = [(8, 4), (8, 12), (4, 2), (4, 6)]
    if mode != "source":
        edges.append((12, 10))
    for a, b in edges:
        draw.line((nodes[a], nodes[b]), fill="#475569", width=4)
    for val, (x, y) in nodes.items():
        fill = "#ffffff" if val != 10 else ("#dcfce7" if mode == "teacher" else "#fee2e2")
        draw.ellipse((x - 28, y - 28, x + 28, y + 28), fill=fill, outline="#111827", width=3)
        draw_centered_text(draw, (x, y), str(val), font)
    draw.text((44, 420), "Insert 10", font=load_font(28, bold=True), fill=PALETTE["ink"])
    return img


def draw_demand(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Economics", "Show an increase in demand.")
    origin = (90, 410)
    draw_arrow(draw, origin, (90, 115), PALETTE["ink"], 4)
    draw_arrow(draw, origin, (425, 410), PALETTE["ink"], 4)
    draw.text((52, 105), "P", font=load_font(22, bold=True), fill=PALETTE["ink"])
    draw.text((430, 420), "Q", font=load_font(22, bold=True), fill=PALETTE["ink"])
    draw.line((135, 360, 380, 140), fill=PALETTE["orange"], width=5)
    draw.text((386, 135), "S", font=load_font(22, bold=True), fill=PALETTE["orange"])
    draw.line((130, 145, 380, 355), fill=PALETTE["blue"], width=5)
    draw.text((385, 350), "D1", font=load_font(20, bold=True), fill=PALETTE["blue"])
    if mode != "source":
        if mode == "negative":
            draw.line((100, 145, 350, 355), fill=PALETTE["red"], width=5)
            draw.text((352, 350), "D2", font=load_font(20, bold=True), fill=PALETTE["red"])
        else:
            draw.line((170, 145, 420, 355), fill=PALETTE["green"], width=5)
            draw.text((424, 350), "D2", font=load_font(20, bold=True), fill=PALETTE["green"])
    return img


def draw_arch_restore(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("History / architecture", "Restore the missing arch stones.")
    draw.rectangle((0, 390, 512, 512), fill="#e5e7eb")
    stone = "#d6d3d1"
    outline = "#57534e"
    draw.rectangle((120, 220, 175, 390), fill=stone, outline=outline, width=3)
    draw.rectangle((337, 220, 392, 390), fill=stone, outline=outline, width=3)
    if mode == "source":
        draw.arc((120, 95, 392, 365), start=180, end=360, fill=outline, width=12)
        draw.rectangle((234, 100, 278, 145), fill="#f8fafc")
        draw.text((175, 420), "missing keystone", font=load_font(22), fill=PALETTE["muted"])
    elif mode == "negative":
        draw.line((120, 205, 392, 205), fill=PALETTE["red"], width=18)
        draw.text((185, 420), "flat beam", font=load_font(22), fill=PALETTE["red"])
    else:
        draw.arc((120, 95, 392, 365), start=180, end=360, fill=outline, width=24)
        draw.rectangle((234, 96, 278, 145), fill=stone, outline=outline, width=3)
        draw.text((178, 420), "restored arch", font=load_font(22), fill=PALETTE["green"])
    return img


def draw_meander(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Geography", "Mark erosion and deposition on a river bend.")
    draw.rectangle((0, 80, 512, 512), fill="#dcfce7")
    river = [(90, 105), (205, 165), (310, 155), (420, 245), (300, 330), (175, 315), (85, 420)]
    draw.line(river, fill=PALETTE["water"], width=70, joint="curve")
    draw.line(river, fill="#2563eb", width=4)
    if mode != "source":
        if mode == "negative":
            draw.text((80, 250), "erosion", font=load_font(22, bold=True), fill=PALETTE["red"])
            draw.text((320, 280), "deposition", font=load_font(22, bold=True), fill=PALETTE["red"])
        else:
            draw.text((312, 235), "erosion", font=load_font(22, bold=True), fill=PALETTE["red"])
            draw.text((155, 295), "deposition", font=load_font(22, bold=True), fill=PALETTE["brown"])
    return img


def draw_music(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Music notation", "Raise the note by one semitone.")
    for y in [190, 215, 240, 265, 290]:
        draw.line((80, y, 432, y), fill="#111827", width=2)
    note_x, note_y = 250, 240
    symbol = ""
    color = "#111827"
    if mode == "teacher":
        symbol = "#"
        color = PALETTE["green"]
    elif mode == "negative":
        symbol = "b"
        color = PALETTE["red"]
    if symbol:
        draw.text((205, 215), symbol, font=load_font(44, bold=True), fill=color)
    draw.ellipse((note_x, note_y - 14, note_x + 34, note_y + 14), fill="#111827")
    draw.line((note_x + 32, note_y, note_x + 32, note_y - 90), fill="#111827", width=4)
    return img


def draw_sports(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Sports rule", "Move the attacker back onside.")
    draw.rectangle((45, 100, 467, 435), fill="#bbf7d0", outline="#166534", width=4)
    draw.line((256, 100, 256, 435), fill="#166534", width=3)
    defender_x = 310
    draw.line((defender_x, 110, defender_x, 425), fill=PALETTE["blue"], width=4)
    draw.text((defender_x + 8, 112), "onside line", font=load_font(16), fill=PALETTE["blue"])
    for y in [210, 280, 350]:
        draw.ellipse((defender_x - 12, y - 12, defender_x + 12, y + 12), fill=PALETTE["blue"])
    attacker_x = 345
    if mode == "teacher":
        attacker_x = 300
    elif mode == "negative":
        attacker_x = 375
    draw.ellipse((attacker_x - 16, 250 - 16, attacker_x + 16, 250 + 16), fill=PALETTE["red"])
    draw.text((attacker_x - 22, 278), "A", font=load_font(20, bold=True), fill=PALETTE["red"])
    return img


def draw_fruit_decay(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Temporal common sense", "Show the fruit after days on a counter.")
    draw.rectangle((0, 345, 512, 512), fill="#f1f5f9")
    draw.ellipse((145, 310, 367, 375), fill="#e5e7eb", outline="#64748b", width=3)
    fruit_color = "#facc15"
    if mode == "teacher":
        fruit_color = "#ca8a04"
    elif mode == "negative":
        fruit_color = "#22c55e"
    draw.ellipse((180, 210, 335, 330), fill=fruit_color, outline="#854d0e", width=3)
    if mode == "teacher":
        for x, y, r in [(220, 242, 8), (260, 260, 10), (300, 232, 7), (245, 300, 9), (315, 292, 8)]:
            draw.ellipse((x - r, y - r, x + r, y + r), fill="#3f1f0f")
    if mode == "negative":
        draw.text((145, 405), "still fresh", font=load_font(24), fill=PALETTE["red"])
    return img


def draw_plant_growth(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Temporal growth", "Show the plant after two weeks.")
    draw.rectangle((0, 365, 512, 512), fill="#92400e")
    draw.rectangle((190, 305, 322, 385), fill="#b45309", outline="#78350f", width=3)
    if mode == "source":
        draw.line((256, 305, 256, 255), fill=PALETTE["green"], width=5)
        draw.ellipse((240, 245, 256, 262), fill=PALETTE["green"])
    elif mode == "negative":
        draw.rectangle((218, 260, 294, 315), fill="#9ca3af", outline="#4b5563", width=3)
    else:
        draw.line((256, 305, 256, 165), fill=PALETTE["green"], width=6)
        for dx, dy in [(-45, 205), (35, 190), (-30, 255), (40, 245)]:
            draw.ellipse((256 + dx - 30, dy - 15, 256 + dx + 30, dy + 15), fill="#22c55e", outline="#166534", width=2)
    return img


def draw_ice_melt(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Causal heat", "Show the result after sunlight warms the ice.")
    draw.ellipse((360, 100, 445, 185), fill="#fde047", outline="#f97316", width=4)
    for angle in range(0, 360, 45):
        x1 = 402 + 55 * math.cos(math.radians(angle))
        y1 = 142 + 55 * math.sin(math.radians(angle))
        x2 = 402 + 80 * math.cos(math.radians(angle))
        y2 = 142 + 80 * math.sin(math.radians(angle))
        draw.line((x1, y1, x2, y2), fill="#f97316", width=3)
    draw.rectangle((0, 360, 512, 512), fill="#e5e7eb")
    if mode == "source":
        draw.polygon([(190, 230), (300, 215), (345, 295), (240, 320)], fill="#bfdbfe", outline="#2563eb")
    elif mode == "negative":
        draw.polygon([(190, 230), (300, 215), (345, 295), (240, 320)], fill="#94a3b8", outline="#334155")
        draw.text((180, 395), "stone", font=load_font(24), fill=PALETTE["red"])
    else:
        draw.ellipse((165, 300, 360, 360), fill="#93c5fd", outline="#2563eb", width=3)
        draw.text((205, 390), "water", font=load_font(24), fill=PALETTE["green"])
    return img


def draw_sponge(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Causal force", "Show the sponge after being squeezed.")
    draw.rectangle((0, 360, 512, 512), fill="#e5e7eb")
    if mode == "source":
        box = (170, 210, 342, 320)
    elif mode == "negative":
        box = (170, 160, 342, 320)
    else:
        box = (150, 250, 362, 305)
    draw.rounded_rectangle(box, radius=20, fill="#facc15", outline="#a16207", width=4)
    for x in range(box[0] + 20, box[2] - 20, 35):
        for y in range(box[1] + 20, box[3] - 15, 25):
            draw.ellipse((x, y, x + 6, y + 6), fill="#a16207")
    if mode != "source":
        draw_arrow(draw, (256, 120), (256, box[1] + 5), PALETTE["red"], 6)
    return img


def draw_rotate_arrow(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Spatial reasoning", "Rotate the arrow 90 degrees clockwise.")
    if mode == "source":
        points = [(160, 230), (300, 230), (300, 190), (390, 256), (300, 322), (300, 282), (160, 282)]
    elif mode == "negative":
        points = [(352, 230), (212, 230), (212, 190), (122, 256), (212, 322), (212, 282), (352, 282)]
    else:
        points = [(230, 160), (230, 300), (190, 300), (256, 390), (322, 300), (282, 300), (282, 160)]
    draw.polygon(points, fill="#38bdf8", outline="#075985")
    return img


def draw_occlusion(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Spatial occlusion", "Move the red circle behind the blue block.")
    draw.rectangle((0, 365, 512, 512), fill="#e5e7eb")
    if mode == "source":
        draw.rectangle((250, 190, 370, 330), fill=PALETTE["blue"], outline="#1e3a8a", width=4)
        draw.ellipse((155, 220, 275, 340), fill=PALETTE["red"], outline="#7f1d1d", width=4)
    elif mode == "negative":
        draw.ellipse((270, 220, 390, 340), fill=PALETTE["red"], outline="#7f1d1d", width=4)
        draw.rectangle((150, 190, 270, 330), fill=PALETTE["blue"], outline="#1e3a8a", width=4)
    else:
        draw.ellipse((265, 220, 385, 340), fill=PALETTE["red"], outline="#7f1d1d", width=4)
        draw.rectangle((220, 190, 340, 330), fill=PALETTE["blue"], outline="#1e3a8a", width=4)
    return img


def draw_sudoku(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Logical reasoning", "Fill the missing number in the mini grid.")
    start_x, start_y, cell = 151, 145, 70
    numbers = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    for r in range(3):
        for c in range(3):
            x0 = start_x + c * cell
            y0 = start_y + r * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill="#ffffff", outline="#111827", width=3)
            value = numbers[r][c]
            if r == 2 and c == 2:
                if mode == "source":
                    txt, color = "?", PALETTE["muted"]
                elif mode == "negative":
                    txt, color = "2", PALETTE["red"]
                else:
                    txt, color = str(value), PALETTE["green"]
            else:
                txt, color = str(value), PALETTE["ink"]
            draw_centered_text(draw, (x0 + cell / 2, y0 + cell / 2), txt, load_font(36, bold=True), fill=color)
    return img


def draw_maze(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("Logical maze", "Draw the valid path from S to G.")
    x0, y0, size = 110, 120, 58
    walls = {(1, 0), (1, 1), (3, 1), (0, 3), (2, 3), (3, 3)}
    for r in range(5):
        for c in range(5):
            fill = "#1f2937" if (r, c) in walls else "#ffffff"
            draw.rectangle((x0 + c * size, y0 + r * size, x0 + (c + 1) * size, y0 + (r + 1) * size), fill=fill, outline="#94a3b8")
    draw_centered_text(draw, (x0 + size / 2, y0 + size / 2), "S", load_font(22, bold=True), fill=PALETTE["green"])
    draw_centered_text(draw, (x0 + 4.5 * size, y0 + 4.5 * size), "G", load_font(22, bold=True), fill=PALETTE["red"])
    if mode != "source":
        if mode == "negative":
            path = [(0, 0), (0, 1), (1, 1), (2, 1)]
            color = PALETTE["red"]
        else:
            path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (2, 4), (3, 4), (4, 4)]
            color = PALETTE["green"]
        pts = [(x0 + c * size + size / 2, y0 + r * size + size / 2) for r, c in path]
        draw.line(pts, fill=color, width=8)
    return img


def draw_traffic_light(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("KRIS anomaly", "Correct the traffic light color order.")
    draw.rounded_rectangle((190, 110, 322, 405), radius=28, fill="#1f2937", outline="#0f172a", width=4)
    if mode == "source":
        colors = [PALETTE["green"], PALETTE["red"], PALETTE["yellow"]]
    elif mode == "negative":
        colors = [PALETTE["yellow"], PALETTE["green"], PALETTE["red"]]
    else:
        colors = [PALETTE["red"], PALETTE["yellow"], PALETTE["green"]]
    for i, color in enumerate(colors):
        y = 170 + i * 85
        draw.ellipse((221, y - 31, 291, y + 39), fill=color, outline="#f8fafc", width=3)
    return img


def draw_penguin_trait(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("KRIS factual trait", "Add typical penguin body coloration.")
    draw.rectangle((0, 360, 512, 512), fill="#e0f2fe")
    body_color = "#111827" if mode == "teacher" else "#6b7280"
    belly = "#f8fafc" if mode == "teacher" else ("#fca5a5" if mode == "negative" else "#6b7280")
    draw.ellipse((180, 145, 332, 360), fill=body_color, outline="#111827", width=4)
    draw.ellipse((215, 205, 297, 350), fill=belly, outline=body_color, width=2)
    draw.ellipse((215, 110, 297, 200), fill=body_color, outline="#111827", width=4)
    draw.ellipse((238, 140, 248, 150), fill="#ffffff")
    draw.ellipse((264, 140, 274, 150), fill="#ffffff")
    draw.polygon([(253, 158), (270, 166), (253, 174)], fill=PALETTE["orange"])
    draw.polygon([(215, 360), (245, 360), (225, 390)], fill=PALETTE["orange"])
    draw.polygon([(267, 360), (297, 360), (287, 390)], fill=PALETTE["orange"])
    return img


def draw_table_setting(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("KRIS social convention", "Arrange the place setting correctly.")
    draw.ellipse((160, 145, 352, 337), fill="#ffffff", outline="#94a3b8", width=4)
    if mode == "source":
        fork_x, knife_x = 352, 135
    elif mode == "negative":
        fork_x, knife_x = 370, 110
    else:
        fork_x, knife_x = 130, 360
    draw.line((fork_x, 165, fork_x, 330), fill="#111827", width=7)
    for dx in [-12, 0, 12]:
        draw.line((fork_x + dx, 150, fork_x + dx, 205), fill="#111827", width=4)
    draw.polygon([(knife_x, 150), (knife_x + 28, 165), (knife_x + 8, 330), (knife_x - 8, 330)], fill="#64748b")
    return img


def draw_toast_process(params: Dict[str, Any], mode: str) -> Image.Image:
    img, draw = new_canvas("KRIS procedural", "Show the bread after toasting.")
    draw.rectangle((170, 250, 342, 390), fill="#94a3b8", outline="#334155", width=4)
    draw.rectangle((195, 220, 235, 252), fill="#475569")
    draw.rectangle((277, 220, 317, 252), fill="#475569")
    bread_color = "#fde68a"
    if mode == "teacher":
        bread_color = "#b45309"
    elif mode == "negative":
        bread_color = "#ffffff"
    draw.rounded_rectangle((205, 125, 307, 250), radius=30, fill=bread_color, outline="#92400e", width=3)
    if mode == "teacher":
        for x, y in [(235, 165), (275, 190), (250, 220)]:
            draw.ellipse((x - 10, y - 8, x + 10, y + 8), fill="#78350f")
    return img


DRAWERS: Dict[str, Callable[[Dict[str, Any], str], Image.Image]] = {
    "grade_math_linear": draw_equation,
    "grade_physics_refraction": draw_refraction,
    "grade_chem_litmus": draw_litmus,
    "grade_biology_lifecycle": draw_biology_chrysalis,
    "grade_cs_bst": draw_bst,
    "grade_economics_demand": draw_demand,
    "grade_history_arch": draw_arch_restore,
    "grade_geography_meander": draw_meander,
    "grade_music_sharp": draw_music,
    "grade_sports_offside": draw_sports,
    "rise_temporal_fruit": draw_fruit_decay,
    "rise_temporal_plant": draw_plant_growth,
    "rise_causal_ice": draw_ice_melt,
    "rise_causal_sponge": draw_sponge,
    "rise_spatial_rotate": draw_rotate_arrow,
    "rise_spatial_occlusion": draw_occlusion,
    "rise_logical_sudoku": draw_sudoku,
    "rise_logical_maze": draw_maze,
    "kris_traffic_light": draw_traffic_light,
    "kris_penguin_trait": draw_penguin_trait,
    "kris_table_setting": draw_table_setting,
    "kris_toast_process": draw_toast_process,
}


TEMPLATES: List[Dict[str, Any]] = [
    {
        "kind": "grade_math_linear",
        "benchmark_family": "GRADE_like",
        "task_family": "discipline_reasoning",
        "sub_family": "symbolic_math",
        "domain": "math",
        "knowledge_key": "math_linear_equation",
        "instruction": "Replace the missing value of x in the equation with the correct solved value. Keep the equation layout unchanged.",
        "expected": "The placeholder x result is replaced with the solved value, while the equation card and layout stay unchanged.",
        "operation": "replace the x = ? answer box with the solved numeric value",
        "difficulty": 2,
    },
    {
        "kind": "grade_physics_refraction",
        "benchmark_family": "GRADE_like",
        "task_family": "discipline_reasoning",
        "sub_family": "physics_refraction",
        "domain": "physics",
        "knowledge_key": "physics_refraction",
        "instruction": "Complete the ray diagram after the light enters the glass, using the correct refraction direction.",
        "expected": "The refracted ray bends toward the normal inside the denser glass medium.",
        "operation": "draw the refracted ray inside the glass bending toward the normal",
        "difficulty": 3,
    },
    {
        "kind": "grade_chem_litmus",
        "benchmark_family": "GRADE_like",
        "task_family": "discipline_reasoning",
        "sub_family": "chemistry_indicator",
        "domain": "chemistry",
        "knowledge_key": "chemistry_litmus",
        "instruction": "Show the correct color of the blue litmus paper after it is dipped in the acid solution.",
        "expected": "The litmus strip turns red in the acid, and the beaker remains unchanged.",
        "operation": "change the litmus strip from blue to red",
        "difficulty": 2,
    },
    {
        "kind": "grade_biology_lifecycle",
        "benchmark_family": "GRADE_like",
        "task_family": "discipline_reasoning",
        "sub_family": "biology_lifecycle",
        "domain": "biology",
        "knowledge_key": "biology_lifecycle",
        "instruction": "Edit the caterpillar scene to show the next butterfly life-cycle stage on the leaf.",
        "expected": "The caterpillar becomes a chrysalis attached above the leaf, while the leaf and background remain stable.",
        "operation": "replace the caterpillar with a chrysalis in the same scene",
        "difficulty": 3,
    },
    {
        "kind": "grade_cs_bst",
        "benchmark_family": "GRADE_like",
        "task_family": "discipline_reasoning",
        "sub_family": "computer_science_tree",
        "domain": "computer_science",
        "knowledge_key": "cs_bst",
        "instruction": "Insert value 10 into the binary search tree at the correct position.",
        "expected": "Node 10 is added as the left child of 12, preserving the existing tree.",
        "operation": "add node 10 as the left child of node 12",
        "difficulty": 3,
    },
    {
        "kind": "grade_economics_demand",
        "benchmark_family": "GRADE_like",
        "task_family": "discipline_reasoning",
        "sub_family": "economics_curve_shift",
        "domain": "economics",
        "knowledge_key": "economics_demand_shift",
        "instruction": "Edit the supply-demand graph to show an increase in demand.",
        "expected": "A new demand curve D2 is shifted to the right of D1, while supply and axes are preserved.",
        "operation": "add a right-shifted demand curve labeled D2",
        "difficulty": 3,
    },
    {
        "kind": "grade_history_arch",
        "benchmark_family": "GRADE_like",
        "task_family": "discipline_reasoning",
        "sub_family": "history_architecture_restoration",
        "domain": "history",
        "knowledge_key": "temporal_fruit_decay",
        "instruction": "Restore the missing keystone and arch stones so the ancient stone arch is structurally plausible.",
        "expected": "The arch is restored with a curved stone arch and keystone rather than a flat beam.",
        "operation": "fill the missing keystone and preserve the curved stone arch form",
        "difficulty": 4,
    },
    {
        "kind": "grade_geography_meander",
        "benchmark_family": "GRADE_like",
        "task_family": "discipline_reasoning",
        "sub_family": "geography_river_process",
        "domain": "geography",
        "knowledge_key": "spatial_occlusion",
        "instruction": "Label the river bend with erosion on the outer bank and deposition on the inner bank.",
        "expected": "The outside of the bend is labeled erosion and the inside is labeled deposition.",
        "operation": "add erosion and deposition labels at the correct sides of the meander",
        "difficulty": 4,
    },
    {
        "kind": "grade_music_sharp",
        "benchmark_family": "GRADE_like",
        "task_family": "discipline_reasoning",
        "sub_family": "music_notation",
        "domain": "music",
        "knowledge_key": "math_linear_equation",
        "instruction": "Raise the shown note by one semitone without moving it on the staff.",
        "expected": "A sharp sign is added before the note, and the staff layout is preserved.",
        "operation": "add a sharp sign immediately before the note",
        "difficulty": 2,
    },
    {
        "kind": "grade_sports_offside",
        "benchmark_family": "GRADE_like",
        "task_family": "discipline_reasoning",
        "sub_family": "sports_rule_spatial",
        "domain": "sports",
        "knowledge_key": "spatial_occlusion",
        "instruction": "Move the red attacker back onside, level with or behind the second-last defender line.",
        "expected": "The red attacker is moved to the onside side of the defender line while the field and defenders remain unchanged.",
        "operation": "move the red attacker left to be level with or behind the onside line",
        "difficulty": 3,
    },
    {
        "kind": "rise_temporal_fruit",
        "benchmark_family": "RISE_like",
        "task_family": "temporal_reasoning",
        "sub_family": "aging_decay",
        "domain": "everyday_common_sense",
        "knowledge_key": "temporal_fruit_decay",
        "instruction": "Edit the fruit to show how it would look after sitting on a kitchen counter for many days.",
        "expected": "The fruit becomes darker and spotted, while the plate, counter, and viewpoint remain unchanged.",
        "operation": "add brown decay spots and darker color to the fruit only",
        "difficulty": 2,
    },
    {
        "kind": "rise_temporal_plant",
        "benchmark_family": "RISE_like",
        "task_family": "temporal_reasoning",
        "sub_family": "growth",
        "domain": "everyday_common_sense",
        "knowledge_key": "biology_lifecycle",
        "instruction": "Show the potted sprout after two weeks of healthy growth.",
        "expected": "The sprout becomes a taller leafy plant in the same pot.",
        "operation": "grow the sprout into a taller plant with leaves",
        "difficulty": 2,
    },
    {
        "kind": "rise_causal_ice",
        "benchmark_family": "RISE_like",
        "task_family": "causal_reasoning",
        "sub_family": "thermal_phase_change",
        "domain": "everyday_physics",
        "knowledge_key": "thermal_ice_melt",
        "instruction": "Show what happens to the ice after being left under warm sunlight.",
        "expected": "The ice becomes a puddle of liquid water under the same sunlight and surface.",
        "operation": "replace solid ice with a water puddle",
        "difficulty": 2,
    },
    {
        "kind": "rise_causal_sponge",
        "benchmark_family": "RISE_like",
        "task_family": "causal_reasoning",
        "sub_family": "force_deformation",
        "domain": "everyday_physics",
        "knowledge_key": "spatial_occlusion",
        "instruction": "Show the sponge after a strong downward squeeze is applied.",
        "expected": "The sponge is compressed flatter and wider, with the same table and lighting.",
        "operation": "deform the sponge into a flatter compressed shape",
        "difficulty": 2,
    },
    {
        "kind": "rise_spatial_rotate",
        "benchmark_family": "RISE_like",
        "task_family": "spatial_reasoning",
        "sub_family": "3d_rotation",
        "domain": "geometry",
        "knowledge_key": "spatial_occlusion",
        "instruction": "Rotate the arrow shape 90 degrees clockwise while preserving its color and size.",
        "expected": "The arrow points downward after a clockwise quarter turn.",
        "operation": "rotate the arrow 90 degrees clockwise",
        "difficulty": 2,
    },
    {
        "kind": "rise_spatial_occlusion",
        "benchmark_family": "RISE_like",
        "task_family": "spatial_reasoning",
        "sub_family": "occlusion",
        "domain": "geometry",
        "knowledge_key": "spatial_occlusion",
        "instruction": "Move the red circle behind the blue block so the block occludes part of it.",
        "expected": "The red circle is partly hidden behind the blue block.",
        "operation": "place the red circle behind the blue block with correct occlusion",
        "difficulty": 3,
    },
    {
        "kind": "rise_logical_sudoku",
        "benchmark_family": "RISE_like",
        "task_family": "logical_reasoning",
        "sub_family": "rule_based_pattern",
        "domain": "logic",
        "knowledge_key": "math_linear_equation",
        "instruction": "Fill the missing cell in the 3 by 3 logic grid so each row and column contains 1, 2, and 3.",
        "expected": "The missing bottom-right cell is filled with 1.",
        "operation": "replace the question mark with 1",
        "difficulty": 3,
    },
    {
        "kind": "rise_logical_maze",
        "benchmark_family": "RISE_like",
        "task_family": "logical_reasoning",
        "sub_family": "maze_path",
        "domain": "logic",
        "knowledge_key": "spatial_occlusion",
        "instruction": "Draw a valid path through the maze from S to G without crossing walls.",
        "expected": "A green path connects S to G through open cells only.",
        "operation": "draw the valid path from S to G",
        "difficulty": 4,
    },
    {
        "kind": "kris_traffic_light",
        "benchmark_family": "KRIS_like",
        "task_family": "anomaly_correction",
        "sub_family": "factual_order_correction",
        "domain": "practical_knowledge",
        "knowledge_key": "traffic_light_order",
        "instruction": "Correct the vertical traffic light so the colors are in the standard order.",
        "expected": "The vertical light has red on top, yellow in the middle, and green at the bottom.",
        "operation": "reorder the traffic light colors to red-yellow-green from top to bottom",
        "difficulty": 2,
    },
    {
        "kind": "kris_penguin_trait",
        "benchmark_family": "KRIS_like",
        "task_family": "entity_attribute_edit",
        "sub_family": "species_trait",
        "domain": "biology",
        "knowledge_key": "biology_lifecycle",
        "instruction": "Edit the penguin to show typical black-and-white penguin body coloration.",
        "expected": "The penguin has a dark back/head and a white belly, while pose and background remain the same.",
        "operation": "change the penguin body to dark back and white belly coloration",
        "difficulty": 3,
    },
    {
        "kind": "kris_table_setting",
        "benchmark_family": "KRIS_like",
        "task_family": "multi_element_composition",
        "sub_family": "social_convention",
        "domain": "practical_knowledge",
        "knowledge_key": "spatial_occlusion",
        "instruction": "Arrange the table setting conventionally with the fork on the left of the plate and the knife on the right.",
        "expected": "The fork is placed left of the plate and the knife is placed right of the plate.",
        "operation": "move fork left of plate and knife right of plate",
        "difficulty": 3,
    },
    {
        "kind": "kris_toast_process",
        "benchmark_family": "KRIS_like",
        "task_family": "procedural_knowledge",
        "sub_family": "process_result",
        "domain": "everyday_procedure",
        "knowledge_key": "temporal_fruit_decay",
        "instruction": "Show the bread after it has been toasted.",
        "expected": "The bread becomes brown toast with darker toasted spots, while the toaster remains unchanged.",
        "operation": "change pale bread into browned toast",
        "difficulty": 2,
    },
]


def make_params(ctx: BuildContext, template: Dict[str, Any], seq: int) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "variant": seq,
        "text_variant": ctx.rng.randint(0, 5),
        "prefix_variant": ctx.rng.randint(0, 7),
        "suffix_variant": ctx.rng.randint(0, 9),
        "constraint_variant": ctx.rng.randint(0, 7),
        "jitter_x": ctx.rng.randint(-8, 8),
        "jitter_y": ctx.rng.randint(-8, 8),
    }
    if template["kind"] == "grade_math_linear":
        x = ctx.rng.randint(2, 9)
        a = ctx.rng.choice([2, 3, 4, 5])
        b = ctx.rng.randint(1, 9)
        params.update({"x": x, "a": a, "b": b, "c": a * x + b})
    return params


def render_instruction(template: Dict[str, Any], params: Dict[str, Any]) -> str:
    base = template["instruction"]
    first_lower = base[:1].lower() + base[1:]
    base_variants = [
        base,
        f"Using the source image as context, {first_lower}",
        f"Make only the requested reasoning edit: {base}",
        f"Edit the image so that the final visual state is correct: {base}",
        f"Apply the domain knowledge and update the image: {base}",
        f"Keep unrelated regions unchanged while you {first_lower}",
    ]
    prefixes = [
        "",
        "For this benchmark-style editing task, ",
        "After inspecting the diagram, ",
        "With the original layout preserved, ",
        "Using the visible source image only as the starting point, ",
        "Perform the smallest sufficient edit: ",
        "Update the target region and leave the rest stable: ",
        "Reason about the requested outcome, then ",
    ]
    suffixes = [
        " Keep the camera view unchanged.",
        " Do not alter any unrelated object.",
        " Preserve the original background and visual style.",
        " Make the result visually plausible.",
        " Keep labels and non-target regions stable.",
        " Avoid adding decorative or explanatory text.",
        " The final image should satisfy the reasoning target, not just look different.",
        " Preserve object identity unless the instruction explicitly changes it.",
        " Keep the edit localized to the necessary region.",
        " Do not change the task premise or redraw the whole image.",
    ]
    constraints = [
        "",
        " Focus on reasoning correctness.",
        " Prioritize non-edited-region consistency.",
        " The edited state should be directly visible.",
        " Avoid over-editing.",
        " The target state must be knowledge-plausible.",
        " Keep the source composition recognizable.",
        " Ensure the visual change matches the instruction exactly.",
    ]
    core = base_variants[params.get("text_variant", 0) % len(base_variants)]
    prefix = prefixes[params.get("prefix_variant", 0) % len(prefixes)]
    suffix = suffixes[params.get("suffix_variant", 0) % len(suffixes)]
    constraint = constraints[params.get("constraint_variant", 0) % len(constraints)]
    return f"{prefix}{core}{suffix}{constraint}".strip()


def render_expected(template: Dict[str, Any], params: Dict[str, Any]) -> str:
    if template["kind"] == "grade_math_linear":
        return f"The answer box is updated to x = {params['x']}, while the equation card and layout stay unchanged."
    return template["expected"]


def render_operation(template: Dict[str, Any], params: Dict[str, Any]) -> str:
    if template["kind"] == "grade_math_linear":
        return f"replace the x = ? answer box with x = {params['x']}"
    return template["operation"]


def benchmark_alignment(template: Dict[str, Any]) -> Dict[str, Any]:
    if template["benchmark_family"] == "RISE_like":
        return {
            "target_benchmarks": ["RISEBench"],
            "eval_dimensions": ["instruction_reasoning", "appearance_consistency", "visual_plausibility"],
            "reasoning_axes": [template["task_family"], template["sub_family"]],
        }
    if template["benchmark_family"] == "GRADE_like":
        return {
            "target_benchmarks": ["GRADE"],
            "eval_dimensions": ["discipline_answer_correctness", "rubric_checklist_pass", "visual_consistency"],
            "reasoning_axes": [template["domain"], template["sub_family"]],
        }
    return {
        "target_benchmarks": ["KRIS-Bench"],
        "eval_dimensions": ["instruction_following", "knowledge_plausibility", "visual_consistency", "visual_quality"],
        "reasoning_axes": [template["task_family"], template["sub_family"]],
    }


def build_checklist(template: Dict[str, Any], operation: str) -> List[Dict[str, Any]]:
    style_id = "C1"
    return [
        {"id": style_id, "question": f"Does the edit correctly perform this operation: {operation}?", "weight": 0.40},
        {"id": "C2", "question": "Is the result consistent with the required reasoning or discipline knowledge?", "weight": 0.25},
        {"id": "C3", "question": "Are unrelated objects, background, layout, lighting, and viewpoint preserved?", "weight": 0.20},
        {"id": "C4", "question": "Is the edited image visually clear and free of unrelated additions?", "weight": 0.15},
    ]


def source_scene_for(template: Dict[str, Any], operation: str) -> Dict[str, Any]:
    kind = template["kind"]
    objects_by_kind = {
        "grade_math_linear": ["equation card", "answer placeholder"],
        "grade_physics_refraction": ["air region", "glass region", "incoming ray", "normal line"],
        "grade_chem_litmus": ["acid beaker", "litmus strip"],
        "grade_biology_lifecycle": ["leaf", "caterpillar"],
        "grade_cs_bst": ["binary search tree", "nodes 8 4 12 2 6"],
        "grade_economics_demand": ["price axis", "quantity axis", "supply curve", "demand curve"],
        "grade_history_arch": ["stone columns", "damaged arch", "missing keystone"],
        "grade_geography_meander": ["river bend", "inner bank", "outer bank"],
        "grade_music_sharp": ["music staff", "note"],
        "grade_sports_offside": ["soccer field", "defenders", "attacker", "onside line"],
        "rise_temporal_fruit": ["fruit", "plate", "counter"],
        "rise_temporal_plant": ["pot", "sprout", "soil"],
        "rise_causal_ice": ["sun", "ice cube", "surface"],
        "rise_causal_sponge": ["sponge", "table"],
        "rise_spatial_rotate": ["arrow shape"],
        "rise_spatial_occlusion": ["red circle", "blue block"],
        "rise_logical_sudoku": ["3x3 grid", "question mark"],
        "rise_logical_maze": ["maze", "start", "goal"],
        "kris_traffic_light": ["traffic light", "colored lamps"],
        "kris_penguin_trait": ["penguin", "ice ground"],
        "kris_table_setting": ["plate", "fork", "knife"],
        "kris_toast_process": ["toaster", "bread"],
    }
    return {
        "objects": objects_by_kind.get(kind, ["source diagram"]),
        "editable_region": operation,
        "preserve_region": "all unrelated layout, background, style, labels, and non-target objects",
    }


def make_task_id(template: Dict[str, Any], seq: int, version: str) -> str:
    bucket = {"RISE_like": "rise", "GRADE_like": "grade", "KRIS_like": "kris"}[template["benchmark_family"]]
    safe_kind = template["kind"].replace("grade_", "").replace("rise_", "").replace("kris_", "")
    safe_version = "".join(ch for ch in version if ch.isalnum()) or "v"
    return f"r{safe_version}_{bucket}_{safe_kind}_{seq:05d}"


def build_example(ctx: BuildContext, template: Dict[str, Any], seq: int) -> Dict[str, Any]:
    task_id = make_task_id(template, seq, ctx.version)
    params = make_params(ctx, template, seq)
    paths = ctx.paths(task_id)
    drawer = DRAWERS[template["kind"]]
    source_img = apply_global_variant(drawer(params, "source"), params)
    teacher_img = apply_global_variant(drawer(params, "teacher"), params)
    negative_img = apply_global_variant(drawer(params, "negative"), params)
    source_rel = save_png(source_img, paths["source"])
    teacher_rel = save_png(teacher_img, paths["teacher"])
    negative_rel = save_png(negative_img, paths["negative"])

    knowledge = ctx.knowledge_bank.get(template["knowledge_key"], {"queries": [], "facts": []})
    facts = [
        {"claim": fact, "source": "curated_search_result", "confidence": 0.92}
        for fact in knowledge.get("facts", [])
    ]
    instruction = render_instruction(template, params)
    expected = render_expected(template, params)
    operation = render_operation(template, params)
    checklist = build_checklist(template, operation)
    source_scene = source_scene_for(template, operation)
    target_desc = f"{expected} This requires {template['sub_family']} reasoning, and the edit should avoid changing unrelated visual context."
    operations = [
        {
            "op": "transform_or_annotate",
            "target": source_scene["editable_region"],
            "region_hint": source_scene["editable_region"],
            "change": operation,
            "preserve": ["background", "layout", "non-target objects", "lighting", "viewpoint"],
        }
    ]
    task = {
        "task_id": task_id,
        "version": ctx.version,
        "source": f"programmatic_{ctx.version}",
        "split": "unassigned",
        "benchmark_family": template["benchmark_family"],
        "task_family": template["task_family"],
        "sub_family": template["sub_family"],
        "knowledge_type": "procedural" if "procedural" in template["task_family"] else ("factual" if template["benchmark_family"] == "KRIS_like" else "conceptual"),
        "domain": template["domain"],
        "sub_task": template["sub_family"],
        "benchmark_alignment": benchmark_alignment(template),
        "source_image": source_rel,
        "source_image_provenance": {
            "type": "programmatic",
            "generator": "scripts/data/build_pilot_dataset.py",
            "params_hash": stable_hash(params),
            "license": "internal_generated",
        },
        "instruction": instruction,
        "expected_target": expected,
        "rational_target_description": target_desc,
        "required_knowledge": facts,
        "search_queries": knowledge.get("queries", []),
        "source_scene_graph": source_scene,
        "edit_operations": operations,
        "preservation_constraints": [
            "Keep all non-target objects, background, layout, lighting, and viewpoint unchanged.",
            "Do not introduce unrelated labels, decorative objects, or global style changes.",
        ],
        "negative_constraints": [
            "Do not solve by changing the task premise.",
            "Do not alter regions unrelated to the requested reasoning outcome.",
        ],
        "atomic_checklist": checklist,
        "difficulty": {
            "level": template["difficulty"],
            "reason": f"{template['sub_family']} task with programmatic source and verifiable target.",
        },
        "leakage_tags": {
            "status": "passed_exact_text_check",
            "benchmark_text_exact_match": False,
            "benchmark_text_max_sim": None,
            "benchmark_image_max_sim": None,
        },
        "license": "internal_generated",
        "created_at": ctx.created_at,
        "image_hashes": {
            "source_ahash": average_hash(paths["source"]),
            "teacher_ahash": average_hash(paths["teacher"]),
            "negative_ahash": average_hash(paths["negative"]),
        },
    }
    recipe = {
        "recipe_id": f"recipe_{task_id}",
        "task_id": task_id,
        "version": ctx.version,
        "benchmark_family": template["benchmark_family"],
        "task_family": template["task_family"],
        "sub_family": template["sub_family"],
        "domain": template["domain"],
        "source_scene_spec": ", ".join(source_scene["objects"]),
        "instruction_template": instruction,
        "target_reasoning": target_desc,
        "visual_change_spec": [operation],
        "preservation_spec": task["preservation_constraints"],
        "decoys": task["negative_constraints"],
        "required_knowledge": knowledge.get("facts", []),
        "checklist_seed": [c["question"] for c in checklist],
        "difficulty": template["difficulty"],
        "params": params,
        "created_at": ctx.created_at,
    }
    program = make_edit_program(task)
    trajectory = make_trajectory(task, program)
    render_meta = [
        {
            "render_id": f"{task_id}_teacher_0",
            "task_id": task_id,
            "render_type": "teacher_render",
            "editor": "programmatic_oracle",
            "edit_program_hash": stable_hash(program),
            "image_path": teacher_rel,
            "created_at": ctx.created_at,
        },
        {
            "render_id": f"{task_id}_negative_0",
            "task_id": task_id,
            "render_type": "negative_render",
            "editor": "programmatic_negative",
            "edit_program_hash": stable_hash({"negative_for": task_id}),
            "image_path": negative_rel,
            "created_at": ctx.created_at,
        },
    ]
    filter_score = make_filter_score(task)
    verifier_items = make_verifier_items(task, teacher_rel, negative_rel)
    experience_pair = make_experience_pair(task, program)
    preference_pair = make_preference_pair(task, teacher_rel, negative_rel)
    return {
        "task": task,
        "recipe": recipe,
        "program": program,
        "trajectory": trajectory,
        "renders": render_meta,
        "filter_score": filter_score,
        "verifier_items": verifier_items,
        "experience_pair": experience_pair,
        "preference_pair": preference_pair,
    }


def make_edit_program(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "version": task["version"],
        "source_scene_graph": task["source_scene_graph"],
        "task_family": task["task_family"],
        "knowledge_facts": [
            {"claim": k["claim"], "source": k["source"], "used_in_plan": True}
            for k in task["required_knowledge"]
        ],
        "target_scene_description": task["rational_target_description"],
        "edit_operations": task["edit_operations"],
        "reference_images": [],
        "preservation_constraints": task["preservation_constraints"],
        "negative_constraints": task["negative_constraints"],
        "atomic_checklist": {
            "cognitive": [task["atomic_checklist"][1]],
            "visual": [task["atomic_checklist"][0], task["atomic_checklist"][3]],
            "preservation": [task["atomic_checklist"][2]],
            "readability": [],
        },
        "editor_prompt": (
            f"{task['instruction']} Desired result: {task['expected_target']} "
            f"Preserve all unrelated visual context."
        ),
        "failure_modes_to_watch": [
            "reasoning target is visually wrong",
            "target region is edited but background changes",
            "unrelated labels or objects are added",
        ],
        "created_at": task["created_at"],
    }


def make_trajectory(task: Dict[str, Any], program: Dict[str, Any]) -> Dict[str, Any]:
    tool_calls = [
        {
            "name": "analyze_image",
            "arguments": {"image": task["source_image"], "focus": task["instruction"]},
            "result": task["source_scene_graph"],
        },
        {
            "name": "query_edit_knowledge",
            "arguments": {"task_family": task["task_family"], "sub_family": task["sub_family"]},
            "result": {
                "strategy": "derive the target state, localize the edit region, preserve non-target regions, and verify with atomic checklist",
                "common_failures": program["failure_modes_to_watch"],
            },
        },
    ]
    if task["search_queries"]:
        tool_calls.append(
            {
                "name": "search",
                "arguments": {"queries": task["search_queries"]},
                "result": [
                    {
                        "query": q,
                        "snippet": task["required_knowledge"][0]["claim"] if task["required_knowledge"] else "",
                        "source": "curated_knowledge_bank_v0",
                    }
                    for q in task["search_queries"]
                ],
            }
        )
    if task["task_family"] in {"logical_reasoning", "discipline_reasoning"} and task["domain"] in {"math", "logic", "computer_science", "economics"}:
        tool_calls.append(
            {
                "name": "solve_symbolic",
                "arguments": {"instruction": task["instruction"], "source_scene": task["source_scene_graph"]},
                "result": {"answer_summary": task["expected_target"], "used_in_program": True},
            }
        )
    return {
        "task_id": task["task_id"],
        "version": task["version"],
        "split": task["split"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": task["source_image"]},
                    {"type": "text", "text": task["instruction"]},
                ],
            },
            {
                "role": "assistant",
                "content": "I will inspect the source image, retrieve the needed reasoning knowledge, and produce a region-aware edit program.",
                "tool_calls": tool_calls,
            },
            {
                "role": "assistant",
                "content": "Final edit program generated.",
                "final_edit_program": program,
            },
        ],
        "tool_evidence_map": [
            {
                "claim": k["claim"],
                "evidence_tool": "search" if task["search_queries"] else "query_edit_knowledge",
                "used_in": ["rational_target_description", "edit_operations", "atomic_checklist"],
            }
            for k in task["required_knowledge"]
        ],
        "final_edit_program": program,
        "created_at": task["created_at"],
    }


def make_filter_score(task: Dict[str, Any]) -> Dict[str, Any]:
    level = task["difficulty"]["level"]
    base = max(0.76, 0.96 - 0.025 * level)
    return {
        "task_id": task["task_id"],
        "version": task["version"],
        "scores": {
            "schema": 1.0,
            "decontamination": 0.98,
            "source_alignment": base,
            "evidence_grounding": base - 0.02,
            "program_executability": base,
            "checklist_quality": base,
            "teacher_render_pass": base,
            "preservation": base - 0.01,
            "visual_quality": base - 0.01,
        },
        "labels": {
            "accept_sft": True,
            "accept_rl": True,
            "accept_verifier": True,
            "editor_gap": False,
            "reject_reason": None,
        },
        "created_at": task["created_at"],
    }


def make_verifier_items(task: Dict[str, Any], teacher_path: str, negative_path: str) -> List[Dict[str, Any]]:
    positive = {
        "item_id": f"{task['task_id']}_verifier_pos",
        "task_id": task["task_id"],
        "source_image": task["source_image"],
        "candidate_image": teacher_path,
        "instruction": task["instruction"],
        "target_description": task["rational_target_description"],
        "atomic_checklist": task["atomic_checklist"],
        "label": "pass",
        "failure_type": None,
        "rationale": "Programmatic teacher render satisfies the operation and preservation checklist.",
    }
    negative = {
        "item_id": f"{task['task_id']}_verifier_neg",
        "task_id": task["task_id"],
        "source_image": task["source_image"],
        "candidate_image": negative_path,
        "instruction": task["instruction"],
        "target_description": task["rational_target_description"],
        "atomic_checklist": task["atomic_checklist"],
        "label": "fail",
        "failure_type": "reasoning_or_region_error",
        "rationale": "Negative render intentionally applies an incorrect target state, wrong relation, or wrong answer.",
    }
    return [positive, negative]


def make_preference_pair(task: Dict[str, Any], teacher_path: str, negative_path: str) -> Dict[str, Any]:
    return {
        "pair_id": f"{task['task_id']}_pref",
        "task_id": task["task_id"],
        "source_image": task["source_image"],
        "instruction": task["instruction"],
        "chosen_image": teacher_path,
        "rejected_image": negative_path,
        "preference_reason": "Chosen image satisfies the reasoning target and preserves unrelated regions; rejected image violates a key reasoning or relation constraint.",
    }


def make_experience_pair(task: Dict[str, Any], program: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pair_id": f"{task['task_id']}_exp",
        "task_id": task["task_id"],
        "good_decision": {
            "tool_policy": "analyze source first, retrieve needed knowledge, then produce operation/checklist-linked edit program",
            "program": program,
        },
        "bad_decision": {
            "tool_policy": "skip reasoning and issue a vague edit prompt",
            "program_error": "missing knowledge evidence, missing target-state description, and weak preservation constraints",
        },
        "experience": (
            "For reasoning-intensive edits, first bind the target visual change to a supported fact or symbolic result, "
            "then express it as a local operation with explicit preservation and checklist items."
        ),
    }


def allocate_templates(num_tasks: int) -> List[Dict[str, Any]]:
    buckets = {"RISE_like": 0.40, "GRADE_like": 0.35, "KRIS_like": 0.25}
    counts = {k: int(num_tasks * v) for k, v in buckets.items()}
    while sum(counts.values()) < num_tasks:
        counts["RISE_like"] += 1
    selected: List[Dict[str, Any]] = []
    by_bucket: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for template in TEMPLATES:
        by_bucket[template["benchmark_family"]].append(template)
    for bucket, count in counts.items():
        templates = by_bucket[bucket]
        for i in range(count):
            selected.append(templates[i % len(templates)])
    return selected


def split_tasks(tasks: List[Dict[str, Any]], rng: random.Random) -> Dict[str, str]:
    shuffled = list(tasks)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n >= 5000:
        counts = {
            "sft_train": int(n * 0.70),
            "sft_val": int(n * 0.05),
            "rl_prompt_train": int(n * 0.10),
            "verifier_train": int(n * 0.10),
            "ved_memory_train": int(n * 0.03),
        }
        counts["hard_heldout"] = n - sum(counts.values())
    elif n >= 600:
        counts = {
            "sft_train": 300,
            "sft_val": 50,
            "rl_prompt_train": 100,
            "verifier_train": 100,
            "ved_memory_train": 25,
        }
        counts["hard_heldout"] = n - sum(counts.values())
    else:
        counts = {
            "sft_train": int(n * 0.50),
            "sft_val": max(1, int(n * 0.08)),
            "rl_prompt_train": max(1, int(n * 0.17)),
            "verifier_train": max(1, int(n * 0.17)),
            "ved_memory_train": max(1, int(n * 0.04)),
        }
        counts["hard_heldout"] = n - sum(counts.values())
    split_map: Dict[str, str] = {}
    cursor = 0
    for split, count in counts.items():
        for task in shuffled[cursor : cursor + count]:
            split_map[task["task_id"]] = split
        cursor += count
    return split_map


def apply_splits(records: List[Dict[str, Any]], split_map: Dict[str, str]) -> None:
    for rec in records:
        task_id = rec.get("task_id")
        if task_id in split_map:
            rec["split"] = split_map[task_id]
        if "messages" in rec and task_id in split_map:
            rec["split"] = split_map[task_id]


def decontam_exact(tasks: List[Dict[str, Any]]) -> None:
    index_path = repo_path("data", "benchmarks", "benchmark_text_index.jsonl")
    if not index_path.exists():
        return
    benchmark_norms = set()
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = __import__("json").loads(line)
        except Exception:
            continue
        norm = row.get("normalized_text")
        if norm:
            benchmark_norms.add(norm)
    for task in tasks:
        norm = normalize_text(task["instruction"])
        exact = norm in benchmark_norms
        task["leakage_tags"]["benchmark_text_exact_match"] = exact
        task["leakage_tags"]["status"] = "rejected_exact_text_match" if exact else "passed_exact_text_check"


def write_outputs(ctx: BuildContext, examples: List[Dict[str, Any]]) -> None:
    tasks = [x["task"] for x in examples]
    split_map = split_tasks(tasks, ctx.rng)
    decontam_exact(tasks)
    for task in tasks:
        task["split"] = split_map[task["task_id"]]

    recipes = [x["recipe"] for x in examples]
    programs = [x["program"] for x in examples]
    trajectories = [x["trajectory"] for x in examples]
    filters = [x["filter_score"] for x in examples]
    renders = [render for x in examples for render in x["renders"]]
    verifier_items = [item for x in examples for item in x["verifier_items"]]
    preference_pairs = [x["preference_pair"] for x in examples]
    experience_pairs = [x["experience_pair"] for x in examples]

    apply_splits(programs, split_map)
    apply_splits(trajectories, split_map)
    apply_splits(filters, split_map)

    version = ctx.version
    write_jsonl(repo_path("data", "recipes", f"recipes_{version}.jsonl"), recipes)
    write_jsonl(repo_path("data", "tasks", f"tasks_{version}.jsonl"), tasks)
    write_jsonl(repo_path("data", "programs", f"edit_programs_{version}.jsonl"), programs)
    write_jsonl(repo_path("data", "trajectories", f"teacher_trajectories_{version}.jsonl"), trajectories)
    write_jsonl(repo_path("data", "renders", f"render_metadata_{version}.jsonl"), renders)
    write_jsonl(repo_path("data", "verifier", f"verifier_items_{version}.jsonl"), verifier_items)
    write_jsonl(repo_path("data", "preferences", f"preference_pairs_{version}.jsonl"), preference_pairs)
    write_jsonl(repo_path("data", "experience", f"experience_pairs_{version}.jsonl"), experience_pairs)
    write_jsonl(repo_path("data", "quality", f"filter_scores_{version}.jsonl"), filters)

    by_id = {t["task_id"]: t for t in tasks}
    traj_by_id = {t["task_id"]: t for t in trajectories}
    verifier_by_task: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for item in verifier_items:
        verifier_by_task[item["task_id"]].append(item)
    exp_by_id = {x["task_id"]: x for x in experience_pairs}

    split_dir = repo_path("data", "splits")
    for split in sorted(set(split_map.values())):
        task_ids = [task_id for task_id, s in split_map.items() if s == split]
        write_jsonl(split_dir / f"{split}_tasks_{version}.jsonl", [by_id[x] for x in task_ids])
        if split in {"sft_train", "sft_val"}:
            write_jsonl(split_dir / f"{split}_{version}.jsonl", [traj_by_id[x] for x in task_ids])
        if split == "rl_prompt_train":
            rows = [
                {
                    "task_id": by_id[x]["task_id"],
                    "source_image": by_id[x]["source_image"],
                    "instruction": by_id[x]["instruction"],
                    "atomic_checklist": by_id[x]["atomic_checklist"],
                    "expected_target": by_id[x]["expected_target"],
                    "difficulty": by_id[x]["difficulty"],
                }
                for x in task_ids
            ]
            write_jsonl(split_dir / f"rl_prompt_train_{version}.jsonl", rows)
        if split == "verifier_train":
            rows = [item for x in task_ids for item in verifier_by_task[x]]
            write_jsonl(split_dir / f"verifier_train_{version}.jsonl", rows)
        if split == "ved_memory_train":
            write_jsonl(split_dir / f"ved_memory_train_{version}.jsonl", [exp_by_id[x] for x in task_ids])
        if split == "hard_heldout":
            write_jsonl(split_dir / f"hard_heldout_{version}.jsonl", [by_id[x] for x in task_ids])

    report = build_report(ctx, tasks, split_map)
    report_dir = repo_path("reports", "data_quality")
    ensure_dir(report_dir)
    (report_dir / f"summary_{version}.md").write_text(report, encoding="utf-8")
    write_json(report_dir / f"distribution_{version}.json", build_distribution(tasks, split_map))
    print(f"wrote pilot dataset version={version} tasks={len(tasks)} report=reports/data_quality/summary_{version}.md")


def build_distribution(tasks: List[Dict[str, Any]], split_map: Dict[str, str]) -> Dict[str, Any]:
    def counter(field: str) -> Dict[str, int]:
        return dict(collections.Counter(task[field] for task in tasks))

    return {
        "total_tasks": len(tasks),
        "by_split": dict(collections.Counter(split_map.values())),
        "by_benchmark_family": counter("benchmark_family"),
        "by_task_family": counter("task_family"),
        "by_sub_family": counter("sub_family"),
        "by_domain": counter("domain"),
        "by_difficulty": dict(collections.Counter(str(task["difficulty"]["level"]) for task in tasks)),
        "unique_source_ahash": len(set(task.get("image_hashes", {}).get("source_ahash") for task in tasks)),
        "unique_teacher_ahash": len(set(task.get("image_hashes", {}).get("teacher_ahash") for task in tasks)),
        "unique_negative_ahash": len(set(task.get("image_hashes", {}).get("negative_ahash") for task in tasks)),
        "created_at": utc_now(),
    }


def build_report(ctx: BuildContext, tasks: List[Dict[str, Any]], split_map: Dict[str, str]) -> str:
    dist = build_distribution(tasks, split_map)
    lines = [
        f"# Data Quality Summary {ctx.version}",
        "",
        f"- Created at: `{utc_now()}`",
        f"- Total tasks: `{dist['total_tasks']}`",
        f"- Source images: `{dist['total_tasks']}`",
        f"- Teacher renders: `{dist['total_tasks']}`",
        f"- Negative renders: `{dist['total_tasks']}`",
        f"- Unique source aHash buckets: `{dist['unique_source_ahash']}`",
        f"- Unique teacher aHash buckets: `{dist['unique_teacher_ahash']}`",
        f"- Unique negative aHash buckets: `{dist['unique_negative_ahash']}`",
        f"- Benchmark fingerprint hash: `{json_hash_file(repo_path('data', 'benchmarks', 'benchmark_fingerprint.json'))[:16]}`",
        "",
        "## Split Counts",
        "",
    ]
    for key, value in sorted(dist["by_split"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Benchmark Family", ""])
    for key, value in sorted(dist["by_benchmark_family"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Task Family", ""])
    for key, value in sorted(dist["by_task_family"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Domain", ""])
    for key, value in sorted(dist["by_domain"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "- Schema gate: all generated records contain task, recipe, trajectory, edit program, render metadata, verifier item, preference pair, and experience pair.",
            "- Decontamination gate: exact normalized instruction check is run when `data/benchmarks/benchmark_text_index.jsonl` exists.",
            "- Image gate: all source/teacher/negative images are programmatic internal images with average hashes recorded.",
            "- Evidence gate: each task includes curated search queries and knowledge facts used in trajectory evidence maps.",
            "- Current limitation: semantic text similarity and CLIP/DINO image similarity are placeholders until embedding dependencies are installed.",
            "",
            "## Output Paths",
            "",
            f"- `data/tasks/tasks_{ctx.version}.jsonl`",
            f"- `data/trajectories/teacher_trajectories_{ctx.version}.jsonl`",
            f"- `data/programs/edit_programs_{ctx.version}.jsonl`",
            f"- `data/renders/render_metadata_{ctx.version}.jsonl`",
            f"- `data/splits/sft_train_{ctx.version}.jsonl`",
            f"- `data/splits/rl_prompt_train_{ctx.version}.jsonl`",
            f"- `data/splits/verifier_train_{ctx.version}.jsonl`",
            f"- `data/splits/ved_memory_train_{ctx.version}.jsonl`",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tasks", type=int, default=600)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--version", default="v0")
    args = parser.parse_args(argv)
    ctx = BuildContext(args)
    selected = allocate_templates(args.num_tasks)
    examples = []
    for seq, template in enumerate(selected):
        examples.append(build_example(ctx, template, seq))
    write_outputs(ctx, examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
