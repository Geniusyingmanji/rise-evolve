from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_hash(obj: Any, n: int = 16) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_font(size: int = 24, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if bold:
        candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple:
    if hasattr(draw, "textbbox"):
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    return draw.textsize(text, font=font)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple,
    text: str,
    font: ImageFont.ImageFont,
    fill: str = "#111827",
) -> None:
    x, y = xy
    w, h = text_size(draw, text, font)
    draw.text((x - w / 2, y - h / 2), text, font=font, fill=fill)


def wrap_text(text: str, width: int = 42) -> str:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    length = 0
    for word in words:
        extra = 1 if current else 0
        if current and length + len(word) + extra > width:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + extra
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def new_canvas(title: str = "", subtitle: str = "", size: int = 512, show_header: bool = False) -> tuple:
    img = Image.new("RGB", (size, size), "#f8fafc")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, size - 1, size - 1), outline="#cbd5e1", width=2)
    if show_header and title:
        draw.text((24, 18), title, font=load_font(22, bold=True), fill="#111827")
    if show_header and subtitle:
        draw.text((24, 48), wrap_text(subtitle, 52), font=load_font(14), fill="#475569")
    return img, draw


def average_hash(path: Path, hash_size: int = 8) -> str:
    img = Image.open(path).convert("L").resize((hash_size, hash_size))
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= avg else "0" for p in pixels)
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def save_png(img: Image.Image, path: Path) -> str:
    ensure_dir(path.parent)
    img.save(path)
    return str(path.relative_to(ROOT))


PALETTE = {
    "ink": "#111827",
    "muted": "#64748b",
    "border": "#cbd5e1",
    "red": "#dc2626",
    "orange": "#f97316",
    "yellow": "#facc15",
    "green": "#16a34a",
    "blue": "#2563eb",
    "sky": "#38bdf8",
    "purple": "#7c3aed",
    "brown": "#92400e",
    "sand": "#fde68a",
    "water": "#93c5fd",
    "light": "#f8fafc",
    "dark": "#1f2937",
}


def json_hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()
