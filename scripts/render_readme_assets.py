#!/usr/bin/env python3
"""Render README graphics with layout checks.

This script exists because README graphics are easy to make visually wrong with
ad-hoc drawing code: a card can look fine in code while its text spills outside
the box once GitHub scales it. Every text block below is fitted to a measured
pixel rectangle; --check fails if anything would overflow.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "graphics"

COL = {
    "bg": (5, 10, 20),
    "panel": (15, 25, 42),
    "panel2": (18, 31, 52),
    "line": (45, 64, 94),
    "text": (237, 245, 255),
    "muted": (155, 174, 203),
    "cyan": (49, 208, 198),
    "green": (86, 211, 100),
    "purple": (177, 140, 255),
    "coral": (255, 123, 114),
    "amber": (242, 204, 96),
    "blue": (83, 166, 255),
}

REGULAR = [
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
BOLD = [
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]
MONO = [
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
]


def load_font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    candidates = MONO if mono else (BOLD if bold else REGULAR)
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()  # type: ignore[return-value]


@dataclass(frozen=True)
class TextBlock:
    name: str
    box: tuple[int, int, int, int]
    text: str
    size: int
    min_size: int = 14
    bold: bool = False
    mono: bool = False
    fill: tuple[int, int, int] = COL["text"]
    anchor: str | None = None


class FitError(RuntimeError):
    pass


class Canvas:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.image = self.gradient_bg(width, height)
        self.draw = ImageDraw.Draw(self.image)
        self.failures: list[str] = []
        self.add_grid()

    @staticmethod
    def gradient_bg(width: int, height: int) -> Image.Image:
        img = Image.new("RGB", (width, height), COL["bg"])
        px = img.load()
        if px is None:
            raise RuntimeError("Pillow did not return a pixel access object")
        for y in range(height):
            for x in range(width):
                g1 = max(0.0, 1.0 - (((x - width * 0.15) / width) ** 2 * 5 + ((y - height * 0.05) / height) ** 2 * 7))
                g2 = max(0.0, 1.0 - (((x - width * 0.90) / width) ** 2 * 5 + ((y - height * 0.20) / height) ** 2 * 6))
                px[x, y] = (
                    int(5 + 22 * g1 + 18 * g2),
                    int(10 + 35 * g1 + 8 * g2),
                    int(20 + 60 * g1 + 38 * g2),
                )
        return img.convert("RGBA")

    def add_grid(self) -> None:
        for x in range(0, self.width, 48):
            self.draw.line([(x, 0), (x, self.height)], fill=(18, 30, 52), width=1)
        for y in range(0, self.height, 48):
            self.draw.line([(0, y), (self.width, y)], fill=(18, 30, 52), width=1)

    def shadow_box(self, box: tuple[int, int, int, int], *, radius: int, fill: tuple[int, int, int], outline: tuple[int, int, int], width: int = 2) -> None:
        x1, y1, x2, y2 = box
        shadow = Image.new("RGBA", self.image.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.rounded_rectangle((x1 + 10, y1 + 16, x2 + 10, y2 + 16), radius=radius, fill=(0, 0, 0, 95))
        shadow = shadow.filter(ImageFilter.GaussianBlur(18))
        self.image.alpha_composite(shadow)
        self.draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    def arrow(self, start: tuple[int, int], end: tuple[int, int], *, fill: tuple[int, int, int], width: int = 4, dash: bool = False) -> None:
        x1, y1 = start
        x2, y2 = end
        if dash:
            for i in range(0, 24, 2):
                a = i / 24
                b = (i + 1) / 24
                self.draw.line(
                    [(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a), (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b)],
                    fill=fill,
                    width=width,
                )
        else:
            self.draw.line([start, end], fill=fill, width=width)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 15
        self.draw.polygon(
            [
                (x2, y2),
                (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6)),
                (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6)),
            ],
            fill=fill,
        )

    def pill(self, x: int, y: int, label: str, color: tuple[int, int, int]) -> int:
        font = load_font(19)
        pad = 14
        bb = self.draw.textbbox((0, 0), label, font=font)
        width = int(bb[2] - bb[0] + pad * 2)
        self.draw.rounded_rectangle((x, y, x + width, y + 34), radius=17, fill=(22, 38, 64), outline=color, width=1)
        self.draw.text((x + pad, y + 6), label, font=font, fill=COL["text"])
        return width

    def _wrap_lines(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        lines: list[str] = []
        for raw_line in text.split("\n"):
            words = raw_line.split()
            if not words:
                lines.append("")
                continue
            current = words[0]
            for word in words[1:]:
                candidate = f"{current} {word}"
                if self.draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
        return lines

    def _measure_lines(self, lines: Iterable[str], font: ImageFont.FreeTypeFont, spacing: int) -> tuple[int, int]:
        lines = list(lines)
        if not lines:
            return 0, 0
        widths = [int(self.draw.textbbox((0, 0), line, font=font)[2]) for line in lines]
        line_height = int(self.draw.textbbox((0, 0), "Ag", font=font)[3] - self.draw.textbbox((0, 0), "Ag", font=font)[1])
        return max(widths), line_height * len(lines) + spacing * (len(lines) - 1)

    def fit_text(self, block: TextBlock, *, spacing: int = 6) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
        x1, y1, x2, y2 = block.box
        max_width = x2 - x1
        max_height = y2 - y1
        for size in range(block.size, block.min_size - 1, -1):
            font = load_font(size, bold=block.bold, mono=block.mono)
            lines = self._wrap_lines(block.text, font, max_width)
            width, height = self._measure_lines(lines, font, spacing)
            if width <= max_width and height <= max_height:
                return font, lines, spacing
        message = f"{block.name} text does not fit {block.box}: {block.text!r}"
        self.failures.append(message)
        raise FitError(message)

    def draw_fit_text(self, block: TextBlock, *, spacing: int = 6) -> None:
        font, lines, spacing = self.fit_text(block, spacing=spacing)
        x1, y1, x2, y2 = block.box
        text = "\n".join(lines)
        if block.anchor == "center":
            width, height = self._measure_lines(lines, font, spacing)
            x = x1 + (x2 - x1 - width) // 2
            y = y1 + (y2 - y1 - height) // 2
        elif block.anchor == "top-center":
            width, _ = self._measure_lines(lines, font, spacing)
            x = x1 + (x2 - x1 - width) // 2
            y = y1
        else:
            x, y = x1, y1
        self.draw.multiline_text((x, y), text, font=font, fill=block.fill, spacing=spacing)

    def card(self, *, name: str, box: tuple[int, int, int, int], title: str, body: str, outline: tuple[int, int, int], footer: str | None = None, title_size: int = 30, body_size: int = 23) -> None:
        self.shadow_box(box, radius=28, fill=COL["panel"], outline=outline, width=3)
        x1, y1, x2, y2 = box
        pad = 28
        self.draw_fit_text(
            TextBlock(f"{name}.title", (x1 + pad, y1 + 24, x2 - pad, y1 + 68), title, title_size, min_size=18, bold=True),
            spacing=2,
        )
        footer_height = 0
        if footer:
            footer_height = 32
            self.draw_fit_text(
                TextBlock(f"{name}.footer", (x1 + pad, y2 - 42, x2 - pad, y2 - 12), footer, 16, min_size=12, mono=True, fill=COL["muted"]),
                spacing=2,
            )
        self.draw_fit_text(
            TextBlock(f"{name}.body", (x1 + pad, y1 + 82, x2 - pad, y2 - 24 - footer_height), body, body_size, min_size=15, fill=(212, 226, 246)),
            spacing=7,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.convert("RGB").save(path, quality=95)


def render_hero(write: bool) -> Canvas:
    c = Canvas(1600, 840)
    c.shadow_box((70, 70, 1530, 770), radius=42, fill=(11, 20, 34), outline=COL["line"], width=2)
    c.draw_fit_text(TextBlock("hero.title", (135, 135, 900, 220), "DailyEdge", 88, min_size=42, bold=True))
    c.draw_fit_text(TextBlock("hero.subtitle", (138, 238, 1210, 290), "Alert-only evidence for stock trade setups.", 42, min_size=24, bold=True, fill=COL["cyan"]))
    c.draw_fit_text(
        TextBlock(
            "hero.body",
            (138, 318, 980, 475),
            "Scans a versioned universe, arms breakout tripwires, re-stamps risk at the signal bar, exposes read-only API context for agents, and records evidence so strategies can be promoted or killed by data.",
            30,
            min_size=21,
        ),
        spacing=10,
    )
    points = []
    x0, y0 = 1120, 400
    for i in range(145):
        x = x0 + i * 2
        y = y0 - math.sin(i / 15) * 32 - i * 0.28 + math.sin(i / 5) * 8
        points.append((x, y))
    c.draw.line([(x, y + 8) for x, y in points], fill=(49, 208, 198, 60), width=12)
    c.draw.line(points, fill=COL["cyan"], width=5)
    kpis = [("25", "AI/semi names"), ("3R", "current target"), ("0", "broker endpoints"), ("55", "tests passing")]
    start_x = 135
    for value, label in kpis:
        box = (start_x, 520, start_x + 300, 650)
        c.shadow_box(box, radius=24, fill=COL["panel2"], outline=COL["line"], width=2)
        c.draw_fit_text(TextBlock(f"hero.kpi.{value}", (start_x + 28, 540, start_x + 272, 600), value, 58, min_size=30, bold=True, fill=COL["green"] if value != "0" else COL["coral"]))
        c.draw_fit_text(TextBlock(f"hero.kpi.{label}", (start_x + 28, 608, start_x + 272, 640), label, 23, min_size=16, fill=COL["muted"]))
        start_x += 335
    c.draw_fit_text(
        TextBlock("hero.footer", (180, 710, 1420, 745), "No auto-buy · read-only API · human executes · analytical opinion only", 20, min_size=14, mono=True, fill=COL["amber"], anchor="center"),
        spacing=2,
    )
    if write:
        c.save(OUT / "dailyedge-hero.png")
    return c


def render_architecture(write: bool) -> Canvas:
    c = Canvas(1600, 1180)
    c.draw_fit_text(TextBlock("arch.title", (80, 62, 1080, 130), "DailyEdge v2 architecture", 58, min_size=32, bold=True))
    c.draw_fit_text(TextBlock("arch.subtitle", (84, 136, 1350, 170), "Scanner → evidence → read-only API → operator frontend → human decision", 25, min_size=18, fill=COL["muted"]))
    x = 82
    for label, color in [("alert-only", COL["green"]), ("no broker endpoint", COL["coral"]), ("versioned universe", COL["cyan"]), ("human executes", COL["amber"])] :
        x += c.pill(x, 195, label, color) + 12

    boxes = {
        "data": (80, 325, 410, 505),
        "engine": (500, 285, 900, 535),
        "api": (1000, 285, 1435, 535),
        "config": (80, 650, 410, 850),
        "evidence": (500, 620, 900, 870),
        "ui": (1000, 620, 1435, 870),
        "human": (500, 930, 1435, 1115),
    }
    # Arrows drawn before cards so no arrow ever crosses text.
    c.arrow((410, 415), (500, 415), fill=COL["blue"])
    c.arrow((900, 415), (1000, 415), fill=COL["cyan"])
    c.arrow((700, 535), (700, 620), fill=COL["purple"])
    c.arrow((410, 750), (500, 750), fill=COL["amber"])
    c.arrow((1218, 535), (1218, 620), fill=COL["green"])
    c.arrow((1218, 870), (1218, 930), fill=COL["coral"])
    c.arrow((900, 745), (1000, 745), fill=COL["purple"], dash=True)
    c.arrow((700, 870), (700, 930), fill=COL["purple"], dash=True)

    c.card(name="arch.data", box=boxes["data"], title="Market data", body="Price history, volume, ATR, and SPY regime context.", outline=COL["blue"])
    c.card(name="arch.engine", box=boxes["engine"], title="Research engine", body="Scores the 25-name AI/semi universe and arms approved tripwires.", outline=COL["cyan"], footer="scripts/universe_scanner.py")
    c.card(name="arch.api", box=boxes["api"], title="FastAPI API", body="Read-only endpoints expose state, scanner output, evidence, glossary, and agent context.", outline=COL["green"], footer="GET /api/agent/context")
    c.card(name="arch.config", box=boxes["config"], title="Versioned config", body="Universe, primary setup, RR=3, and watcher projections live in git.", outline=COL["amber"])
    c.card(name="arch.evidence", box=boxes["evidence"], title="Evidence lab", body="Records ticker list, per-name setup results, baseline margin tests, and regime counts.", outline=COL["purple"], footer="scripts/evidence_cache.py --v2")
    c.card(name="arch.ui", box=boxes["ui"], title="Operator app", body="Frontend explains the stack, glossary, armed watchers, and no-autobuy contract.", outline=COL["cyan"], footer="frontend/index.html")
    c.card(name="arch.human", box=boxes["human"], title="Manual execution only", body="Analytical plans and alerts only. You place, skip, size, and journal trades manually.", outline=COL["coral"], footer="NO AUTO-BUY · NO ORDERS API")
    c.draw_fit_text(TextBlock("arch.generated", (80, 1130, 1050, 1155), "Generated docs asset · docs/assets/graphics/dailyedge-architecture.png", 16, min_size=12, mono=True, fill=COL["muted"]))
    if write:
        c.save(OUT / "dailyedge-architecture.png")
    return c


def render_rule_loop(write: bool) -> Canvas:
    c = Canvas(1600, 1200)
    c.draw_fit_text(TextBlock("loop.title", (80, 58, 1350, 128), "How a DailyEdge signal earns trust", 58, min_size=32, bold=True))
    c.draw_fit_text(TextBlock("loop.subtitle", (84, 136, 1400, 172), "A gated alert workflow: hard evidence, re-stamped risk, no autonomous execution.", 25, min_size=18, fill=COL["muted"]))

    rule_box = (440, 220, 1160, 390)
    c.shadow_box(rule_box, radius=30, fill=(10, 18, 31), outline=COL["green"], width=3)
    c.draw_fit_text(TextBlock("loop.rule.label", (500, 242, 1100, 280), "Current forward rule", 30, min_size=20, bold=True, anchor="center"))
    c.draw_fit_text(TextBlock("loop.rule.name", (500, 290, 1100, 340), "breakout_20_rr3", 44, min_size=26, mono=True, fill=COL["green"], anchor="center"))
    c.draw_fit_text(TextBlock("loop.rule.shape", (500, 348, 1100, 375), "1.5×ATR stop · 3R target · alert-only", 23, min_size=16, fill=COL["text"], anchor="center"), spacing=2)

    steps = [
        ("1", "Scan universe", "25-name AI/semi list is a model parameter, not an accident."),
        ("2", "Score setup", "breakout_20_rr3 ranks candidates; ema_cross remains disabled."),
        ("3", "Arm watcher", "Store trigger now; stop/target are only projections."),
        ("4", "Fire", "Cross trigger; use signal-bar ATR and next open."),
        ("5", "Re-stamp", "Fresh stop = 1.5×ATR; target = 3R from entry."),
        ("6", "Referee", "Regime, costs, concentration, kill rules, and margin tests."),
        ("7", "Human decision", "Manual trade/skip. No API route can buy."),
        ("8", "Ledger evidence", "Forward results decide promote, probation, or retire."),
    ]
    positions = [(80, 470), (455, 470), (830, 470), (1205, 470), (1205, 725), (830, 725), (455, 725), (80, 725)]
    card_w, card_h = 315, 170
    centers = [(x + card_w // 2, y + card_h // 2) for x, y in positions]
    for idx in range(len(centers) - 1):
        c.arrow(centers[idx], centers[idx + 1], fill=COL["cyan"] if idx < 4 else COL["purple"], width=3)
    c.arrow(centers[-1], (440, 305), fill=COL["purple"], width=3, dash=True)

    for (num, title, body), (x, y) in zip(steps, positions):
        box = (x, y, x + card_w, y + card_h)
        c.shadow_box(box, radius=24, fill=COL["panel"], outline=COL["line"], width=2)
        c.draw.ellipse((x + 20, y + 22, x + 62, y + 64), fill=(25, 55, 86), outline=COL["cyan"], width=2)
        c.draw_fit_text(TextBlock(f"loop.step.{num}.num", (x + 32, y + 30, x + 52, y + 55), num, 18, min_size=14, fill=COL["cyan"], anchor="center"), spacing=1)
        c.draw_fit_text(TextBlock(f"loop.step.{num}.title", (x + 76, y + 24, x + card_w - 20, y + 68), title, 30, min_size=18, bold=True))
        c.draw_fit_text(TextBlock(f"loop.step.{num}.body", (x + 20, y + 84, x + card_w - 20, y + card_h - 22), body, 19, min_size=14, fill=(212, 226, 246)), spacing=5)

    notes = [
        ("Margin beats baseline", "True only if bootstrap P(E_setup > E_base) ≥ 0.80 AND gap ≥ +0.05R."),
        ("Regime conservation", "For each setup: n_bear + n_not_bear + n_unclassified = n_all."),
        ("Universe printed", "Evidence stores ticker list, per-name n/E_cons, and top-3 R share."),
    ]
    x = 80
    for title, body in notes:
        box = (x, 1010, x + 455, 1125)
        c.shadow_box(box, radius=20, fill=(14, 24, 40), outline=COL["amber"], width=2)
        c.draw_fit_text(TextBlock(f"loop.note.{title}.title", (x + 20, 1028, x + 435, 1058), title, 19, min_size=15, fill=COL["amber"]))
        c.draw_fit_text(TextBlock(f"loop.note.{title}.body", (x + 20, 1065, x + 435, 1110), body, 16, min_size=12, mono=True, fill=COL["muted"]), spacing=4)
        x += 495
    c.draw_fit_text(TextBlock("loop.generated", (80, 1150, 1050, 1175), "Generated docs asset · docs/assets/graphics/dailyedge-rule-loop.png", 16, min_size=12, mono=True, fill=COL["muted"]))
    if write:
        c.save(OUT / "dailyedge-rule-loop.png")
    return c


def render_all(*, write: bool) -> list[Canvas]:
    return [render_hero(write), render_architecture(write), render_rule_loop(write)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate text fits without writing files")
    args = parser.parse_args()

    canvases = render_all(write=not args.check)
    failures = [failure for canvas in canvases for failure in canvas.failures]
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    if args.check:
        for path in [OUT / "dailyedge-hero.png", OUT / "dailyedge-architecture.png", OUT / "dailyedge-rule-loop.png"]:
            if not path.exists():
                print(f"FAIL: missing generated asset {path.relative_to(ROOT)}")
                return 1
        print("README graphics layout check passed: all text blocks fit their boxes.")
    else:
        for path in sorted(OUT.glob("dailyedge-*.png")):
            img = Image.open(path)
            print(f"wrote {path.relative_to(ROOT)} {img.size} {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
