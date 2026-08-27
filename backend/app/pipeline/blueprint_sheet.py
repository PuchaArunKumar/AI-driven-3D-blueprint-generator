"""Compose the projected views into a single blueprint sheet as SVG.

The three orthographic views are laid out in **third-angle projection** - plan
above the front elevation, side elevation to its right - drawn at one shared
scale so the views can be measured against each other, with dimension lines and
a title block.

SVG rather than a raster: a blueprint is line art, so it should stay vector and
print at any size.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: The sheet is a real A3 landscape: 420 x 297 mm. Declaring physical units
#: makes the scale ratio in the title block meaningful rather than decorative.
SHEET_WIDTH = 1640
SHEET_HEIGHT = 1160
PAPER_WIDTH_MM = 420.0
PAPER_HEIGHT_MM = 297.0
#: Millimetres of paper per viewBox unit.
MM_PER_UNIT = PAPER_WIDTH_MM / SHEET_WIDTH
MARGIN = 26
TITLE_WIDTH = 470
TITLE_HEIGHT = 168
PANEL_GAP = 18
VIEW_PADDING = 58


@dataclass(frozen=True)
class SheetTheme:
    """Colours for one sheet style."""

    background: str
    grid: str
    grid_major: str
    outline: str
    inline: str
    hidden: str
    dimension: str
    text: str
    frame: str


THEMES: dict[str, SheetTheme] = {
    "white": SheetTheme(
        background="#ffffff", grid="#e9edf2", grid_major="#dbe2ea",
        outline="#11151c", inline="#5b6980", hidden="#a9b4c4",
        dimension="#2f6fb5", text="#3b4657", frame="#11151c",
    ),
    "blueprint": SheetTheme(
        background="#0e4b8f", grid="#2f6fb5", grid_major="#4a8ad0",
        outline="#ffffff", inline="#c9dcf5", hidden="#7ea8d8",
        dimension="#eaf2ff", text="#dbe8fa", frame="#ffffff",
    ),
}


def _format_length(metres: float) -> str:
    millimetres = metres * 1000.0
    if millimetres < 10:
        return f"{millimetres:.1f} mm"
    if millimetres < 1000:
        return f"{millimetres:.0f} mm"
    return f"{metres:.3f} m"


def _paths_to_svg(paths: list, transform, closed: bool) -> str:
    """Turn model-space polylines into a single SVG path string."""
    commands: list[str] = []
    for path in paths:
        if len(path) < 2:
            continue
        points = [transform(x, y) for x, y in path]
        segment = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in points)
        commands.append(segment + (" Z" if closed else ""))
    return " ".join(commands)


def _dimension(x1: float, y1: float, x2: float, y2: float, label: str,
               theme: SheetTheme, vertical: bool = False) -> str:
    """An arrowed dimension line with a centred label."""
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    rotate = f' transform="rotate(-90 {mid_x:.1f} {mid_y:.1f})"' if vertical else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{theme.dimension}" stroke-width="0.9" '
        f'marker-start="url(#dim-start)" marker-end="url(#dim-end)"/>'
        f'<text x="{mid_x:.1f}" y="{mid_y - 5:.1f}" fill="{theme.dimension}" '
        f'font-size="13" font-family="ui-monospace,monospace" '
        f'text-anchor="middle"{rotate}>{html.escape(label)}</text>'
    )


def _panel(view: dict, box: tuple[float, float, float, float], scale: float,
           title: str, code: str, horizontal_label: str, vertical_label: str,
           theme: SheetTheme, origin: tuple[float, float]) -> str:
    """Render one view in its panel at a given drawing origin.

    ``origin`` is the sheet position of the view's bottom-left corner. Passing
    it explicitly (rather than centring in the panel) is what keeps the plan
    vertically aligned with the front elevation and the side elevation
    horizontally aligned with it - the defining property of a projection
    layout.
    """
    left, top, width, height = box
    parts = [
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="none" stroke="{theme.frame}" stroke-width="0.8" opacity="0.45"/>',
        f'<text x="{left + 12:.1f}" y="{top + 22:.1f}" fill="{theme.text}" '
        f'font-size="13" font-weight="600" font-family="Inter,system-ui,sans-serif" '
        f'letter-spacing="1.4">{html.escape(title.upper())}</text>',
        f'<text x="{left + width - 12:.1f}" y="{top + 22:.1f}" fill="{theme.text}" '
        f'font-size="12" font-family="ui-monospace,monospace" '
        f'text-anchor="end" opacity="0.7">{html.escape(code)}</text>',
    ]

    view_width = view.get("width", 0.0)
    view_height = view.get("height", 0.0)
    if view_width <= 0 or view_height <= 0:
        parts.append(
            f'<text x="{left + width / 2:.1f}" y="{top + height / 2:.1f}" '
            f'fill="{theme.text}" font-size="13" text-anchor="middle" '
            f'opacity="0.6">no geometry</text>'
        )
        return "".join(parts)

    drawn_width = view_width * scale
    drawn_height = view_height * scale
    origin_x, origin_y = origin

    minimum_x = min(
        (point[0] for group in ("outline", "inline", "hidden")
         for path in view.get(group, []) for point in path),
        default=0.0,
    )
    minimum_y = min(
        (point[1] for group in ("outline", "inline", "hidden")
         for path in view.get(group, []) for point in path),
        default=0.0,
    )

    def transform(x: float, y: float) -> tuple[float, float]:
        # SVG y grows downward; model y grows up.
        return origin_x + (x - minimum_x) * scale, origin_y - (y - minimum_y) * scale

    hidden = _paths_to_svg(view.get("hidden", []), transform, closed=False)
    inline = _paths_to_svg(view.get("inline", []), transform, closed=False)
    outline = _paths_to_svg(view.get("outline", []), transform, closed=True)

    if hidden:
        parts.append(f'<path d="{hidden}" fill="none" stroke="{theme.hidden}" '
                     f'stroke-width="0.7" stroke-dasharray="5 4" opacity="0.75"/>')
    if inline:
        parts.append(f'<path d="{inline}" fill="none" stroke="{theme.inline}" '
                     f'stroke-width="0.85" stroke-linecap="round"/>')
    if outline:
        parts.append(f'<path d="{outline}" fill="none" stroke="{theme.outline}" '
                     f'stroke-width="1.9" stroke-linejoin="round"/>')

    # Dimension lines below and to the left of the drawn extent.
    baseline = origin_y + 26
    parts.append(_dimension(origin_x, baseline, origin_x + drawn_width, baseline,
                            f"{horizontal_label} {_format_length(view_width)}", theme))
    left_line = origin_x - 26
    parts.append(_dimension(left_line, origin_y, left_line, origin_y - drawn_height,
                            f"{vertical_label} {_format_length(view_height)}", theme,
                            vertical=True))
    return "".join(parts)


def _title_block(name: str, stats: dict, spec: dict | None, scale_label: str,
                 theme: SheetTheme) -> str:
    left = SHEET_WIDTH - MARGIN - TITLE_WIDTH
    top = SHEET_HEIGHT - MARGIN - TITLE_HEIGHT
    rows = [
        ("SCALE", scale_label),
        ("SHEET", "A3 landscape"),
        ("PROJECTION", "third angle"),
        ("MATERIAL", ", ".join((spec or {}).get("materials") or []) or "not specified"),
        ("TRIANGLES", f"{stats.get('triangles', 0):,}"),
        ("TOLERANCE",
         f"±{spec['tolerances_mm']} mm" if (spec or {}).get("tolerances_mm")
         else "not specified"),
    ]

    parts = [
        f'<rect x="{left}" y="{top}" width="{TITLE_WIDTH}" height="{TITLE_HEIGHT}" '
        f'fill="none" stroke="{theme.frame}" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{top + 42}" x2="{left + TITLE_WIDTH}" y2="{top + 42}" '
        f'stroke="{theme.frame}" stroke-width="1"/>',
        f'<text x="{left + 14}" y="{top + 28}" fill="{theme.outline}" font-size="18" '
        f'font-weight="700" font-family="Inter,system-ui,sans-serif">'
        f'{html.escape(name[:38])}</text>',
    ]
    for index, (key, value) in enumerate(rows):
        row_y = top + 66 + (index % 3) * 32
        column_x = left + 14 + (index // 3) * 232
        parts.append(
            f'<text x="{column_x}" y="{row_y}" fill="{theme.text}" font-size="10" '
            f'font-family="ui-monospace,monospace" letter-spacing="1.1" '
            f'opacity="0.75">{key}</text>'
            f'<text x="{column_x}" y="{row_y + 15}" fill="{theme.outline}" '
            f'font-size="13" font-family="ui-monospace,monospace">'
            f'{html.escape(str(value))}</text>'
        )
    return "".join(parts)


def render_blueprint_sheet(drawing: dict, stats: dict, spec: dict | None,
                           name: str = "Generated model",
                           theme_name: str = "white") -> str:
    """Compose the three views plus a title block into one SVG sheet."""
    theme = THEMES.get(theme_name, THEMES["white"])

    content_width = SHEET_WIDTH - 2 * MARGIN
    content_height = SHEET_HEIGHT - 2 * MARGIN
    panel_width = (content_width - PANEL_GAP) / 2
    panel_height = (content_height - PANEL_GAP) / 2

    # Third angle: plan above the front elevation, side elevation to its right.
    boxes = {
        "top": (MARGIN, MARGIN, panel_width, panel_height),
        "front": (MARGIN, MARGIN + panel_height + PANEL_GAP, panel_width, panel_height),
        "side": (MARGIN + panel_width + PANEL_GAP, MARGIN + panel_height + PANEL_GAP,
                 panel_width, panel_height),
    }

    def extent(view: str, key: str) -> float:
        return max(float(drawing.get(view, {}).get(key, 0.0) or 0.0), 1e-9)

    # One scale across every view, so the sheet is internally measurable.
    scale = min(
        min((panel_width - 2 * VIEW_PADDING) / extent(view, "width"),
            (panel_height - 2 * VIEW_PADDING) / extent(view, "height"))
        for view in boxes if drawing.get(view)
    )

    # Front sits at the heart of the layout; the other two align to it.
    front_width = extent("front", "width") * scale
    front_height = extent("front", "height") * scale
    front_x = MARGIN + (panel_width - front_width) / 2
    front_y = MARGIN + panel_height + PANEL_GAP + (panel_height + front_height) / 2

    top_height = extent("top", "height") * scale
    side_width = extent("side", "width") * scale
    side_panel_left = MARGIN + panel_width + PANEL_GAP

    origins = {
        # Shares the front's horizontal placement: both measure width across.
        "top": (front_x, MARGIN + (panel_height + top_height) / 2),
        "front": (front_x, front_y),
        # Shares the front's baseline: both measure height up.
        "side": (side_panel_left + (panel_width - side_width) / 2, front_y),
    }

    # With physical paper units the ratio is real: how many millimetres of
    # model one millimetre of paper represents.
    paper_mm_per_metre = scale * MM_PER_UNIT
    ratio = 1000.0 / paper_mm_per_metre if paper_mm_per_metre > 0 else 1.0
    scale_label = f"1:{ratio:.0f}" if ratio >= 10 else f"1:{ratio:.1f}"

    labels = {
        "front": ("Front elevation", "A-01", "W", "H"),
        "side": ("Side elevation", "A-02", "D", "H"),
        "top": ("Plan", "A-03", "W", "D"),
    }

    panels = "".join(
        _panel(drawing[view], boxes[view], scale, *labels[view], theme, origins[view])
        for view in boxes if drawing.get(view)
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{PAPER_WIDTH_MM}mm" \
height="{PAPER_HEIGHT_MM}mm" viewBox="0 0 {SHEET_WIDTH} {SHEET_HEIGHT}">
  <defs>
    <pattern id="fine" width="16" height="16" patternUnits="userSpaceOnUse">
      <path d="M16 0 L0 0 0 16" fill="none" stroke="{theme.grid}"
            stroke-width="0.6" opacity="0.55"/>
    </pattern>
    <pattern id="coarse" width="96" height="96" patternUnits="userSpaceOnUse">
      <rect width="96" height="96" fill="url(#fine)"/>
      <path d="M96 0 L0 0 0 96" fill="none" stroke="{theme.grid_major}" stroke-width="0.9"/>
    </pattern>
    <marker id="dim-end" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,1 L8,4 L0,7 z" fill="{theme.dimension}"/>
    </marker>
    <marker id="dim-start" markerWidth="8" markerHeight="8" refX="1" refY="4" orient="auto">
      <path d="M8,1 L0,4 L8,7 z" fill="{theme.dimension}"/>
    </marker>
  </defs>
  <rect width="{SHEET_WIDTH}" height="{SHEET_HEIGHT}" fill="{theme.background}"/>
  <rect width="{SHEET_WIDTH}" height="{SHEET_HEIGHT}" fill="url(#coarse)"/>
  <rect x="{MARGIN / 2}" y="{MARGIN / 2}" width="{SHEET_WIDTH - MARGIN}" \
height="{SHEET_HEIGHT - MARGIN}" fill="none" stroke="{theme.frame}" stroke-width="2"/>
  {panels}
  {_title_block(name, stats, spec, scale_label, theme)}
  <text x="{MARGIN + 6}" y="{SHEET_HEIGHT - MARGIN - 10}" fill="{theme.text}" \
font-size="11" font-family="Inter,system-ui,sans-serif" opacity="0.8">
    Design output - not a certified manufacturing drawing. No GD&amp;T or material certification.
  </text>
</svg>"""
