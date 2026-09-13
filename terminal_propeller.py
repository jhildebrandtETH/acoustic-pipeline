"""Small decorative rotor for the live terminal dashboard (no CFD data)."""

import math


WIDTH = 37
HEIGHT = 16
PALETTE = {1: 236, 2: 238, 3: 239, 4: 66}


def pixels(elapsed, width=WIDTH, height=HEIGHT):
    """Rasterize two swept blades and fading tip streams on square pixels."""
    grid = [[0] * width for _ in range(height * 2)]
    cx, cy = (width - 1) / 2, (height * 2 - 1) / 2
    scale = min(width - 2, height * 2 - 2) / 30
    angle = elapsed * 0.9

    # Older particles lag behind the tips and drift gently outward.
    for blade in (0, math.pi):
        steps = max(100, int(100 * scale))
        for step in range(steps, -1, -1):
            age = step / steps
            theta = angle + blade + math.atan2(0.81, 11) - age * 1.7
            radius = math.hypot(11, 0.81) + age * 3.2
            radius += 0.18 * math.sin(age * 20)
            x = round(cx + scale * radius * math.cos(theta))
            y = round(cy + scale * radius * math.sin(theta))
            if 0 <= x < width and 0 <= y < height * 2:
                grid[y][x] = 2 if age < 0.45 else 1

    cosine, sine = math.cos(angle), math.sin(angle)
    for y, row in enumerate(grid):
        for x in range(width):
            dx, dy = (x - cx) / scale, (y - cy) / scale
            along = dx * cosine + dy * sine
            across = -dx * sine + dy * cosine
            radius = abs(along)
            if 1.2 <= radius <= 11:
                sweep = 0.09 * (radius - 2)
                half_width = 0.5 + 1.25 * math.sin(math.pi * (radius - 1.2) / 9.8)
                if abs(across * (1 if along >= 0 else -1) - sweep) <= half_width:
                    row[x] = 3
            if dx * dx + dy * dy <= 2.8:
                row[x] = 4
    return grid


def frame(elapsed, *, width=WIDTH, height=HEIGHT, overlay=(), unicode=True, color=True):
    """Pack square pixels into terminal half blocks; support plain encodings."""
    grid = pixels(elapsed, width, height)
    lines = []
    for y in range(0, height * 2, 2):
        text = overlay[y // 2] if y // 2 < len(overlay) else ""
        # Status rows stay in the terminal's normal foreground/background.
        # Blank rows and the space to the right reveal the moving backdrop.
        parts = [text]
        previous_style = None
        for top, bottom in zip(grid[y][len(text):], grid[y + 1][len(text):]):
            if not unicode:
                level = max(top, bottom)
                glyph = " ..:+"[level]
                fg, bg = level, 0
            elif top:
                glyph, fg, bg = "▀", top, bottom
            elif bottom:
                glyph, fg, bg = "▄", bottom, 0
            else:
                glyph, fg, bg = " ", 0, 0
            if not color and unicode and top and bottom:
                glyph = "█"
            if color:
                style = (fg, bg)
                if style != previous_style:
                    foreground = f"38;5;{PALETTE[fg]}" if fg else "39"
                    background = f"48;5;{PALETTE[bg]}" if bg else "49"
                    parts.append(f"\033[{foreground};{background}m")
                    previous_style = style
            parts.append(glyph)
        lines.append("".join(parts) + ("\033[0m" if color else ""))
    return lines


def status_overlay(text, columns, rows):
    """Wrap foreground text, reserving the last column/row to avoid scrolling."""
    if columns < 20 or rows < 10:
        return None
    width = columns - 1
    lines = []
    for line in text.split("\n"):
        lines.extend([line[start:start + width] for start in range(0, len(line), width)] or [""])
    return lines if len(lines) < rows else None
