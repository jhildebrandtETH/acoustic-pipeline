"""checkMesh diagnostics and a transparent, deliberately heuristic quality index.

Set messages follow OpenFOAM checkGeometry/checkTopology output. Counts from
different sets may overlap; they must never be presented as unique bad cells.
"""
import math
import re
from pathlib import Path


def latest_mesh_log(case_path):
    """Select the last mesh block, including an unfinished last check."""
    path = Path(case_path) / "log.checkMesh"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Appended executions and multi-time logs must not reuse an earlier success.
    text = re.split(r"(?m)^\s*Exec\s*:", text)[-1]
    blocks = re.split(r"(?m)^\s*Mesh stats\s*$", text)
    return blocks[-1] if len(blocks) > 1 else text


def read_mesh_quality(case_path, mesh_info):
    text = latest_mesh_log(case_path)
    endings = list(re.finditer(r"Mesh OK\.?|Failed\s+(\d+)\s+mesh checks?", text))
    failed = int(endings[-1][1] or 0) if endings else None
    diagnostics = {}
    for match in re.finditer(
        r"Writing\s+(\d+)\s+([^\n]*?)\s+to set\s+(\w+)", text
    ):
        count, description, name = match.groups()
        # The first entity word is the set type (e.g. faces with low volume ratio cells).
        entity = re.search(r"\b(cells|faces|points|edges)\b", description)
        if entity:
            diagnostics[name] = dict(name=name, label=name, entity=entity[1], count=int(count))

    # Some versions report counts without writing sets. Prefer the set if present.
    fallbacks = [
        ("nonOrthoFaces", "faces", r"Number of severely non-orthogonal[^\n:]*:\s*(\d+)"),
        ("skewFaces", "faces", r"(\d+)\s+highly skew faces"),
        ("concaveCells", "cells", r"(\d+)\s+concave cells"),
        ("underdeterminedCells", "cells", r"(\d+)\s+under-determined cells"),
        ("zeroVolumeCells", "cells", r"(?:zero or negative volume cells\s*:\s*)(\d+)"),
        # Foundation v13 reports these counts without Writing ... to set lines.
        ("underdeterminedCells", "cells", r"Cells with small determinant[^\n]*number of cells:\s*(\d+)"),
        ("concaveCells", "cells", r"Concave cells[^\n]*number of cells:\s*(\d+)"),
        ("lowWeightFaces", "faces", r"Faces with small interpolation weight[^\n]*number of faces:\s*(\d+)"),
        ("lowVolRatioFaces", "faces", r"Faces with small (?:volume|vol) ratio[^\n]*number of faces:\s*(\d+)"),
        ("lowQualityTetFaces", "faces", r"Error in face tets:\s*(\d+)\s+faces"),
        ("concaveFaces", "faces", r"There are\s+(\d+)\s+faces with concave angles"),
        ("warpedFaces", "faces", r"There are\s+(\d+)\s+faces with ratio between projected and actual area"),
        ("shortEdges", "edges", r"Edges too small[^\n]*number too small:\s*(\d+)"),
    ]
    for name, entity, pattern in fallbacks:
        match = re.search(pattern, text, re.I)
        if match and name not in diagnostics:
            diagnostics[name] = dict(name=name, label=name, entity=entity, count=int(match[1]))
        if match:
            line_start = text.rfind("\n", 0, match.start()) + 1
            prefix = text[line_start:match.start()].lstrip()
            diagnostics[name]["status"] = "Failed" if prefix.startswith("***") else "Reported"

    rows = sorted(diagnostics.values(), key=lambda row: (row["entity"] != "cells", row["entity"], row["name"]))
    for row in rows:
        row.setdefault("status", "Reported")
        denominator = mesh_info.get(row["entity"])
        cells = mesh_info.get("cells")
        row["percent"] = 100 * row["count"] / denominator if denominator else None
        row["per_100_cells"] = 100 * row["count"] / cells if cells else None
        # Each defect category costs 0..20 points, on a logarithmic prevalence scale.
        row["penalty"] = min(20.0, 5 * math.log10(1 + row["percent"] / .01)) if row["percent"] is not None else None

    cell_counts = [row["count"] for row in rows if row["entity"] == "cells"]
    cells = mesh_info.get("cells")
    bounds = (max(cell_counts), min(sum(cell_counts), cells)) if cell_counts and cells else None
    critical_names = {"zeroVolumeCells", "nonClosedCells", "zeroAreaFaces", "outOfRangeFaces", "zipUpCells"}
    critical = any(row["count"] > 0 and row["name"] in critical_names for row in rows)
    min_volume = mesh_info.get("min_volume")
    critical = critical or (min_volume is not None and min_volume <= 0)
    valid = (failed is not None and bool(cells) and bool(mesh_info.get("faces"))
             and bool(mesh_info.get("points"))
             and all((row["entity"] == "edges" and row["percent"] is None)
                     or (row["percent"] is not None and row["percent"] <= 100) for row in rows))
    score = None
    raw_score = None
    adjustment = .19 if critical else .59 if failed else 1.
    defect_penalty = sum(row["penalty"] for row in rows if row["penalty"] is not None)
    failure_penalty = min(20, 5 * failed) if failed is not None else None
    if valid:
        raw_score = max(0., 100 - defect_penalty - failure_penalty)
        # A hard min(score, 59) hid all improvements above the ceiling. Scale
        # the whole range instead so prevalence still moves the score/marker.
        score = round(raw_score * adjustment, 1)
    regime = ("Unavailable" if score is None else "Critical" if score < 40 else
              "Poor" if score < 60 else "Review" if score < 80 else "Good" if score < 95 else "Excellent")
    warnings = [line.strip() for line in text.splitlines() if re.match(r"\s*\*+\s*\S", line)]
    failed_messages = [line for line in warnings if line.startswith("***")]
    return dict(score=score, regime=regime, failed_checks=failed, critical=critical,
                index_version=2, raw_score=raw_score, adjustment=adjustment,
                defect_penalty=defect_penalty, failure_penalty=failure_penalty,
                diagnostics=rows, cell_bounds=bounds, warnings=warnings,
                failed_messages=failed_messages,
                scope="Logged checks only; unreported checks are not assumed to pass.")


def append_mesh_quality_report(c, mesh_info, quality):
    """Start a dedicated quality page; paginate all diagnostics without truncation."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle
    from xml.sax.saxutils import escape

    width, height = A4
    usable = width - 100
    style = ParagraphStyle("mesh-quality", fontName="Helvetica", fontSize=9, leading=13)
    y = 0

    def page():
        nonlocal y
        c.showPage()
        c.setPageSize(A4)
        c.setFillColor(colors.HexColor("#142f43"))
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, height - 52, "Mesh Quality Metrics")
        c.setFillColor(colors.black)
        y = height - 78

    def paragraph(text, bold=False):
        nonlocal y
        p = Paragraph(("<b>" + escape(text) + "</b>") if bold else escape(text), style)
        _, ph = p.wrap(usable, height)
        if y - ph < (80 if bold else 45):
            page()
        p.drawOn(c, 50, y - ph)
        y -= ph + 7

    def table(data, widths):
        nonlocal y
        wrapped = [[Paragraph(escape(str(value)), style) for value in row] for row in data]
        t = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5edf2")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7f9")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        while t is not None:
            _, th = t.wrap(usable, height)
            if th <= y - 45:
                t.drawOn(c, 50, y - th)
                y -= th + 12
                break
            parts = t.split(usable, y - 45)
            if parts:
                _, th = parts[0].wrap(usable, height)
                parts[0].drawOn(c, 50, y - th)
                t = parts[1] if len(parts) > 1 else None
            page()

    page()
    score = quality["score"]
    paragraph(f"Mesh Quality Index: {'N/A' if score is None else f'{score:.1f} / 100'}  |  {quality['regime']}", True)
    paragraph(f"checkMesh: {mesh_info['status']}  |  Failed checks: {quality['failed_checks'] if quality['failed_checks'] is not None else 'not confirmed'}")
    for index in range(100):
        low, high, start, end = (("#d64040", "#f1c34b", 0, 70) if index < 70
                                 else ("#f1c34b", "#24995a", 70, 100))
        c.setFillColor(colors.linearlyInterpolatedColor(colors.HexColor(low), colors.HexColor(high), start, end, index))
        c.rect(50 + usable * index / 100, y - 16, usable / 100 + .1, 16, fill=1, stroke=0)
    c.setFillColor(colors.black)
    if score is not None:
        x = 50 + usable * score / 100
        c.setLineWidth(2)
        c.line(x, y + 3, x, y - 20)
    c.setFont("Helvetica", 8)
    for value in (0, 40, 60, 80, 95, 100):
        c.drawCentredString(50 + usable * value / 100, y - 30, str(value))
    y -= 45
    paragraph("Regimes: <40 Critical | 40-59 Poor | 60-79 Review | 80-94 Good | 95-100 Excellent")
    paragraph("Bad-cell overview", True)
    paragraph("Failed checks count failed tests, not cell-defect categories. Rows marked Failed have an explicit failure message; Reported rows may be warnings or diagnostic sets.")
    bounds = quality["cell_bounds"]
    if bounds:
        lo, hi = bounds
        cells = mesh_info["cells"]
        paragraph(f"Cells flagged by reported cell sets: {lo:,} to {hi:,} ({100*lo/cells:.4g}% to {100*hi/cells:.4g}% of {cells:,} cells). Bounds account for unknown overlap.")
    else:
        paragraph("No cell-defect counts reported. Unique bad-cell share is unavailable from this log.")
    paragraph("Sets may overlap. Face/point counts are not bad-cell counts; per 100 cells is a normalized incidence, not a cell percentage.")
    rows = [["Diagnostic / entity", "Count", "% of entity", "Per 100 cells", "Penalty"]]
    for row in quality["diagnostics"]:
        fmt = lambda value: "N/A" if value is None else f"{value:.4g}"
        rows.append([f"{row['label']} ({row['entity']}) - {row['status']}", f"{row['count']:,}", fmt(row["percent"]), fmt(row["per_100_cells"]), fmt(row["penalty"])])
    if len(rows) > 1:
        table(rows, [195, 62, 78, 88, usable - 423])
    else:
        paragraph("No defect-set counts found in log.checkMesh.")
    if any(row["percent"] is None for row in quality["diagnostics"]):
        paragraph("N/A means the entity total is absent from the log (e.g. total edges). These rows remain visible but contribute no prevalence penalty; the failed-check penalty still applies.")
    paragraph("Geometry statistics", True)
    metrics = [("Max aspect ratio", "max_aspect_ratio"), ("Max skewness", "max_skewness"),
               ("Max non-orthogonality [deg]", "max_non_orthogonality"),
               ("Mean non-orthogonality [deg]", "mean_non_orthogonality"),
               ("Min cell volume", "min_volume"), ("Min determinant", "min_determinant")]
    table([["Metric", "Value"]] + [[label, "Not reported" if mesh_info.get(key) is None else f"{mesh_info[key]:.6g}"] for label, key in metrics], [usable * .65, usable * .35])
    paragraph("How the index works (heuristic v2)", True)
    paragraph("Start at 100. Each reported defect category subtracts min(20, 5 log10(1 + p/0.01)) points, where p is its percentage of cells, faces or points. Subtract another min(20, 5 x failed checks), with a floor of zero. Multiply by 0.59 for failed checks, or 0.19 for critical volume/topology defects (otherwise 1.00). This replaces the v1 hard caps so relative improvements remain visible.")
    if score is not None:
        paragraph(f"This mesh: max(0, 100 - {quality['defect_penalty']:.2f} defect points - {quality['failure_penalty']:.2f} failure points) x {quality['adjustment']:.2f} = {score:.1f} / 100.")
    paragraph("Prevalence examples: 0.01% costs 1.5 points; 0.1% costs 5.2; 1% costs 10.0; 10% costs 15.0. Penalties accumulate across categories; they are not unique-cell percentages.")
    paragraph("The failure total includes checks without defect counts. Geometry extrema are context, not extra penalties. Missing completion or cell/face/point denominators gives N/A. This is a screening index, not a solver-accuracy guarantee. Compare scores only with equivalent checkMesh options.")
    paragraph(quality["scope"])
    if quality["warnings"]:
        paragraph(f"checkMesh messages ({len(quality['failed_messages'])} explicit failure messages; {quality['failed_checks']} failed checks in summary)", True)
        for warning in quality["warnings"]:
            paragraph(warning)
