"""Render a comparison report as a Markdown report or a CSV compliance matrix."""
from __future__ import annotations

import csv
import io

from app.comparison.schemas import ComparisonCell, ComparisonReport

STATUS_LABELS = {
    "match": "Match",
    "deviation": "Deviation",
    "missing": "Missing",
    "not_in_baseline": "Not in baseline",
}
ASSESSMENT_LABELS = {
    "baseline": "Baseline",
    "compliant": "Compliant",
    "deviation": "Deviation",
    "not_specified": "Not specified",
    "no_requirement": "No requirement",
}


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _cell_markdown(cell: ComparisonCell) -> str:
    if cell.assessment == "not_specified":
        return "— *not specified*"
    text = _md(cell.value or "")
    if cell.page_number:
        text += f" (p.{cell.page_number})"
    if cell.assessment == "deviation":
        text = f"**⚠ {text}**"
        if cell.note:
            text += f"<br>*{_md(cell.note)}*"
    elif cell.assessment == "compliant":
        text = f"✓ {text}"
    if not cell.verified:
        text += " *(unverified)*"
    return text


def to_markdown(report: ComparisonReport) -> str:
    names = [_md(d.filename) for d in report.documents]
    baseline = report.filename_for(report.baseline_document_id)
    summary = report.summary
    lines = [
        "# Specification Comparison",
        "",
        f"**Baseline:** {_md(baseline)}  ",
        f"**Compared:** {', '.join(names[1:])}  ",
        f"**Generated:** {report.generated_at:%Y-%m-%d %H:%M} UTC",
        "",
        f"{summary['total']} parameters: {summary['match']} match, "
        f"{summary['deviation']} deviation, {summary['missing']} missing, "
        f"{summary['not_in_baseline']} not in baseline.",
    ]

    for aspect in report.aspects:
        lines += ["", f"## {aspect.title}", ""]
        if aspect.error:
            lines.append(f"> {aspect.error}")
            continue
        if aspect.summary:
            lines += [aspect.summary, ""]
        if not aspect.rows:
            lines.append("*No parameters found.*")
            continue
        header = ["Parameter", *[f"{n} (baseline)" if i == 0 else n for i, n in enumerate(names)], "Status"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for row in aspect.rows:
            cells = [_cell_markdown(cell) for cell in row.cells]
            lines.append(
                "| "
                + " | ".join([_md(row.parameter), *cells, STATUS_LABELS[row.status]])
                + " |"
            )

    if report.missing_specifications:
        lines += ["", "## Missing specifications", ""]
        for item in report.missing_specifications:
            lines.append(
                f"- **{_md(item['aspect'])} – {_md(item['parameter'])}**: "
                f"not stated in {', '.join(item['missing_in'])}"
            )
    return "\n".join(lines) + "\n"


def to_csv(report: ComparisonReport) -> str:
    """One row per parameter; value/assessment/page/quote per document."""
    output = io.StringIO()
    writer = csv.writer(output)
    header = ["Aspect", "Parameter", "Status"]
    for document in report.documents:
        header += [
            f"{document.filename} value",
            f"{document.filename} assessment",
            f"{document.filename} page",
            f"{document.filename} quote",
            f"{document.filename} note",
        ]
    writer.writerow(header)
    for aspect in report.aspects:
        for row in aspect.rows:
            line = [aspect.title, row.parameter, STATUS_LABELS[row.status]]
            for cell in row.cells:
                line += [
                    cell.value or "",
                    ASSESSMENT_LABELS[cell.assessment],
                    cell.page_number or "",
                    cell.quote or "",
                    cell.note or "",
                ]
            writer.writerow(line)
    return output.getvalue()