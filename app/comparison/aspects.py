"""Specification aspects the comparator knows how to look for."""
from __future__ import annotations

import re

from pydantic import BaseModel


class ComparisonAspect(BaseModel):
    key: str
    title: str
    # Retrieval query used to find relevant chunks in each document.
    query: str
    # Parameters the extractor should look for under this aspect.
    focus: str


DEFAULT_ASPECTS: list[ComparisonAspect] = [
    ComparisonAspect(
        key="fan_motor",
        title="Fan motor ratings",
        query=(
            "fan motor horsepower kW voltage phase enclosure TEFC speed rpm "
            "insulation class service factor efficiency gearbox drive VFD"
        ),
        focus=(
            "motor type and enclosure (e.g. TEFC), power (hp/kW), voltage, "
            "phase and frequency, speed, insulation class, service factor, "
            "efficiency, torque type, drive or gearbox, variable frequency "
            "drive features, fan type, blade material and bearing life"
        ),
    ),
    ComparisonAspect(
        key="control_panel",
        title="Control panels",
        query=(
            "control panel enclosure NEMA disconnect switch contactor "
            "transformer selector switch terminal block building automation"
        ),
        focus=(
            "enclosure rating (NEMA/IP), disconnect type and lockout, "
            "fusing, control transformer, contactors, selector switches "
            "(HAND-OFF-AUTO, bypass), indicator lights, terminal blocks, "
            "BAS/BMS interface and temperature controller"
        ),
    ),
    ComparisonAspect(
        key="safety_equipment",
        title="Safety equipment",
        query=(
            "ladder handrail knee-rail toe-board safety cage platform "
            "walkway fan guard vibration switch shutdown live load"
        ),
        focus=(
            "ladders, handrails, knee-rails, toe-boards, safety cages, "
            "access platforms and walkways (materials, dimensions, design "
            "loads, governing codes), fan guards, vibration cut-out switches "
            "and safety shutdowns"
        ),
    ),
    ComparisonAspect(
        key="pan_heater",
        title="Pan heaters",
        query=(
            "pan heater basin electric immersion heater kW water temperature "
            "ambient temperature low water cut-off heater control panel"
        ),
        focus=(
            "heater type and material, number and power (kW), basin water "
            "temperature to maintain, design ambient temperature and wind "
            "speed, low-water cut-off, control probe and heater control "
            "panel features"
        ),
    ),
]

_ASPECTS_BY_KEY = {aspect.key: aspect for aspect in DEFAULT_ASPECTS}


def custom_aspect(text: str) -> ComparisonAspect:
    text = " ".join(text.split())
    key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "custom"
    return ComparisonAspect(key=key, title=text, query=text, focus=text)


def resolve_aspects(names: list[str] | None) -> list[ComparisonAspect]:
    """Map aspect keys to defaults; any other text becomes a custom aspect."""
    if not names:
        return list(DEFAULT_ASPECTS)
    resolved: list[ComparisonAspect] = []
    seen: set[str] = set()
    for name in names:
        if not name.strip():
            continue
        aspect = _ASPECTS_BY_KEY.get(name.strip().lower()) or custom_aspect(name)
        if aspect.key not in seen:
            seen.add(aspect.key)
            resolved.append(aspect)
    return resolved or list(DEFAULT_ASPECTS)
