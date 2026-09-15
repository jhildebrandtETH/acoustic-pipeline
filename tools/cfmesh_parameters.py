"""Resolve sphere-relative domain dimensions and base/level resolution inputs."""

import math
from pathlib import Path

from tools.cfmesh_pipeline import optional, query


REGIONS = (
    "propeller", "interface", "rotaryRegion", "innerCylinder",
    "outerCylinder", "acousticSphere",
)


def effective_sphere_radius(parameters, diameter, acoustic_surface, acoustic_diameter):
    factor = (
        float(acoustic_diameter)
        if acoustic_surface == "permeable" and acoustic_diameter is not None
        else float(query(Path(parameters) / "cfmeshRefinementDict", "sphereDiameterFactor"))
    )
    radius = 0.5 * factor * diameter
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Acoustic/refinement sphere radius must be positive and finite")
    return radius


def relative_box(radius, lateral_margin, inlet_margin, inlet_fraction):
    if not all(math.isfinite(v) for v in (radius, lateral_margin, inlet_margin, inlet_fraction)):
        raise ValueError("Sphere-relative domain inputs must be finite")
    if radius <= 0 or lateral_margin <= 0 or inlet_margin <= 0:
        raise ValueError("Sphere radius and domain margins must be positive")
    if not 0 < inlet_fraction < 1:
        raise ValueError("inletFraction must be strictly between 0 and 1")
    side = radius * (1 + lateral_margin)
    inlet = radius * (1 + inlet_margin)
    outlet = inlet * (1 - inlet_fraction) / inlet_fraction
    if not all(math.isfinite(v) for v in (side, inlet, outlet)):
        raise ValueError("Derived sphere-relative domain bounds must be finite")
    if outlet <= radius:
        raise ValueError("inletFraction leaves insufficient outlet distance to contain the sphere")
    return [-side, -outlet, -side], [side, inlet, side]


def domain_bounds(parameters, radius):
    path = Path(parameters) / "cfmeshDomainDict"
    mode = optional(path, "boxSizing") or "absolute"  # Legacy case snapshots.
    if mode == "sphereRelative":
        settings = {
            key: float(query(path, key))
            for key in ("lateralMargin", "inletMargin", "inletFraction")
        }
        lower, upper = relative_box(
            radius, settings["lateralMargin"], settings["inletMargin"], settings["inletFraction"],
        )
    elif mode == "absolute":
        lower, upper = (
            list(map(float, query(path, key).strip("()").split()))
            for key in ("boxMin", "boxMax")
        )
        settings = {}
    else:
        raise ValueError("cfmeshDomainDict boxSizing must be sphereRelative or absolute")
    if len(lower) != 3 or len(upper) != 3 or any(
        not math.isfinite(a) or not math.isfinite(b) or a >= b
        for a, b in zip(lower, upper)
    ):
        raise ValueError("Invalid domain bounds")
    return lower, upper, dict(
        mode=mode, sphere_radius_m=radius, **settings,
        centre_y_m=0.5 * (lower[1] + upper[1]),
    )


def level_cell_size(base, level):
    if not math.isfinite(base) or base <= 0:
        raise ValueError("baseCellSize must be positive and finite")
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError("Refinement levels must be nonnegative integers")
    try:
        size = math.ldexp(base, -level)
    except OverflowError as exc:
        raise ValueError("Refinement level is too large") from exc
    if size <= 0:
        raise ValueError("Refinement level is too large")
    return size


def resolution_settings(parameters):
    path = Path(parameters) / "cfmeshCommon.cpp"
    base_text = optional(path, "baseCellSize")
    levels = {}
    if base_text is not None:
        base = float(base_text)
        sizes = {"background": level_cell_size(base, 0)}
        for region in REGIONS:
            raw = query(path, region + "Level")
            try:
                level = int(raw)
                sizes[region] = level_cell_size(base, level)
            except ValueError as exc:
                raise ValueError(f"Invalid {region}Level: {raw!r}: {exc}") from exc
            levels[region] = level
        mode = "levels"
    else:
        # Old snapshots keep their absolute sizes and their original mesh includes.
        sizes = {
            region: float(query(path, region + "CellSize"))
            for region in ("background", *REGIONS)
        }
        if any(not math.isfinite(size) or size <= 0 for size in sizes.values()):
            raise ValueError("Cell sizes must be positive and finite")
        base = sizes["background"]
        mode = "absolute"
    return dict(mode=mode, base_cell_size_m=base, levels=levels, cell_sizes_m=sizes)
