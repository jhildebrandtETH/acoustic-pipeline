"""Translate the normal pipeline's refinement geometry to native cfMesh objects."""

import math
from pathlib import Path
import numpy as np
import trimesh
from cfmesh_pipeline import query, write


def prepare_refinements(
    case,
    diameter,
    rotor_radius,
    rotor_half_length,
    segments,
    acoustic_surface,
    acoustic_diameter,
):
    parameters_directory = Path(case) / "Parameters"
    refinement_dictionary = parameters_directory / "cfmeshRefinementDict"
    generated_dictionary = parameters_directory / "cfmeshRegions.generated"
    if query(refinement_dictionary, "enabled").lower() in {"false", "no", "off", "0"}:
        write(generated_dictionary, "// Region refinements disabled.\n")
        return {}
    dimension_names = (
        "sphereDiameterFactor",
        "wakeOffsetFactor",
        "innerRadiusFactor",
        "innerHeightFactor",
        "outerRadiusFactor",
        "outerHeightFactor",
    )
    controls = {
        entry_name: float(query(refinement_dictionary, entry_name))
        for entry_name in dimension_names
    }
    if acoustic_surface == "permeable":
        controls["sphereDiameterFactor"] = float(acoustic_diameter)
    if not all(math.isfinite(entry_value) for entry_value in controls.values()):
        raise ValueError("Refinement dimensions must be finite")
    if any(
        entry_value <= 0
        for entry_name, entry_value in controls.items()
        if entry_name != "wakeOffsetFactor"
    ):
        raise ValueError("Refinement dimension factors must be positive")
    sphere_radius = 0.5 * controls["sphereDiameterFactor"] * diameter
    wake_offset = controls["wakeOffsetFactor"] * sphere_radius
    domain_dictionary = parameters_directory / "cfmeshDomainDict"
    box_minimum = np.fromstring(query(domain_dictionary, "boxMin").strip("()"), sep=" ")
    box_maximum = np.fromstring(query(domain_dictionary, "boxMax").strip("()"), sep=" ")
    if np.any(box_minimum >= -sphere_radius) or np.any(box_maximum <= sphere_radius):
        raise ValueError(
            "Refinement sphere must fit inside the domain; adjust sphereDiameterFactor or box bounds"
        )
    regions = {}
    for region_name, cylinder_radius, cylinder_half_length, cylinder_center_y in (
        ("rotaryRegion", rotor_radius, rotor_half_length, 0.0),
        (
            "innerCylinder",
            controls["innerRadiusFactor"] * sphere_radius,
            0.5 * controls["innerHeightFactor"] * sphere_radius,
            wake_offset,
        ),
        (
            "outerCylinder",
            controls["outerRadiusFactor"] * sphere_radius,
            0.5 * controls["outerHeightFactor"] * sphere_radius,
            wake_offset,
        ),
    ):
        if np.any(
            box_minimum
            >= [
                -cylinder_radius,
                cylinder_center_y - cylinder_half_length,
                -cylinder_radius,
            ]
        ) or np.any(
            box_maximum
            <= [
                cylinder_radius,
                cylinder_center_y + cylinder_half_length,
                cylinder_radius,
            ]
        ):
            raise ValueError(f"{region_name} refinement must fit inside the domain")
        cell_size = float(
            query(parameters_directory / "cfmeshCommon.cpp", region_name + "CellSize")
        )
        regions[region_name] = dict(
            type="cone",
            p0=[0.0, cylinder_center_y - cylinder_half_length, 0.0],
            p1=[0.0, cylinder_center_y + cylinder_half_length, 0.0],
            radius0=cylinder_radius,
            radius1=cylinder_radius,
            cellSize=cell_size,
        )
    regions["acousticSurface"] = dict(
        type="sphere",
        centre=[0.0, 0.0, 0.0],
        radius=sphere_radius,
        cellSize=float(
            query(parameters_directory / "cfmeshCommon.cpp", "acousticSphereCellSize")
        ),
    )
    if any(
        not math.isfinite(entry_value["cellSize"]) or entry_value["cellSize"] <= 0
        for entry_value in regions.values()
    ):
        raise ValueError("Refinement cell sizes must be positive and finite")

    def value(entry_value):
        if isinstance(entry_value, list):
            return (
                "("
                + " ".join(format(entry_value, ".12g") for entry_value in entry_value)
                + ")"
            )
        return (
            format(entry_value, ".12g")
            if isinstance(entry_value, float)
            else str(entry_value)
        )

    dictionary_text = "// Case snapshot: generated before cfMesh from the editable Parameters dictionaries.\n"
    for region_name, entries in regions.items():
        dictionary_text += (
            region_name
            + "\n{\n"
            + "".join(
                f"    {entry_name} {value(entry_value)};\n"
                for entry_name, entry_value in entries.items()
            )
            + "}\n"
        )
    write(generated_dictionary, dictionary_text)
    # Reference STLs make the same regions visible in ParaView, without meshing them as walls.
    surface_directory = Path(case) / "constant/triSurface"
    surface_directory.mkdir(parents=True, exist_ok=True)
    shaft_rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    for region_name, region in regions.items():
        if region["type"] == "cone":
            height = region["p1"][1] - region["p0"][1]
            region_surface = trimesh.creation.cylinder(
                radius=region["radius0"], height=height, sections=segments
            )
            region_surface.apply_transform(shaft_rotation)
            region_surface.apply_translation(
                [0.0, 0.5 * (region["p1"][1] + region["p0"][1]), 0.0]
            )
            surface_name = (
                "rotaryCylinder" if region_name == "rotaryRegion" else region_name
            )
        else:
            subdivisions = int(query(refinement_dictionary, "sphereSubdivisions"))
            if not 1 <= subdivisions <= 6:
                raise ValueError("sphereSubdivisions must be 1..6")
            region_surface = trimesh.creation.icosphere(
                subdivisions=subdivisions, radius=sphere_radius
            )
            surface_name = "acousticRefinementSphere"
        region_surface.export(surface_directory / (surface_name + ".stl"))
    return regions
