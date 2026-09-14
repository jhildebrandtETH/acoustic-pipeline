"""Native cfMesh rotor/stator backend for the existing acoustic-pipeline scheduler."""

from cfmesh_runtime import (
    command as native_command,
    run as native_run,
    preflight as native_preflight,
)
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading

ROOT = Path(__file__).resolve().parent
BACKEND = "cfmesh-native-rotor-stator-v1"
OUTPUT_LOCK = threading.Lock()


def _path_key(path):
    """Compare Windows-mounted paths case-insensitively, including resolved links."""
    text = Path(path).as_posix().rstrip("/")
    if os.name == "nt" or re.match(r"^/mnt/[a-zA-Z]/", text):
        text = text.casefold()
    return text


def protected_original_paths():
    # Keep the original protected even when this duplicate is moved elsewhere.
    fixed = Path(
        "C:/repos/acoustic-pipeline"
        if os.name == "nt"
        else "/mnt/c/repos/acoustic-pipeline"
    )
    return (fixed.resolve(), (ROOT.parent / "acoustic-pipeline").resolve())


def safe_path(path):
    path = Path(path).expanduser().resolve()
    key = _path_key(path)
    for original in protected_original_paths():
        original_key = _path_key(original)
        if key == original_key or key.startswith(original_key + "/"):
            raise ValueError(
                f"The original repository is protected and cannot contain simulation output: {original}"
            )
    if path == ROOT or path == Path(path.anchor):
        raise ValueError(
            "Choose a dedicated simulation directory, not the pipeline or filesystem root"
        )
    return path


def query(path, entry=None):
    command = ["foamDictionary", Path(path).resolve()]
    command += ["-entry", entry, "-value"] if entry else ["-expand"]
    dictionary_process = native_run(
        command, cwd=Path(path).resolve().parent, text=True, capture_output=True
    )
    if dictionary_process.returncode:
        raise ValueError(
            f"Cannot read {path} {entry or ''}: {dictionary_process.stderr.strip()}"
        )
    return dictionary_process.stdout.strip()


def optional(path, entry):
    try:
        return query(path, entry)
    except ValueError:
        return None


def parameter_file(parameters, name, *, read_only=False):
    if not name or Path(name).name != name or "\\" in name or name in {".", ".."}:
        raise ValueError("Study file must be a filename in Parameters, without a path")
    for candidate in (Path(parameters) / name, Path(parameters) / (name + ".cpp")):
        if not read_only:
            safe_path(candidate)
        if candidate.is_file():
            return candidate
    raise ValueError(f"Study dictionary not found in {parameters}: {name}")


def preflight(args):
    safe_path(args.sim_dir)
    native_preflight()
    from tools import resolve_cfmesh_executable
    resolve_cfmesh_executable()
    # Load native acoustic libraries before long-running worker threads start.
    if not args.mesh_only:
        import matplotlib

        matplotlib.use("Agg")
        import postprocessing
    import docker

    client = docker.from_env()
    client.ping()
    client.images.get("microfluidica/openfoam:13")
    if args.mode not in (None, "AMI"):
        raise ValueError(
            "This native-cfMesh pipeline implements AMI/NCC rotating meshes."
        )
    if args.study:
        path = parameter_file(ROOT / "Parameters", args.study_file, read_only=True)
        query(path, args.study_parameter)
    for name in (
        "cfmeshDomainDict",
        "cfmeshRotorDict",
        "cfmeshStatorDict",
        "cfmeshRefinementDict",
    ):
        query(ROOT / "Parameters" / name)


def write(path, text):
    safe_path(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def header(name, kind="dictionary"):
    return f"FoamFile {{ version 2.0; format ascii; class {kind}; object {name}; }}\n"


def write_ftr(path, patches):
    # Each patch tuple has name, type, vertices, faces and desired orientation.
    vertices = []
    triangles = []
    vertex_ids = {}
    for patch_index, (name, kind, patch_vertices, patch_faces, reverse) in enumerate(
        patches
    ):
        point_indices = []
        for point in patch_vertices:
            point = tuple(float(x) for x in point)
            if point not in vertex_ids:
                vertex_ids[point] = len(vertices)
                vertices.append(point)
            point_indices.append(vertex_ids[point])
        for face in patch_faces:
            oriented_face = list(reversed(face)) if reverse else face
            triangles.append(
                ([point_indices[int(x)] for x in oriented_face], patch_index)
            )
    with Path(path).open("w") as output_file:
        output_file.write(f"{len(patches)}\n(\n")
        for name, kind, *_ in patches:
            output_file.write(f"{name} {kind}\n")
        output_file.write(f")\n{len(vertices)}\n(\n")
        for vertex in vertices:
            output_file.write(
                "(" + " ".join(format(float(x), ".17g") for x in vertex) + ")\n"
            )
        output_file.write(f")\n{len(triangles)}\n(\n")
        for oriented_face, patch in triangles:
            output_file.write(
                f"(({oriented_face[0]} {oriented_face[1]} {oriented_face[2]}) {patch})\n"
            )
        output_file.write(")\n")


def prepare_geometry(case, source, acoustic_surface, acoustic_diameter, study=None):
    import numpy as np
    import trimesh
    from cfmesh_test.prepare_case import read_stl

    case = safe_path(case)
    parameters_directory = case / "Parameters"
    if study:
        filename, entry, value = study
        study_path = parameter_file(parameters_directory, filename)
        native_run(
            ["foamDictionary", study_path, "-entry", entry, "-set", str(value)],
            cwd=case,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    domain_dictionary = parameters_directory / "cfmeshDomainDict"
    scale = float(query(domain_dictionary, "scale"))
    box_minimum = np.array(
        list(map(float, query(domain_dictionary, "boxMin").strip("()").split()))
    )
    box_maximum = np.array(
        list(map(float, query(domain_dictionary, "boxMax").strip("()").split()))
    )
    factor = float(query(domain_dictionary, "rotorRadiusFactor"))
    rotor_half_length = float(query(domain_dictionary, "rotorHalfLength"))
    segments = int(query(domain_dictionary, "cylinderSegments"))
    if (
        not all(math.isfinite(x) and x > 0 for x in (scale, factor, rotor_half_length))
        or segments < 32
    ):
        raise ValueError(
            "Invalid cfmeshDomainDict scale, cylinder dimensions, or segment count"
        )
    if (
        box_minimum.shape != (3,)
        or box_maximum.shape != (3,)
        or not np.all(box_maximum > box_minimum)
    ):
        raise ValueError("Invalid domain bounds")
    vertices, triangles, signed_volume, source_hash = read_stl(Path(source), scale)
    vertices, triangles = np.asarray(vertices), np.asarray(triangles)
    diameter = float(np.ptp(vertices, axis=0).max())
    rotor_radius = factor * diameter
    if not 0.02 < diameter < 2:
        raise ValueError("Check STL units: measured propeller span must be 0.02–2 m")
    # Ensure the whole swept propeller fits inside the polygonal cylinder.
    if np.hypot(vertices[:, 0], vertices[:, 2]).max() >= rotor_radius * math.cos(
        math.pi / segments
    ):
        raise ValueError("Propeller swept radius does not fit inside rotor cylinder")
    if np.abs(vertices[:, 1]).max() >= rotor_half_length:
        raise ValueError("Propeller does not fit axially inside rotor cylinder")
    if np.any(
        box_minimum >= [-rotor_radius, -rotor_half_length, -rotor_radius]
    ) or np.any(box_maximum <= [rotor_radius, rotor_half_length, rotor_radius]):
        raise ValueError("Rotor cylinder must be strictly inside the domain")
    rotor_cylinder = trimesh.creation.cylinder(
        radius=rotor_radius, height=2 * rotor_half_length, sections=segments
    )
    rotor_cylinder.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    )
    groups = (
        ("side", np.abs(rotor_cylinder.face_normals[:, 1]) < 0.5),
        ("top", rotor_cylinder.face_normals[:, 1] > 0.5),
        ("bottom", rotor_cylinder.face_normals[:, 1] < -0.5),
    )
    boxv = [
        tuple(box_maximum[i] if j & (1 << i) else box_minimum[i] for i in range(3))
        for j in range(8)
    ]
    quads = [
        (0, 4, 6, 2),
        (1, 3, 7, 5),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 2, 3, 1),
        (4, 5, 7, 6),
    ]
    boxnames = [
        "walls_xmin",
        "walls_xmax",
        "outlet",
        "inlet",
        "walls_zmin",
        "walls_zmax",
    ]
    rotor = [("propeller", "wall", vertices, triangles, signed_volume > 0)]
    stator = [
        (
            name,
            "patch",
            boxv,
            [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])],
            False,
        )
        for name, quad in zip(boxnames, quads)
    ]
    for name, mask in groups:
        rotor.append(
            (
                "rotaryRegion_" + name,
                "patch",
                rotor_cylinder.vertices,
                rotor_cylinder.faces[mask],
                False,
            )
        )
        stator.append(
            (
                "rotaryRegion_slave_" + name,
                "patch",
                rotor_cylinder.vertices,
                rotor_cylinder.faces[mask],
                True,
            )
        )
    from cfmesh_refinement import prepare_refinements

    refinements = prepare_refinements(
        case,
        diameter,
        rotor_radius,
        rotor_half_length,
        segments,
        acoustic_surface,
        acoustic_diameter,
    )
    for role, patches in (("rotor", rotor), ("stator", stator)):
        region_directory = case / "cfmesh" / role
        (region_directory / "constant/triSurface").mkdir(parents=True)
        (region_directory / "system").mkdir()
        (region_directory / "0").mkdir()
        shutil.copytree(parameters_directory, region_directory / "Parameters")
        write_ftr(region_directory / "constant/triSurface/domain.ftr", patches)
        write(
            region_directory / "system/meshDict",
            header("meshDict") + f'#include "../Parameters/cfmesh{role.title()}Dict"\n',
        )
        write(
            region_directory / "system/controlDict",
            header("controlDict") + """
application cartesianMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 12;
writeCompression off; runTimeModifiable false;
""",
        )
        write(
            region_directory / "system/fvSchemes",
            header("fvSchemes") + """
ddtSchemes { default steadyState; } gradSchemes { default Gauss linear; }
divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; } snGradSchemes { default corrected; }
""",
        )
        write(
            region_directory / "system/fvSolution",
            header("fvSolution") + "solvers {}\n",
        )
        interface = "rotaryRegion" if role == "rotor" else "rotaryRegion_slave"
        merge = [(interface, "patch", interface + "_.*")]
        if role == "stator":
            merge.append(("walls", "wall", "walls_.*"))
        patch_dictionary_text = (
            header("createPatchDict") + "pointSync false;\npatches\n(\n"
        )
        for name, kind, pattern in merge:
            patch_dictionary_text += f'{{ name {name}; patchInfo {{ type {kind}; }} constructFrom patches; patches ("{pattern}"); }}\n'
        write(
            region_directory / "system/createPatchDict", patch_dictionary_text + ");\n"
        )
    surface_directory = case / "constant/triSurface"
    trimesh.Trimesh(vertices=vertices, faces=triangles, process=False).export(
        surface_directory / "propeller.stl"
    )
    sphere_radius = None
    if acoustic_surface == "permeable":
        sphere_radius = 0.5 * float(acoustic_diameter) * diameter
        if sphere_radius <= np.linalg.norm(vertices, axis=1).max():
            raise ValueError("Acoustic sphere must enclose the complete propeller")
        if np.any(box_minimum >= -sphere_radius) or np.any(
            box_maximum <= sphere_radius
        ):
            raise ValueError("Acoustic sphere must be strictly inside the domain")
        trimesh.creation.icosphere(
            subdivisions=int(
                query(
                    parameters_directory / "cfmeshRefinementDict", "sphereSubdivisions"
                )
            ),
            radius=sphere_radius,
        ).export(surface_directory / "permeableSurface.stl")
    report = dict(
        backend=BACKEND,
        source=str(source),
        source_sha256=source_hash,
        scale=scale,
        diameter_m=diameter,
        propeller_volume_m3=abs(signed_volume),
        rotor_radius_m=rotor_radius,
        rotor_half_length_m=rotor_half_length,
        cylinder_segments=segments,
        axis=[0, 1, 0],
        origin=[0, 0, 0],
        box_min=box_minimum.tolist(),
        box_max=box_maximum.tolist(),
        expected_fluid_volume_m3=float(np.prod(box_maximum - box_minimum))
        - abs(signed_volume),
        expected_rotor_volume_m3=float(rotor_cylinder.volume) - abs(signed_volume),
        acoustic_sphere_radius_m=sphere_radius,
        refinement_regions=refinements,
    )
    write(case / "cfmesh/geometry.json", json.dumps(report, indent=2) + "\n")
    (case / "sim.foam").touch()


def native_mesh_run(case, role, command, cores, callback, live):
    from tools import report_case_stage

    region_directory = case / "cfmesh" / role
    stage = command[0]
    report_case_stage(
        callback, f"cfMesh.{role}.{stage}", f"{role}: {shlex.join(command)}"
    )
    with (region_directory / f"log.{stage}").open("w") as log, (
        case / "log.cfmesh"
    ).open("a") as all_log:
        all_log.write(f"\n[{role}] + {shlex.join(command)}\n")
        process = subprocess.Popen(
            native_command(command, cwd=region_directory, cores=cores),
            cwd=region_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            all_log.write(line)
            all_log.flush()
            if live:
                with OUTPUT_LOCK:
                    print(f"[{case.name}/{role}] {line}", end="", flush=True)
        process.stdout.close()
        if process.wait():
            raise RuntimeError(f"{role} {stage} failed: {region_directory}/log.{stage}")


def docker_run(container, case, command, log_name, callback, live, relative="."):
    from tools import safe_exec, report_case_stage

    prefix = f"cd {shlex.quote('/simulation/'+relative)} && "
    cmd = "source /opt/openfoam13/etc/bashrc && set -o pipefail && " + prefix
    cmd += shlex.join(command) + " 2>&1 | tee " + shlex.quote("/simulation/" + log_name)
    report_case_stage(callback, command[0], shlex.join(command))
    if not safe_exec(
        container,
        ["bash", "-c", cmd],
        command[0],
        print_output=live,
        status_callback=callback,
    ):
        raise RuntimeError(f"{command[0]} failed; see {case/log_name}")


def strip_header(path):
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", Path(path).read_text(), flags=re.S)
    return text.split("}", 1)[1]


def patch_info(case):
    text = (case / "constant/polyMesh/boundary").read_text()
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    patches = {}
    for patch_name, patch_dictionary in re.findall(r"(\w+)\s*\{([^{}]*)\}", text):
        face_count_match = re.search(r"nFaces\s+(\d+)", patch_dictionary)
        if face_count_match:
            patches[patch_name] = dict(
                nFaces=int(face_count_match[1]),
                type=re.search(r"type\s+(\w+)", patch_dictionary)[1],
            )
    return patches


def run_mesh(container, case, cores, layers, allow_bad, callback=None, live=False):
    case = safe_path(case)
    for role in ("rotor", "stator"):
        region_directory = case / "cfmesh" / role
        mesh_dictionary = region_directory / "system/meshDict"
        # Expand the user's complete dictionary, preserving native cfMesh controls.
        effective = query(mesh_dictionary)
        write(region_directory / "system/meshDict", effective + "\n")
        if layers == "none" or (layers == "cfmesh" and role == "stator"):
            native_run(
                [
                    "foamDictionary",
                    mesh_dictionary,
                    "-entry",
                    "workflowControls",
                    "-set",
                    "{ stopAfter edgeExtraction; }",
                ],
                cwd=case,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        elif (
            layers == "cfmesh"
            and role == "rotor"
            and optional(mesh_dictionary, "workflowControls") is not None
        ):
            native_run(
                [
                    "foamDictionary",
                    mesh_dictionary,
                    "-entry",
                    "workflowControls",
                    "-remove",
                ],
                cwd=case,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        stop = optional(mesh_dictionary, "workflowControls/stopAfter")
        if stop not in (None, "edgeExtraction"):
            raise ValueError(
                "Only a complete cfMesh run or stopAfter edgeExtraction is supported"
            )
        native_mesh_run(case, role, ["cartesianMesh"], cores, callback, live)
        if stop:
            log_text = (region_directory / "log.cartesianMesh").read_text()
            if "Stopping after step edgeExtraction" not in log_text:
                raise RuntimeError("Mesher did not honour the required no-layer stop")
            native_mesh_run(
                case,
                role,
                [
                    "improveMeshQuality",
                    "-nLoops",
                    "2",
                    "-nIterations",
                    "20",
                    "-nSurfaceIterations",
                    "0",
                ],
                cores,
                callback,
                live,
            )
        docker_run(
            container,
            case,
            ["createPatch", "-overwrite"],
            f"log.createPatch.{role}",
            callback,
            live,
            "cfmesh/" + role,
        )
    rotor = case / "cfmesh/rotor"
    # Every rotor cell belongs to the moving zone. mergeMeshes preserves zone membership.
    owner_text = strip_header(rotor / "constant/polyMesh/owner")
    owners = list(
        map(int, owner_text[owner_text.index("(") + 1 : owner_text.rindex(")")].split())
    )
    neighbour_text = strip_header(rotor / "constant/polyMesh/neighbour")
    neighbours = list(
        map(
            int,
            neighbour_text[
                neighbour_text.index("(") + 1 : neighbour_text.rindex(")")
            ].split(),
        )
    )
    rotor_cell_count = max(owners + neighbours) + 1
    zone_dictionary = (
        header("cellZones", "cellZoneList") + "1\n(\nrotaryRegion\n{\ntype cellZone;\n"
    )
    zone_dictionary += (
        f"cellLabels List<label> {rotor_cell_count}\n(\n"
        + "\n".join(map(str, range(rotor_cell_count)))
        + "\n);\n}\n)\n"
    )
    write(rotor / "constant/polyMesh/cellZones", zone_dictionary)
    mesh_directory = case / "constant/polyMesh"
    if mesh_directory.exists():
        raise FileExistsError(
            "Root mesh already exists. Start a fresh order or resume a failed case."
        )
    shutil.copytree(case / "cfmesh/stator/constant/polyMesh", mesh_directory)
    docker_run(
        container,
        case,
        ["mergeMeshes", "-addCases", '("/simulation/cfmesh/rotor")'],
        "log.mergeMeshes",
        callback,
        live,
    )
    expected = {
        "inlet",
        "outlet",
        "walls",
        "propeller",
        "rotaryRegion",
        "rotaryRegion_slave",
    }
    patches = patch_info(case)
    if {k for k, v in patches.items() if v["nFaces"]} != expected:
        raise ValueError(f"Unexpected assembled patches: {patches}")
    if "rotaryRegion" not in (mesh_directory / "cellZones").read_text():
        raise ValueError("Merged mesh lost rotaryRegion cellZone")
    from cfmesh_interface import project_rotating_interface

    from tools import report_case_stage

    report_case_stage(callback, "interface", "projecting the shared rotating cylinder")
    project_rotating_interface(case)
    docker_run(
        container,
        case,
        ["checkMesh", "-allGeometry", "-allTopology"],
        "log.checkMesh",
        callback,
        live,
    )
    log_text = (case / "log.checkMesh").read_text()
    if not re.search(r"Number of regions:\s*2\b", log_text):
        raise ValueError("Expected two disconnected fluid regions before NCC coupling")
    number_pattern = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    measured = re.search(r"Total volume\s*=\s*" + number_pattern, log_text)
    expected_volume = json.loads((case / "cfmesh/geometry.json").read_text())[
        "expected_fluid_volume_m3"
    ]
    if (
        not measured
        or abs(float(measured[1]) - expected_volume) / expected_volume > 0.005
    ):
        raise ValueError(
            "Assembled fluid volume differs from the intended domain by more than 0.5%"
        )
    quality_ok = "Mesh OK." in log_text and not re.search(
        r"Failed\s+\d+\s+mesh checks", log_text
    )
    write(
        case / "cfmesh/mesh-status.json",
        json.dumps(
            dict(
                mesh_quality_ok=quality_ok,
                allow_bad_mesh=allow_bad,
                rotor_cells=rotor_cell_count,
                patches=patches,
            ),
            indent=2,
        )
        + "\n",
    )
    if not quality_ok and not allow_bad:
        raise ValueError(
            "Mesh quality checks failed; inspect log.checkMesh. No solver was started."
        )
    # Add NCC boundary entries to all initial fields before decomposition.
    docker_run(
        container,
        case,
        ["createNonConformalCouples", "-fields", "rotaryRegion_slave", "rotaryRegion"],
        "log.createNonConformalCouples",
        callback,
        live,
    )
    docker_run(container, case, ["checkMesh"], "log.checkMesh.NCC", callback, live)
    coupling_log = (case / "log.createNonConformalCouples").read_text()
    coverage = re.findall(
        r"(?:Source|Target) min/average/max coverage = ([0-9.eE+-]+)/([0-9.eE+-]+)/([0-9.eE+-]+)",
        coupling_log,
    )
    coverage_ok = len(coverage) == 2 and all(
        float(minimum_coverage) >= 0.95 and float(average_coverage) >= 0.999
        for minimum_coverage, average_coverage, maximum_coverage in coverage
    )
    write(
        case / "cfmesh/ncc-status.json",
        json.dumps(
            dict(
                coverage=coverage,
                coverage_ok=coverage_ok,
                minimum_face_coverage=0.95,
                minimum_average_coverage=0.999,
            ),
            indent=2,
        )
        + "\n",
    )
    if not coverage_ok:
        raise ValueError(
            "NCC interface coverage is inadequate; inspect cfmesh/ncc-status.json and log.createNonConformalCouples"
        )
    coupled_patches = patch_info(case)
    if sum(p["type"] == "nonConformalCyclic" for p in coupled_patches.values()) != 2:
        raise ValueError(
            "Expected exactly two nonConformalCyclic patches after coupling"
        )
    ncc_check_log = (case / "log.checkMesh.NCC").read_text()
    ncc_ok = "Mesh OK." in ncc_check_log and not re.search(
        r"Failed\s+\d+\s+mesh checks", ncc_check_log
    )
    status_file = case / "cfmesh/mesh-status.json"
    status = json.loads(status_file.read_text())
    status["ncc_mesh_quality_ok"] = ncc_ok
    status["mesh_quality_ok"] = quality_ok and ncc_ok
    write(status_file, json.dumps(status, indent=2) + "\n")
    if not ncc_ok and not allow_bad:
        raise ValueError("Post-NCC mesh checks failed; inspect log.checkMesh.NCC")
    (case / "sim.foam").touch()
    return quality_ok and ncc_ok
