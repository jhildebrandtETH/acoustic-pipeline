"""Select complete OpenFOAM templates with explicit wall treatment."""
from pathlib import Path


def template_directory(mode, turbulence, wall_functions, *, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    if mode not in ("AMI", "MRF"):
        raise ValueError(f"Unsupported mode: {mode}")
    if turbulence not in ("kOmegaSST", "kEpsilon", "DES"):
        raise ValueError(f"Unsupported turbulence: {turbulence}")
    if turbulence == "DES" and (mode != "AMI" or wall_functions == "yes"):
        raise ValueError("DES requires --mode AMI --wall-functions no")
    if wall_functions == "legacy" and mode == "AMI":
        name = ("Core Template DES - kOmegaSST" if turbulence == "DES"
                else f"Core Template AMI - {turbulence}")
        return root / "CoreTemplates" / "legacy" / name
    if wall_functions not in ("yes", "no"):
        raise ValueError("Specify --wall-functions yes or no")
    treatment = "wall-functions" if wall_functions == "yes" else "wall-resolved"
    return root / "CoreTemplates" / mode / turbulence / treatment
