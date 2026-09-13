"""Reusable order directories, with output isolation and non-destructive history."""

from contextlib import contextmanager
import shutil
from cfmesh_pipeline import ROOT, safe_path


@contextmanager
def order_lock(directory):
    import fcntl

    directory = safe_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = safe_path(directory / ".pipeline.lock")
    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                f"A pipeline is already running in {directory}. Wait for it to finish."
            ) from None
        try:
            yield directory
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def seed_current_stl(directory):
    """An empty order uses the propeller explicitly selected for this duplicate."""
    directory = safe_path(directory)
    stl_directory = safe_path(directory / "STL")
    if stl_directory.is_dir() and any(
        p.suffix.lower() == ".stl" for p in stl_directory.iterdir()
    ):
        return
    source = ROOT / "cfmesh_test/input/fused.stl"
    if not source.is_file():
        raise ValueError(
            f"Place a propeller STL in {stl_directory}; the duplicate's selected input is missing."
        )
    stl_directory.mkdir(parents=True, exist_ok=True)
    destination_path = safe_path(stl_directory / "10x7E.stl")
    if destination_path.exists():
        raise ValueError(f"Input path is not a usable STL file: {destination_path}")
    shutil.copy2(source, destination_path)
    print(f"Using the selected fused propeller: {destination_path}")


def require_new_order(directory):
    """An existing directory is reusable; an existing order needs --resume."""
    order_path = safe_path(directory / "simulation_order.json")
    if order_path.exists():
        raise FileExistsError(
            f"A simulation order already exists in {directory}. "
            "Use --resume to continue it. To start a new order here, "
            "move simulation_order.json out of this directory first; "
            "existing case folders and STL inputs will be preserved."
        )
