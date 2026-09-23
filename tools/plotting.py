"""File-only Matplotlib backend shared by pipeline reporting and acoustics.

Select Agg before importing pyplot, including standalone report entry points.
The scheduler still serializes plotting with MATPLOTLIB_LOCK: a non-GUI backend
does not make Matplotlib's global state safe for concurrent access.
"""
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot  # noqa: E402
