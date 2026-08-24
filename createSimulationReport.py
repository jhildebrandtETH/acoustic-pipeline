from pathlib import Path
from datetime import datetime
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
import math

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from tools import read_openfoam_timestep_and_courant_statistics


def create_simulation_report(case_path, rpm, mode, turbulence_model, output_pdf=None):

    def create_yplus_distribution_plot(case_path, report_dir, patch_name="cubeWall"):

        def get_latest_time_dir(case_path):
            time_dirs = []

            for item in case_path.iterdir():
                if item.is_dir():
                    try:
                        time_dirs.append((float(item.name), item))
                    except ValueError:
                        pass

            if not time_dirs:
                return None

            return max(time_dirs, key=lambda x: x[0])[1]

        latest_time_dir = get_latest_time_dir(case_path)

        if latest_time_dir is None:
            return None, None

        yplus_file = latest_time_dir / "yPlus"

        if not yplus_file.exists():
            return None, None

        text = yplus_file.read_text(encoding="utf-8", errors="ignore")

        patch_pattern = rf"{re.escape(patch_name)}\s*\{{(.*?)\}}"
        patch_match = re.search(patch_pattern, text, re.DOTALL)

        if not patch_match:
            return None, None

        patch_block = patch_match.group(1)

        list_pattern = r"nonuniform\s+List<scalar>\s*(\d+)\s*\((.*?)\)"
        list_match = re.search(list_pattern, patch_block, re.DOTALL)

        if not list_match:
            return None, None

        values_block = list_match.group(2)

        yplus_values = np.array(
            [
                float(v)
                for v in re.findall(
                    r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                    values_block,
                )
            ],
            dtype=float,
        )

        if len(yplus_values) == 0:
            return None, None

        total = len(yplus_values)

        # Compact wall-function quality classes
        class_counts = [
            int(np.sum(yplus_values < 5)),
            int(np.sum((yplus_values >= 5) & (yplus_values <= 30))),
            int(np.sum(yplus_values > 30)),
        ]
        class_percentages = [100.0 * c / total for c in class_counts]

        # Finer block diagram to make the high-y+ region visible
        block_bins = [0.0, 5.0, 30.0, 50.0, 100.0, 200.0, np.inf]
        block_labels = ["<5", "5-30", "30-50", "50-100", "100-200", ">200"]
        block_counts = []

        for lower, upper in zip(block_bins[:-1], block_bins[1:]):
            if np.isinf(upper):
                count = np.sum(yplus_values >= lower)
            elif lower == 0.0:
                count = np.sum(yplus_values < upper)
            else:
                count = np.sum((yplus_values >= lower) & (yplus_values < upper))
            block_counts.append(int(count))

        block_percentages = [100.0 * c / total for c in block_counts]

        yplus_stats = {
            "patch_name": patch_name,
            "time_dir": latest_time_dir.name,
            "n_faces": int(total),
            "average_yplus": float(np.mean(yplus_values)),
            "min_yplus": float(np.min(yplus_values)),
            "max_yplus": float(np.max(yplus_values)),
            "median_yplus": float(np.median(yplus_values)),
            "share_yplus_lt_5_percent": class_percentages[0],
            "share_yplus_5_to_30_percent": class_percentages[1],
            "share_yplus_gt_30_percent": class_percentages[2],
        }

        yplus_plot = report_dir / "yplus_distribution.png"

        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        bars = ax.bar(block_labels, block_percentages, zorder=3)

        ax.set_ylabel("Surface face share [%]")
        ax.set_xlabel("y+ interval")
        ax.set_title(
            f"y+ Distribution of Propeller Surface "
            f"(avg. y+ = {yplus_stats['average_yplus']:.1f})"
        )

        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

        # Extra vertical space prevents labels from touching the top frame or gridlines.
        max_percentage = max(block_percentages)
        ax.set_ylim(0, max_percentage * 1.18 + 3)

        # Labels are shifted above each bar and placed on a white background,
        # so the grid does not reduce readability.
        for bar, percentage, count in zip(bars, block_percentages, block_counts):
            label_y = bar.get_height() + 1.5

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{percentage:.1f}%\n({count})",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.9,
                    pad=1.5,
                ),
                clip_on=False,
                zorder=5,
            )

        note = (
            f"Classes: <5 = {class_percentages[0]:.1f}%, "
            f"5-30 = {class_percentages[1]:.1f}%, "
            f">30 = {class_percentages[2]:.1f}%"
        )
        fig.text(0.5, 0.015, note, ha="center", fontsize=9)

        fig.tight_layout(rect=(0, 0.07, 1, 0.96))
        fig.savefig(yplus_plot, dpi=200)
        plt.close(fig)

        return yplus_plot, yplus_stats

    def read_mesh_element_types(case_path):
        log_checkmesh = case_path / "log.checkMesh"

        element_types = {
            "hexahedra": 0,
            "prisms": 0,
            "wedges": 0,
            "pyramids": 0,
            "tet wedges": 0,
            "tetrahedra": 0,
            "polyhedra": 0,
        }

        if not log_checkmesh.exists():
            return element_types

        text = log_checkmesh.read_text(encoding="utf-8", errors="ignore")

        patterns = {
            "hexahedra": r"hexahedra:\s*([0-9]+)",
            "prisms": r"prisms:\s*([0-9]+)",
            "wedges": r"wedges:\s*([0-9]+)",
            "pyramids": r"pyramids:\s*([0-9]+)",
            "tet wedges": r"tet wedges:\s*([0-9]+)",
            "tetrahedra": r"tetrahedra:\s*([0-9]+)",
            "polyhedra": r"polyhedra:\s*([0-9]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                element_types[key] = int(match.group(1))

        return element_types
    
    def create_mesh_element_plot(element_types, report_dir):
        nonzero = {
            key: value
            for key, value in element_types.items()
            if value > 0
        }

        if not nonzero:
            return None

        total = sum(nonzero.values())

        labels = list(nonzero.keys())
        values = [100.0 * value / total for value in nonzero.values()]

        mesh_plot = report_dir / "mesh_element_types.png"

        plt.figure(figsize=(5.0, 3.4))
        bars = plt.bar(labels, values)

        plt.ylabel("Cell share [%]")
        plt.title("Mesh Element Types")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y")

        # --- Add percentage labels on bars ---
        for bar, val in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(mesh_plot, dpi=200)
        plt.close()

        return mesh_plot

    def read_mesh_information(case_path):
        log_checkmesh = case_path / "log.checkMesh"

        mesh_info = {
            "mesh_ok": False,
            "cells": None,
            "faces": None,
            "points": None,
            "boundary_patches": None,
            "max_aspect_ratio": None,
            "max_skewness": None,
            "max_non_orthogonality": None,
        }

        if not log_checkmesh.exists():
            mesh_info["status"] = "log.checkMesh not found"
            return mesh_info

        text = log_checkmesh.read_text(encoding="utf-8", errors="ignore")

        mesh_info["mesh_ok"] = "Mesh OK" in text
        mesh_info["status"] = "Mesh OK" if mesh_info["mesh_ok"] else "Mesh check failed / not confirmed"

        patterns = {
            "points": r"points:\s*([0-9]+)",
            "faces": r"faces:\s*([0-9]+)",
            "cells": r"cells:\s*([0-9]+)",
            "boundary_patches": r"boundary patches:\s*([0-9]+)",
            "max_aspect_ratio": r"Max aspect ratio\s*=\s*([0-9.eE+-]+)",
            "max_skewness": r"Max skewness\s*=\s*([0-9.eE+-]+)",
            "max_non_orthogonality": r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                value = match.group(1)
                try:
                    mesh_info[key] = float(value) if "." in value or "e" in value.lower() else int(value)
                except ValueError:
                    mesh_info[key] = value

        return mesh_info

    def format_seconds(seconds):
        if seconds is None:
            return "Not found"

        seconds = int(round(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60

        if h > 0:
            return f"{h} h {m} min {s} s"
        if m > 0:
            return f"{m} min {s} s"
        return f"{s} s"



    def format_optional_number(value, format_spec=".6e"):
        if value is None:
            return "Not found"

        try:
            return format(float(value), format_spec)
        except (TypeError, ValueError):
            return str(value)

    def format_optional_bool(value):
        if value is True:
            return "yes"
        if value is False:
            return "no"
        return "Not found"

    def evaluate_thrust_convergence(times, thrusts, rev_time, threshold=1e-3):
        latest_time = float(times[-1])
        last_rev_start = latest_time - rev_time
        idx_start = np.searchsorted(times, last_rev_start, side="left")

        window_times = times[idx_start:]
        window_thrusts = thrusts[idx_start:]

        if len(window_thrusts) == 0:
            return {
                "passed": False,
                "reason": "No thrust samples found in final revolution window.",
                "window_start_s": last_rev_start,
                "window_end_s": latest_time,
                "mean_N": None,
                "std_N": None,
                "relative_std": None,
                "threshold": threshold,
                "n_samples": 0,
            }

        mean_thrust = float(np.mean(window_thrusts))
        std_thrust = float(np.std(window_thrusts, ddof=0))
        relative_std = std_thrust / max(abs(mean_thrust), 1e-12)

        return {
            "passed": bool(relative_std < threshold),
            "reason": None,
            "window_start_s": float(window_times[0]),
            "window_end_s": latest_time,
            "mean_N": mean_thrust,
            "std_N": std_thrust,
            "relative_std": float(relative_std),
            "threshold": float(threshold),
            "n_samples": int(len(window_thrusts)),
        }
    
    def evaluate_moments(times, moments, rev_time):
        latest_time = float(times[-1])
        last_rev_start = latest_time - rev_time
        idx_start = np.searchsorted(times, last_rev_start, side="left")

        window_times = times[idx_start:]
        window_moments = moments[idx_start:]

        if len(window_moments) == 0:
            return {
                "passed": False,
                "reason": "No moments samples found in final revolution window.",
                "window_start_s": last_rev_start,
                "window_end_s": latest_time,
                "mean_N": None,
                "std_N": None,
                "relative_std": None
            }

        mean_moment = float(np.mean(window_moments))
        std_moment = float(np.std(window_moments, ddof=0))
        relative_std = std_moment / max(abs(mean_moment), 1e-12)

        return {
            "passed": bool,
            "reason": None,
            "window_start_s": float(window_times[0]),
            "window_end_s": latest_time,
            "mean_N": mean_moment,
            "std_N": std_moment,
            "relative_std": float(relative_std)
        }
        

    def compute_thrust_stability_history(times, thrusts, rev_time):
        """
        Computes the relative thrust fluctuation over a sliding one-revolution window.

        metric(t) = std(F_window) / |mean(F_window)|

        The value at time t uses all force samples within [t - T_rev, t].
        """
        metric = np.full(len(times), np.nan, dtype=float)
        window_mean = np.full(len(times), np.nan, dtype=float)
        window_std = np.full(len(times), np.nan, dtype=float)
        sample_count = np.zeros(len(times), dtype=int)

        for i, time_value in enumerate(times):
            window_start = time_value - rev_time

            if window_start < times[0]:
                continue

            j = np.searchsorted(times, window_start, side="left")
            window_values = thrusts[j : i + 1]

            if len(window_values) < 2:
                continue

            mean_value = float(np.mean(window_values))
            std_value = float(np.std(window_values, ddof=0))

            window_mean[i] = mean_value
            window_std[i] = std_value
            metric[i] = std_value / max(abs(mean_value), 1e-12)
            sample_count[i] = int(len(window_values))

        return {
            "time": times,
            "relative_std": metric,
            "mean_N": window_mean,
            "std_N": window_std,
            "sample_count": sample_count,
        }

    def create_force_plots(times, thrusts, report_dir, rev_time, thrust_convergence):
        force_plot = report_dir / "force_plot.png"
        conv_plot = report_dir / "force_convergence.png"

        latest_time = float(times[-1])
        last_rev_start = latest_time - rev_time

        # -----------------------------
        # Force history plot
        # -----------------------------
        # The raw force plot is kept as a general overview. Extreme initialization
        # spikes are excluded only from the axis scaling, not from the data itself.
        plot_mask = times > 0.001
        plot_thrusts = thrusts[plot_mask]

        if len(plot_thrusts) > 0:
            y_min = np.percentile(plot_thrusts, 1)
            y_max = np.percentile(plot_thrusts, 99)
            y_margin = 0.15 * max(y_max - y_min, 1e-12)
        else:
            y_min, y_max = np.min(thrusts), np.max(thrusts)
            y_margin = 0.15 * max(y_max - y_min, 1e-12)

        plt.figure(figsize=(12, 5))
        plt.plot(times, thrusts, label="Pressure force Fz")

        plt.axvspan(
            last_rev_start,
            latest_time,
            alpha=0.2,
            label="final revolution window",
        )

        plt.ylim(y_min - y_margin, y_max + y_margin)
        plt.xlabel("Time [s]")
        plt.ylabel("Force Fz [N]")
        plt.title("Pressure Force Fz")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(force_plot, dpi=200)
        plt.close()

        # -----------------------------
        # Thrust stability metric plot
        # -----------------------------
        # This plot directly visualizes the implemented convergence criterion:
        # std(F) / |mean(F)| evaluated over a sliding one-revolution window.
        threshold = thrust_convergence["threshold"]
        status = "PASSED" if thrust_convergence["passed"] else "FAILED"
        final_relative_std = thrust_convergence["relative_std"]

        stability_history = compute_thrust_stability_history(times, thrusts, rev_time)
        metric_time = stability_history["time"]
        metric = stability_history["relative_std"]
        valid = np.isfinite(metric) & (metric > 0.0)

        plt.figure(figsize=(12, 5))

        if np.any(valid):
            plt.plot(
                metric_time[valid],
                metric[valid],
                label=r"sliding 1-rev $\sigma_F / |\overline{F}|$",
            )

        plt.axhline(
            threshold,
            linestyle="--",
            label=f"criterion = {threshold:g}",
        )

        plt.axvspan(
            last_rev_start,
            latest_time,
            alpha=0.2,
            label="final evaluation window",
        )

        if final_relative_std is not None:
            text = (
                f"Final 1-rev result: {final_relative_std:.3e} → {status}\n"
                f"Criterion: relative thrust fluctuation < {threshold:g}"
            )
        else:
            text = f"Final 1-rev result could not be evaluated → {status}"

        plt.text(
            0.02,
            0.95,
            text,
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.85),
        )

        plt.yscale("log")
        plt.xlabel("Time [s]")
        plt.ylabel(r"Relative thrust fluctuation $\sigma_F / |\overline{F}|$")
        plt.title("Thrust Stability Criterion over Sliding One-Revolution Window")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(conv_plot, dpi=200)
        plt.close()

        return force_plot, conv_plot, stability_history
    
    def create_moments_plots(times, moments, report_dir, rev_time):
        moments_plot = report_dir / "moments_plot.png"

        latest_time = float(times[-1])
        last_rev_start = latest_time - rev_time

        # -----------------------------
        # Moments history plot
        # -----------------------------
        
        plot_mask = times > 0.001
        plot_moments = moments[plot_mask]

        if len(plot_moments) > 0:
            y_min = np.percentile(plot_moments, 1)
            y_max = np.percentile(plot_moments, 99)
            y_margin = 0.15 * max(y_max - y_min, 1e-12)
        else:
            y_min, y_max = np.min(moments), np.max(moments)
            y_margin = 0.15 * max(y_max - y_min, 1e-12)

        plt.figure(figsize=(12, 5))
        plt.plot(times, moments, label="Moments (pressure & viscous) M_y")

        plt.axvspan(
            last_rev_start,
            latest_time,
            alpha=0.2,
            label="final revolution window",
        )

        plt.ylim(y_min - y_margin, y_max + y_margin)
        plt.xlabel("Time [s]")
        plt.ylabel("Moment My [Nm]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(moments_plot, dpi=200)
        plt.close()

        return moments_plot

    def read_residual_dataframe(residual_file):
        if not residual_file.exists():
            return None

        with open(residual_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if len(lines) < 3:
            return None

        header = lines[1].lstrip("#").split()

        df = pd.read_csv(
            residual_file,
            sep=r"\s+",
            names=header,
            skiprows=2,
            engine="python",
        )

        if "Time" not in df.columns:
            return None

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["Time"])
        df = df.sort_values("Time")

        return df

    def evaluate_residual_slopes(df, rev_time, latest_time):
        if df is None or len(df) == 0:
            return None

        last_rev_start = latest_time - rev_time
        window = df[(df["Time"] >= last_rev_start) & (df["Time"] <= latest_time)].copy()

        if len(window) < 2:
            return {
                "window_start_s": last_rev_start,
                "window_end_s": latest_time,
                "n_samples": int(len(window)),
                "slopes_per_rev": {},
                "end_residuals": {},
                "mean_residuals": {},
                "reason": "Not enough residual samples in final revolution window.",
            }

        # Independent variable in revolutions relative to the start of the final window.
        x_rev = (window["Time"].to_numpy(dtype=float) - last_rev_start) / rev_time

        slopes_per_rev = {}
        end_residuals = {}
        mean_residuals = {}

        for col in window.columns:
            if col == "Time":
                continue

            values = window[col].to_numpy(dtype=float)
            valid = np.isfinite(values) & (values > 0.0) & np.isfinite(x_rev)

            if np.count_nonzero(valid) < 2:
                slopes_per_rev[col] = None
                end_residuals[col] = None
                mean_residuals[col] = None
                continue

            y_log = np.log10(values[valid])
            x_valid = x_rev[valid]

            # Slope of log10(residual) per propeller revolution.
            slope, _intercept = np.polyfit(x_valid, y_log, 1)

            slopes_per_rev[col] = float(slope)
            end_residuals[col] = float(values[valid][-1])
            mean_residuals[col] = float(np.mean(values[valid]))

        return {
            "window_start_s": float(window["Time"].iloc[0]),
            "window_end_s": float(window["Time"].iloc[-1]),
            "n_samples": int(len(window)),
            "slopes_per_rev": slopes_per_rev,
            "end_residuals": end_residuals,
            "mean_residuals": mean_residuals,
            "reason": None,
        }

    def create_residual_plots(residual_file, report_dir, rev_time, latest_time):
        residual_plot = report_dir / "residuals.png"

        df = read_residual_dataframe(residual_file)

        if df is None:
            return None, None

        last_rev_start = latest_time - rev_time

        plt.figure(figsize=(12, 5))

        for col in df.columns:
            if col != "Time":
                plt.plot(df["Time"], df[col], label=col)

        plt.axvspan(
            last_rev_start,
            latest_time,
            alpha=0.2,
            label="final revolution window",
        )

        plt.yscale("log")
        plt.xlabel("Time [s]")
        plt.ylabel("Residual")
        plt.title("Residual Convergence")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(residual_plot, dpi=200)
        plt.close()

        residual_slope_info = evaluate_residual_slopes(df, rev_time, latest_time)

        return residual_plot, residual_slope_info

    case_path = Path(case_path)
    report_dir = case_path / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    force_file = case_path / "postProcessing" / "forcesBlades" / "merged_forces.dat"
    residual_file = case_path / "postProcessing" / "residuals" / "merged_residuals.dat"
    log_file = case_path / "log.pimpleFoam"
    control_dict_file = case_path / "system" / "controlDict.cpp"

    timestep_courant_info = read_openfoam_timestep_and_courant_statistics(
        log_file,
        control_dict_file,
    )

    mesh_info = read_mesh_information(case_path)
    mesh_element_types = read_mesh_element_types(case_path)
    mesh_element_plot = create_mesh_element_plot(mesh_element_types, report_dir)

    yplus_plot, yplus_stats = create_yplus_distribution_plot(
        case_path,
        report_dir,
        patch_name="cubeWall",
    )


    if output_pdf is None:
        output_pdf = report_dir / "simulation_report.pdf"
    else:
        output_pdf = Path(output_pdf)

    times, thrusts, moments = [], [], []

    with open(force_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.replace("(", " ").replace(")", " ").split()

            try:
                times.append(float(parts[0]))
                thrusts.append(float(parts[2]) + float(parts[5]))
                moments.append(float(parts[8]) + float(parts[11]))
            except (ValueError, IndexError):
                continue

    if not times:
        raise ValueError("No valid force data found.")

    times = np.asarray(times, dtype=float)
    thrusts = np.asarray(thrusts, dtype=float)
    moments = np.asarray(moments, dtype=float)

    idx = np.argsort(times)
    times = times[idx]
    thrusts = thrusts[idx]
    moments = moments[idx]

    latest_time = float(times[-1])
    rev_time = 60.0 / float(rpm)
    eff_revs = latest_time / rev_time

    if latest_time < rev_time:
        print("WARNING: Simulation shorter than one revolution.")

    thrust_convergence = evaluate_thrust_convergence(
        times,
        thrusts,
        rev_time,
        threshold=1e-3,
    )
    thrust_avg = thrust_convergence["mean_N"]


    exec_time = None
    clock_time = None

    if log_file.exists():
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.search(
                    r"ExecutionTime\s*=\s*([0-9.eE+-]+)\s*s\s+ClockTime\s*=\s*([0-9.eE+-]+)\s*s",
                    line,
                )
                if m:
                    exec_time = float(m.group(1))
                    clock_time = float(m.group(2))

    runtime_text = (
        f"{format_seconds(exec_time)} CPU | {format_seconds(clock_time)} wall"
        if exec_time is not None
        else "Not found"
    )

    force_plot, conv_plot, thrust_stability_history = create_force_plots(
        times, thrusts, report_dir, rev_time, thrust_convergence
    )

    moments_plot = create_moments_plots(times, moments, report_dir, rev_time)

    residual_plot, residual_slope_info = create_residual_plots(
        residual_file,
        report_dir,
        rev_time,
        latest_time,
    )
    c = canvas.Canvas(str(output_pdf), pagesize=A4)
    w, h = A4

    y = h - 60

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "CFD Simulation Report")

    y -= 35
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Case Information")

    c.setFont("Helvetica", 11)

    y -= 22
    c.drawString(50, y, f"Case: {case_path.name}")

    y -= 22
    c.drawString(50, y, f"Mode: {mode}")

    y -= 22
    c.drawString(50, y, f"Turbulence Model: {turbulence_model}")

    y -= 22
    c.drawString(50, y, f"RPM: {rpm}")

    y -= 22
    c.drawString(50, y, f"One revolution time: {rev_time:.6f} s")

    y -= 22
    c.drawString(50, y, f"Simulated time: {latest_time:.6f} s")

    y -= 22
    c.drawString(50, y, f"Effective simulated revolutions: {eff_revs:.2f}")

    y -= 22
    c.drawString(50, y, f"Runtime (rank0): {runtime_text}")

    delta_t_summary = timestep_courant_info["delta_t"]
    flow_co_summary = timestep_courant_info["flow_courant"]

    y -= 22
    c.drawString(
        50,
        y,
        "Average solver delta-t: "
        f"{format_optional_number(delta_t_summary['average_s'])} s",
    )

    y -= 22
    c.drawString(
        50,
        y,
        "Peak flow Courant number: "
        f"{format_optional_number(
            flow_co_summary['peak_max_co']
            if flow_co_summary is not None
            else None,
            '.4g',
        )}",
    )

    y -= 40

    # -----------------------------
    # Mesh Information
    # -----------------------------
    y -= 40
    mesh_section_y = y

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Mesh Information")

    c.setFont("Helvetica", 11)

    y -= 22
    c.drawString(50, y, f"Mesh status: {mesh_info['status']}")

    y -= 22
    c.drawString(50, y, f"Cells: {mesh_info['cells']}")

    y -= 22
    c.drawString(50, y, f"Faces: {mesh_info['faces']}")

    y -= 22
    c.drawString(50, y, f"Points: {mesh_info['points']}")

    y -= 22
    c.drawString(50, y, f"Max aspect ratio: {mesh_info['max_aspect_ratio']}")

    y -= 22
    c.drawString(50, y, f"Max skewness: {mesh_info['max_skewness']}")

    y -= 22
    c.drawString(50, y, f"Max non-orthogonality: {mesh_info['max_non_orthogonality']}")

    if mesh_element_plot is not None:
        c.drawImage(
            str(mesh_element_plot),
            320,
            mesh_section_y - 190,
            width=220,
            height=170,
            preserveAspectRatio=True,
            mask="auto",
        )
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Main Result")

    y -= 24
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Last 1-rev averaged thrust: {thrust_avg:.6f} N")

    y -= 24
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Last 1-rev thrust std.: {thrust_convergence['std_N']:.6e} N")

    y -= 22
    c.drawString(50, y, f"Relative thrust std.: {thrust_convergence['relative_std']:.6e}")

    y -= 22
    status_text = "PASSED" if thrust_convergence["passed"] else "FAILED"
    c.drawString(
        50,
        y,
        f"Thrust stability criterion: {status_text} "
        f"(threshold = {thrust_convergence['threshold']:.1e})",
    )

    last_rev_moment_mean = abs(evaluate_moments(times, moments, rev_time)["mean_N"])
    last_rev_power_mean = last_rev_moment_mean * rpm * ((2* math.pi )/(60))
   #last_rev_C_p_mean = last_rev_power_mean / (rho * (rpm / 60)**3 * D)

    y -= 24
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Last 1-rev torque around y-axis: {last_rev_moment_mean:.6f} Nm")

    y -= 24
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Last 1-rev power around y-axis: {last_rev_power_mean:.6f} W")


    c.showPage()

    # -----------------------------
    # Time-step and Courant-number evaluation
    # -----------------------------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, h - 50, "Time-step and Courant-number Evaluation")

    c.setFont("Helvetica", 10)
    c.drawString(
        50,
        h - 72,
        f"Source: {log_file.name} | Status: {timestep_courant_info['status']}",
    )

    y_ts = h - 110

    c.setFont("Helvetica-Bold", 12)
    
    y_ts -= 34
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_ts, "Observed solver time-step statistics")

    c.setFont("Helvetica", 10)
    y_ts -= 20
    c.drawString(
        50,
        y_ts,
        f"Samples: {delta_t_summary['samples']} | "
        f"Source: {delta_t_summary['source'] or 'Not found'}",
    )

    y_ts -= 18
    c.drawString(
        50,
        y_ts,
        "delta-t min / average / max: "
        f"{format_optional_number(delta_t_summary['min_s'])} / "
        f"{format_optional_number(delta_t_summary['average_s'])} / "
        f"{format_optional_number(delta_t_summary['max_s'])} s",
    )

    y_ts -= 18
    c.drawString(
        50,
        y_ts,
        "delta-t median / standard deviation: "
        f"{format_optional_number(delta_t_summary['median_s'])} / "
        f"{format_optional_number(delta_t_summary['std_s'])} s",
    )

    average_delta_t = delta_t_summary["average_s"]
    average_solver_frequency = (
        1.0 / average_delta_t
        if average_delta_t is not None and average_delta_t > 0.0
        else None
    )

    y_ts -= 18
    c.drawString(
        50,
        y_ts,
        "Equivalent average solver update rate: "
        f"{format_optional_number(average_solver_frequency, '.3f')} Hz",
    )

    def draw_courant_summary(title, summary, y_position):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, title)
        y_position -= 20

        c.setFont("Helvetica", 10)

        if summary is None:
            c.drawString(50, y_position, "No matching entries found in solver log.")
            return y_position - 26

        c.drawString(
            50,
            y_position,
            f"Samples: {summary['samples']} | "
            "Logged mean Co, average / maximum: "
            f"{format_optional_number(summary['mean_co_average'], '.4g')} / "
            f"{format_optional_number(summary['mean_co_max'], '.4g')}",
        )
        y_position -= 18

        c.drawString(
            50,
            y_position,
            "Logged maximum Co, average / peak: "
            f"{format_optional_number(summary['max_co_average'], '.4g')} / "
            f"{format_optional_number(summary['peak_max_co'], '.4g')}",
        )
        y_position -= 18

        exceedance_count = summary[
            "configured_max_co_exceedance_count"
        ]
        exceedance_percent = summary[
            "configured_max_co_exceedance_percent"
        ]

        if exceedance_count is not None:
            c.drawString(
                50,
                y_position,
                "Samples above configured maxCo: "
                f"{exceedance_count} "
                f"({format_optional_number(exceedance_percent, '.2f')}%)",
            )
            y_position -= 18

        return y_position - 18

    y_ts -= 34
    y_ts = draw_courant_summary(
        "Flow Courant number",
        timestep_courant_info["flow_courant"],
        y_ts,
    )

    if timestep_courant_info["mesh_courant"] is not None:
        y_ts = draw_courant_summary(
            "Mesh Courant number",
            timestep_courant_info["mesh_courant"],
            y_ts,
        )

    if (
        timestep_courant_info["interface_courant"] is not None
        and y_ts > 110
    ):
        draw_courant_summary(
            "Interface Courant number",
            timestep_courant_info["interface_courant"],
            y_ts,
        )

    c.showPage()

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, h - 50, "Force Evaluation")

    c.drawImage(str(force_plot), 40, 440, width=510, height=240)
    c.drawImage(str(conv_plot), 40, 150, width=510, height=240)

    if moments_plot is not None:
        c.showPage()

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, h - 50, "Torque Evaluation")

        c.drawImage(str(moments_plot), 40, 440, width=510, height=240)

    if residual_plot is not None:
        c.showPage()
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, h - 50, "Residual Evaluation")
        c.drawImage(str(residual_plot), 40, 350, width=510, height=260)

        if residual_slope_info is not None:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, 315, "Residual slopes over final revolution")
            c.setFont("Helvetica", 9)
            c.drawString(50, 298, "Slope definition: linear fit of log10(residual) over the final revolution, reported per revolution.")

            y_table = 278
            c.setFont("Helvetica-Bold", 8)
            c.drawString(50, y_table, "Field")
            c.drawString(160, y_table, "Slope / rev")
            c.drawString(260, y_table, "Final residual")
            c.drawString(370, y_table, "Mean residual")

            y_table -= 14
            c.setFont("Helvetica", 8)

            slope_items = list(residual_slope_info.get("slopes_per_rev", {}).items())
            for field, slope in slope_items[:10]:
                end_value = residual_slope_info.get("end_residuals", {}).get(field)
                mean_value = residual_slope_info.get("mean_residuals", {}).get(field)

                slope_text = "n/a" if slope is None else f"{slope:.3e}"
                end_text = "n/a" if end_value is None else f"{end_value:.3e}"
                mean_text = "n/a" if mean_value is None else f"{mean_value:.3e}"

                c.drawString(50, y_table, str(field)[:18])
                c.drawString(160, y_table, slope_text)
                c.drawString(260, y_table, end_text)
                c.drawString(370, y_table, mean_text)
                y_table -= 12

                if y_table < 40:
                    break

    if yplus_plot is not None:
        c.showPage()
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, h - 50, "Wall Treatment Evaluation")

        if yplus_stats is not None:
            c.setFont("Helvetica", 11)
            c.drawString(
                50,
                h - 80,
                f"Patch: {yplus_stats['patch_name']} | Time: {yplus_stats['time_dir']} | Faces: {yplus_stats['n_faces']}",
            )
            c.drawString(
                50,
                h - 100,
                f"Average y+: {yplus_stats['average_yplus']:.2f} | Median y+: {yplus_stats['median_yplus']:.2f} "
                f"| Min/Max y+: {yplus_stats['min_yplus']:.2f} / {yplus_stats['max_yplus']:.2f}",
            )
            c.drawString(
                50,
                h - 120,
                f"Surface share: y+ < 5: {yplus_stats['share_yplus_lt_5_percent']:.1f}% | "
                f"5 <= y+ <= 30: {yplus_stats['share_yplus_5_to_30_percent']:.1f}% | "
                f"y+ > 30: {yplus_stats['share_yplus_gt_30_percent']:.1f}%",
            )

        c.drawImage(
            str(yplus_plot),
            50,
            h - 500,
            width=500,
            height=330,
            preserveAspectRatio=True,
            mask="auto",
        )

    # -----------------------------
    # Acoustic Evaluation
    # -----------------------------
    acoustic_plot = (
        case_path
        / "report"
        / "spl_spectrum.png"
    )

    # A Path object is never None, so check whether the PNG really exists.
    # The page is still created when the file is missing or unreadable so the
    # report clearly shows why the acoustic figure was not embedded.
    c.showPage()

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, h - 50, "Acoustic Evaluation")

    c.setFont("Helvetica", 10)
    c.drawString(
        50,
        h - 75,
        "Predicted sound-pressure-level spectrum at the defined observer position.",
    )

    if acoustic_plot.is_file():
        try:
            acoustic_image = ImageReader(str(acoustic_plot))
            c.drawImage(
                acoustic_image,
                40,
                h - 520,
                width=510,
                height=400,
                preserveAspectRatio=True,
                anchor="c",
                mask="auto",
            )
            print(f"Acoustic plot added to report: {acoustic_plot}")
        except Exception as exc:
            warning = f"Acoustic plot could not be read: {exc}"
            print(warning)
            c.setFont("Helvetica", 10)
            c.drawString(50, h - 110, warning[:95])
            c.drawString(50, h - 128, f"Expected file: {acoustic_plot}"[:95])
    else:
        warning = f"Acoustic plot not found: {acoustic_plot}"
        print(warning)
        c.setFont("Helvetica", 10)
        c.drawString(50, h - 110, "Acoustic spectrum image was not found.")
        c.drawString(50, h - 128, f"Expected file: {acoustic_plot}"[:95])

    c.save()

    print(f"Report created: {output_pdf}")

    return {
        "case_path": str(case_path),
        "mode": mode,
        "rpm": float(rpm),
        "one_rev_time_s": rev_time,
        "simulated_time_s": latest_time,
        "effective_revolutions": eff_revs,
        "mesh_info": mesh_info,
        "mesh_element_types": mesh_element_types,
        "yplus_plot_path": str(yplus_plot) if yplus_plot is not None else None,
        "acoustic_plot": str(acoustic_plot) if acoustic_plot.is_file() else None,
        "yplus_stats": yplus_stats,
        "average_yplus": (
            yplus_stats["average_yplus"] if yplus_stats is not None else None
        ),
        "mesh_element_plot_path": str(mesh_element_plot) if mesh_element_plot is not None else None,
        "last_one_rev_avg_thrust_N": thrust_avg,
        "last_one_rev_thrust_std_N": thrust_convergence["std_N"],
        "last_one_rev_relative_thrust_std": thrust_convergence["relative_std"],
        "thrust_convergence_threshold": thrust_convergence["threshold"],
        "thrust_convergence_passed": thrust_convergence["passed"],
        "thrust_convergence_window_start_s": thrust_convergence["window_start_s"],
        "thrust_convergence_window_end_s": thrust_convergence["window_end_s"],
        "thrust_convergence_n_samples": thrust_convergence["n_samples"],
        "timestep_courant_info": timestep_courant_info,
        "execution_time_s": exec_time,
        "clock_time_s": clock_time,
        "pdf_path": str(output_pdf),
        "force_plot_path": str(force_plot),
        "force_convergence_plot_path": str(conv_plot),
        "thrust_stability_history_available": thrust_stability_history is not None,
        "residual_plot_path": str(residual_plot) if residual_plot is not None else None,
        "residual_slope_info": residual_slope_info,
    }

