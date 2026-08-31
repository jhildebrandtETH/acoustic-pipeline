from pathlib import Path
from datetime import datetime
import numpy as np
import re
import math

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from tools import (
    create_force_plots,
    create_mesh_element_plot,
    create_moments_plots,
    create_residual_plots,
    create_yplus_distribution_plot,
    draw_courant_summary,
    evaluate_moments,
    evaluate_thrust_convergence,
    format_optional_number,
    format_seconds,
    read_mesh_element_types,
    read_mesh_information,
    read_openfoam_timestep_and_courant_statistics,
)


def create_simulation_report(
    case_path,
    rpm,
    mode,
    turbulence_model,
    output_pdf=None,
    quiet=False,
):
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
        patch_name="propeller",
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
        if not quiet:
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


    y_ts -= 34
    y_ts = draw_courant_summary(
        c,
        "Flow Courant number",
        timestep_courant_info["flow_courant"],
        y_ts,
    )

    if timestep_courant_info["mesh_courant"] is not None:
        y_ts = draw_courant_summary(
            c,
            "Mesh Courant number",
            timestep_courant_info["mesh_courant"],
            y_ts,
        )

    if (
        timestep_courant_info["interface_courant"] is not None
        and y_ts > 110
    ):
        draw_courant_summary(
            c,
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
            if not quiet:
                print(f"Acoustic plot added to report: {acoustic_plot}")
        except Exception as exc:
            warning = f"Acoustic plot could not be read: {exc}"
            if not quiet:
                print(warning)
            c.setFont("Helvetica", 10)
            c.drawString(50, h - 110, warning[:95])
            c.drawString(50, h - 128, f"Expected file: {acoustic_plot}"[:95])
    else:
        warning = f"Acoustic plot not found: {acoustic_plot}"
        if not quiet:
            print(warning)
        c.setFont("Helvetica", 10)
        c.drawString(50, h - 110, "Acoustic spectrum image was not found.")
        c.drawString(50, h - 128, f"Expected file: {acoustic_plot}"[:95])

    c.save()

    if not quiet:
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

