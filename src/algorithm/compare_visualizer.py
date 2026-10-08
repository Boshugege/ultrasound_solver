"""Interactive PyVista browser for results written by ``compare.py``."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyvista as pv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithm.visualizer import _add_static_scene_context, _copy_scalar_field, amplitude_unit


FIELD_LABELS = {
    "amplitude": "Final amplitude",
    "best_amplitude": "Best evaluated amplitude",
    "initial_amplitude": "Initial amplitude",
    "geometric_amplitude": "Geometric phase amplitude",
}


def _load_results(comparison_dir):
    directory = Path(comparison_dir)
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result_sets = {}
    static = None
    for row in summary["rows"]:
        if row["status"] != "completed":
            continue
        algorithm = row["algorithm"]
        result_path = directory / algorithm / "result.npz"
        if not result_path.exists():
            continue
        with np.load(result_path, allow_pickle=True) as data:
            fields = {
                name: np.asarray(data[name], dtype=np.float64)
                for name in FIELD_LABELS if name in data
            }
            if "amplitude" not in fields:
                continue
            if static is None:
                static = {
                    key: np.array(data[key], copy=True)
                    for key in (
                        "dx", "source_positions", "target_points",
                        "transducer_radius", "boundary_conditions", "sdf",
                        "field_unit",
                    )
                    if key in data
                }
            elif fields["amplitude"].shape != next(iter(result_sets.values()))["amplitude"].shape:
                raise ValueError("All compared results must share the same grid shape")
            result_sets[algorithm] = fields
    if not result_sets:
        raise ValueError("No completed result.npz files found")
    return summary, static, result_sets


def show_comparison_results(comparison_dir, percentile=95.0, hide_source_layers=0):
    summary, static, result_sets = _load_results(comparison_dir)
    unit = amplitude_unit(static)
    algorithms = list(result_sets)
    modes = [
        name for name in FIELD_LABELS
        if any(name in fields for fields in result_sets.values())
    ]
    state = {
        "algorithm_index": 0,
        "field_mode": "amplitude",
        "percentile": float(np.clip(percentile, 0.0, 100.0)),
    }
    first = result_sets[algorithms[0]]["amplitude"]
    dx = float(static["dx"])
    nx, ny, nz = first.shape
    lx, ly, lz = (nx - 1) * dx, (ny - 1) * dx, (nz - 1) * dx
    grid = pv.ImageData(
        dimensions=(nx, ny, nz), spacing=(dx, dx, dx), origin=(0, 0, 0)
    )
    grid.point_data["ComparisonField"] = first.flatten(order="F")

    plotter = pv.Plotter(window_size=[1440, 900])
    plotter.set_background("#16161a", top="#22222a")
    plotter.enable_anti_aliasing("msaa")
    _add_static_scene_context(plotter, grid, static, lx, ly, lz, dx)
    hud = plotter.add_text("", position="upper_left", font_size=11,
                           color="white", font="courier")

    def active_field():
        algorithm = algorithms[state["algorithm_index"]]
        fields = result_sets[algorithm]
        return fields.get(state["field_mode"], fields["amplitude"]), algorithm

    def redraw():
        field, algorithm = active_field()
        field = _copy_scalar_field(
            {"field": field}, "field", hide_source_layers
        )
        finite = field[np.isfinite(field)]
        low, high = float(np.min(finite)), float(np.max(finite))
        level = float(np.percentile(finite, state["percentile"]))
        if low == high:
            level = low
        grid.point_data["ComparisonField"] = field.flatten(order="F")
        plotter.add_mesh(
            grid.contour([level], scalars="ComparisonField"),
            name="comparison_amplitude_isosurface",
            cmap="plasma", clim=[low, high],
            opacity=0.78, smooth_shading=True, show_scalar_bar=False,
        )
        hud.SetText(
            0,
            f"Algorithm: {algorithm}\n"
            f"Field: {FIELD_LABELS[state['field_mode']]}\n"
            f"Iso percentile: {state['percentile']:.0f}%\n"
            f"Level: {level:.1f} {unit} | Max: {high:.1f} {unit}",
        )

    def algorithm_callback(value):
        state["algorithm_index"] = int(np.clip(round(value), 0, len(algorithms) - 1))
        redraw()

    def percentile_callback(value):
        state["percentile"] = float(value)
        redraw()

    redraw()
    if len(algorithms) > 1:
        plotter.add_slider_widget(
            algorithm_callback, rng=[0, len(algorithms) - 1], value=0,
            title="Algorithm index", pointa=(0.04, 0.91), pointb=(0.33, 0.91),
            color="#4dabf7", style="modern",
        )
        labels = "\n".join(f"{index}: {name}" for index, name in enumerate(algorithms))
        plotter.add_text(labels, position=(20, 110), font_size=9,
                         color="#d0ebff", font="courier")
    plotter.add_slider_widget(
        percentile_callback, rng=[0.0, 100.0], value=state["percentile"],
        title="Iso percentile", pointa=(0.04, 0.84), pointb=(0.33, 0.84),
        color="#38d9a9", style="modern",
    )

    # PyVista/VTK has no native combo box. A small Tk control uses the platform
    # standard dropdown, while a VTK timer processes its events on the same thread.
    control_root = None
    try:
        import tkinter as tk
        from tkinter import ttk

        control_root = tk.Tk()
        control_root.title("Compared field")
        control_root.resizable(False, False)
        frame = ttk.Frame(control_root, padding=12)
        frame.grid()
        ttk.Label(frame, text="Displayed field").grid(row=0, column=0, sticky="w")
        selected = tk.StringVar(value=FIELD_LABELS[state["field_mode"]])
        combo = ttk.Combobox(
            frame, state="readonly", width=31, textvariable=selected,
            values=[FIELD_LABELS[mode] for mode in modes],
        )
        combo.grid(row=1, column=0, pady=(4, 0))

        def on_field_change(_event):
            label = selected.get()
            state["field_mode"] = next(
                mode for mode, text in FIELD_LABELS.items() if text == label
            )
            redraw()

        combo.bind("<<ComboboxSelected>>", on_field_change)

        def pump_controls(_step):
            try:
                control_root.update_idletasks()
                control_root.update()
            except tk.TclError:
                pass

        plotter.add_timer_event(max_steps=10_000_000, duration=100,
                                callback=pump_controls)
    except Exception as exc:
        print(f"[CompareVisualizer] Field dropdown unavailable: {exc}")

    plotter.show_bounds(
        grid="front", location="outer", all_edges=True, color="#8888aa",
        xtitle="X (m)", ytitle="Y (m)", ztitle="Z (m)",
    )
    plotter.camera_position = [
        (lx * 2.2, -ly * 1.8, lz * 2.0),
        (lx * 0.5, ly * 0.5, lz * 0.5),
        (0, 0, 1),
    ]
    print(f"[CompareVisualizer] Showing {len(algorithms)} completed algorithms")
    try:
        plotter.show()
    finally:
        if control_root is not None:
            try:
                control_root.destroy()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Browse amplitude fields saved by algorithm.compare"
    )
    parser.add_argument(
        "comparison_dir",
        nargs="?",
        default="outputs/algs/phase_oblique_reflecting_x_16x16",
    )
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--hide-source-layers", type=int, default=4)
    args = parser.parse_args()
    show_comparison_results(
        args.comparison_dir, args.percentile, args.hide_source_layers
    )


if __name__ == "__main__":
    main()
