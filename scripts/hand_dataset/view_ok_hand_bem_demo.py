"""Interactive PyVista viewer for the OK-hand BEM focusing demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from algorithm.config import SimulationConfig  # noqa: E402
from algorithm.physics.transducer_array import TransducerArray  # noqa: E402


def _load_demo(output_dir: Path) -> dict:
    layouts = [
        ("OK hand", "ok_hand_scene.vtk", "ok_hand_bem_response.npz",
         "ok_hand_levelset.npz", "ok_hand_bem_scene.yaml"),
        ("MANO hand", "mano_hand_scene.vtk", "mano_hand_bem_response.npz",
         "mano_hand_levelset.npz", "mano_hand_bem_scene.yaml"),
    ]
    layout = next(
        (item for item in layouts if (output_dir / item[1]).is_file()),
        layouts[0],
    )
    scene_name, mesh_name, response_name, level_name, config_name = layout
    mesh_file = output_dir / mesh_name
    response_file = output_dir / response_name
    level_file = output_dir / level_name
    config_file = output_dir / config_name
    metrics_file = output_dir / "metrics.json"
    missing = [p for p in (mesh_file, response_file, level_file, config_file) if not p.exists()]
    if missing:
        names = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Missing demo output files. Run scripts\\hand_dataset\\run_ok_hand_bem_demo.py first:\n"
            f"{names}"
        )

    response_data = np.load(response_file)
    level_data = np.load(level_file, allow_pickle=True)
    cfg = SimulationConfig.from_yaml(str(config_file))
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    return {
        "mesh_file": mesh_file,
        "phases": np.asarray(response_data["phases"], dtype=np.float64),
        "target": np.asarray(response_data["target"], dtype=np.float64),
        "axes": [np.asarray(a, dtype=np.float64) for a in level_data["axes"]],
        "amplitude": np.asarray(level_data["amplitude"], dtype=np.float64),
        "centers": TransducerArray(cfg).centers,
        "metrics": metrics,
        "scene_name": scene_name,
    }


def _faces_to_pyvista(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return np.empty(0, dtype=np.int64)
    return np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()


def _contour_mesh(axes: list[np.ndarray], amplitude: np.ndarray, threshold: float):
    import pyvista as pv
    from skimage.measure import marching_cubes

    peak = float(amplitude.max())
    level = float(threshold * peak)
    if not (float(amplitude.min()) < level < peak):
        return pv.PolyData(), level
    spacing = tuple(float(axis[1] - axis[0]) for axis in axes)
    vertices, faces, _, _ = marching_cubes(amplitude, level=level, spacing=spacing)
    vertices += np.array([axes[0][0], axes[1][0], axes[2][0]])
    return pv.PolyData(vertices, _faces_to_pyvista(faces)), level


def _make_board():
    import pyvista as pv

    vertices = np.array([
        [0.0, 0.0, 0.0],
        [0.18, 0.0, 0.0],
        [0.18, 0.18, 0.0],
        [0.0, 0.18, 0.0],
    ])
    faces = np.array([4, 0, 1, 2, 3])
    return pv.PolyData(vertices, faces)


def _target_sphere(target: np.ndarray):
    import pyvista as pv

    return pv.Sphere(radius=0.0022, center=tuple(target), theta_resolution=32, phi_resolution=16)


def _build_plotter(data: dict, threshold: float, off_screen: bool = False):
    import pyvista as pv

    plotter = pv.Plotter(window_size=(1280, 860), off_screen=off_screen)
    plotter.set_background("white")

    hand = pv.read(data["mesh_file"]).extract_surface(algorithm="dataset_surface").triangulate()
    plotter.add_mesh(
        hand,
        color="#aeb8c2",
        opacity=0.34,
        show_edges=True,
        edge_color="#7c8791",
        line_width=0.35,
        name="ok_hand",
    )
    plotter.add_mesh(_make_board(), color="#e9eef3", opacity=0.38, name="board")

    emitters = pv.PolyData(data["centers"])
    emitters["phase"] = np.mod(data["phases"], 2.0 * np.pi)
    plotter.add_mesh(
        emitters,
        scalars="phase",
        cmap="hsv",
        clim=(0.0, 2.0 * np.pi),
        render_points_as_spheres=True,
        point_size=16,
        name="phase_emitters",
        scalar_bar_args={"title": "phase 0..2pi"},
    )

    plotter.add_mesh(
        _target_sphere(data["target"]),
        color="#ffd8d8",
        opacity=0.42,
        show_edges=True,
        edge_color="#d71920",
        line_width=1.2,
        name="target",
    )
    plotter.add_points(
        data["target"][None, :],
        color="#d71920",
        point_size=9,
        render_points_as_spheres=True,
        name="target_center",
    )

    level_mesh, level_value = _contour_mesh(data["axes"], data["amplitude"], threshold)
    plotter.add_mesh(
        level_mesh,
        color="#00bde3",
        opacity=0.36,
        show_edges=True,
        edge_color="#007894",
        line_width=0.45,
        name="levelset",
    )

    peak = float(data["amplitude"].max())
    text_actor = plotter.add_text(
        _status_text(data, threshold, level_value, peak, level_mesh.n_cells),
        position="upper_left",
        font_size=11,
        color="black",
        name="status_text",
    )

    def update_threshold(value):
        value = float(value)
        new_mesh, new_level = _contour_mesh(data["axes"], data["amplitude"], value)
        plotter.remove_actor("levelset")
        plotter.add_mesh(
            new_mesh,
            color="#00bde3",
            opacity=0.36,
            show_edges=True,
            edge_color="#007894",
            line_width=0.45,
            name="levelset",
        )
        status = _status_text(data, value, new_level, peak, new_mesh.n_cells)
        if hasattr(text_actor, "SetInput"):
            text_actor.SetInput(status)
        else:
            text_actor.SetText(2, status)
        plotter.render()

    if not off_screen:
        plotter.add_slider_widget(
            update_threshold,
            rng=(0.15, 0.90),
            value=threshold,
            title="amplitude level / peak",
            pointa=(0.26, 0.06),
            pointb=(0.74, 0.06),
            style="modern",
        )

    plotter.add_axes()
    plotter.show_grid(
        bounds=(0.0, 0.18, 0.0, 0.18, 0.0, 0.17),
        xtitle="x / m",
        ytitle="y / m",
        ztitle="z / m",
        color="gray",
    )
    plotter.camera_position = [
        (0.245, -0.215, 0.188),
        (0.082, 0.082, 0.064),
        (-0.22, 0.26, 0.94),
    ]
    return plotter


def _status_text(data: dict, threshold: float, level_value: float, peak: float, faces: int) -> str:
    metrics = data.get("metrics", {})
    response_s = metrics.get("response_s")
    response_line = f"BEM response: {float(response_s):.2f}s" if response_s is not None else "BEM response: n/a"
    return (
        f"{data.get('scene_name', 'Hand')} BEM focus viewer\n"
        f"target: {np.array2string(data['target'], precision=3)} m\n"
        f"level: {threshold:.0%} peak = {level_value:.3e}\n"
        f"peak amplitude: {peak:.3e}\n"
        f"level triangles: {faces}\n"
        f"{response_line}\n"
        "mouse: rotate / zoom / pan"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/hand_ok_bem_demo"))
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--screenshot", default="", help="Write screenshot and exit instead of opening a window")
    args = parser.parse_args()

    data = _load_demo(Path(args.output_dir))
    off_screen = bool(args.screenshot)
    plotter = _build_plotter(data, float(args.threshold), off_screen=off_screen)
    if args.screenshot:
        out = Path(args.screenshot)
        out.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(out), auto_close=True)
        print(out)
    else:
        plotter.show(title=f"{data.get('scene_name', 'Hand')} BEM interactive viewer")


if __name__ == "__main__":
    main()
