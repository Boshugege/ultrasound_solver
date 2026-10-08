import argparse
import json
import numpy as np
import pyvista as pv


FIELD_ALIASES = {
    "geometric_amplitude": ("geometric_amplitude", "baseline_amplitude"),
}


def build_parser():
    parser = argparse.ArgumentParser(description="PyVista 硬件加速超声声场 3D 交互可视化")
    parser.add_argument("result", nargs="?", default="ultrasound_field.npz", help="npz 仿真结果路径")
    parser.add_argument("--percentile", type=float, default=80.0, help="初始等值面分位数 (0~100)")
    parser.add_argument("--use-sq", action="store_true", help="显示声强 (|p|^2)，默认显示声压幅值 (|p|)")
    parser.add_argument(
        "--field",
        choices=[
            "amplitude",
            "geometric_amplitude",
            "initial_amplitude",
            "target_amplitude",
            "abs_error",
            "signed_error",
        ],
        default="amplitude",
        help="选择要显示的三维标量场",
    )
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="列出 npz 中可用于可视化的字段后退出",
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="同屏比较传统几何相位和当前优化相位，红色为传统，蓝色为当前方法",
    )
    parser.add_argument(
        "--hide-source-layers",
        type=int,
        default=0,
        help="隐藏靠近 z=0 的底部网格层数，用于排除换能器近场峰值",
    )
    return parser


def create_reflecting_wall_plane(face_key: str, lx: float, ly: float, lz: float) -> pv.PolyData:
    """根据边界面方位生成对应的灰色挡板 3D 平面网格"""
    if face_key == "-x":
        return pv.Plane(center=(0.0, ly * 0.5, lz * 0.5), direction=(1, 0, 0), i_size=ly, j_size=lz)
    elif face_key == "+x":
        return pv.Plane(center=(lx, ly * 0.5, lz * 0.5), direction=(-1, 0, 0), i_size=ly, j_size=lz)
    elif face_key == "-y":
        return pv.Plane(center=(lx * 0.5, 0.0, lz * 0.5), direction=(0, 1, 0), i_size=lx, j_size=lz)
    elif face_key == "+y":
        return pv.Plane(center=(lx * 0.5, ly, lz * 0.5), direction=(0, -1, 0), i_size=lx, j_size=lz)
    elif face_key == "-z":
        return pv.Plane(center=(lx * 0.5, ly * 0.5, 0.0), direction=(0, 0, 1), i_size=lx, j_size=ly)
    elif face_key == "+z":
        return pv.Plane(center=(lx * 0.5, ly * 0.5, lz), direction=(0, 0, -1), i_size=lx, j_size=ly)
    return None


def available_scalar_fields(data) -> list[str]:
    fields = []
    for key in (
        "amplitude",
        "geometric_amplitude",
        "initial_amplitude",
        "target_amplitude",
    ):
        if _resolve_field_key(data, key) is not None:
            fields.append(key)
    if "amplitude" in data and "target_amplitude" in data:
        fields.extend(["abs_error", "signed_error"])
    return fields


def _resolve_field_key(data, key: str) -> str | None:
    for candidate in FIELD_ALIASES.get(key, (key,)):
        if candidate in data:
            return candidate
    return None


def amplitude_unit(data):
    return "1/m (normalized)" if str(data.get("field_unit", "Pa")) == "normalized_1_per_m" else "Pa"


def select_scalar_field(data, field_name: str, use_sq: bool):
    unit = amplitude_unit(data)
    if field_name == "abs_error":
        field = np.abs(data["amplitude"] - data["target_amplitude"])
        return field, "Error |optimized - target|", unit
    if field_name == "signed_error":
        field = data["amplitude"] - data["target_amplitude"]
        return field, "Error optimized - target", unit
    field_key = _resolve_field_key(data, field_name)
    if field_key is None:
        valid = ", ".join(available_scalar_fields(data))
        raise ValueError(f"Field {field_name!r} is unavailable. Valid fields: {valid}")
    field = data[field_key]
    if field_name == "amplitude" and use_sq:
        if "amp_sq" in data:
            field = data["amp_sq"]
        else:
            field = field**2
        return field, "Squared amplitude (|p|²)", "1/m² (normalized)" if unit != "Pa" else "Pa²"
    titles = {
        "amplitude": "Amplitude optimized |p|",
        "geometric_amplitude": "Amplitude geometric phase |p|",
        "initial_amplitude": "Amplitude initial |p|",
        "target_amplitude": "Target amplitude",
    }
    return field, titles[field_name], unit


def _copy_scalar_field(data, key: str, hide_source_layers: int) -> np.ndarray:
    field_key = _resolve_field_key(data, key)
    if field_key is None:
        raise KeyError(key)
    field = np.array(data[field_key], dtype=np.float64, copy=True)
    if hide_source_layers > 0:
        layers = min(int(hide_source_layers), field.shape[2])
        field[:, :, :layers] = np.nan
    return field


def _parse_boundary_conditions(data) -> dict:
    bcs = {
        "-x": "reflecting",
        "+x": "reflecting",
        "-y": "reflecting",
        "+y": "reflecting",
        "-z": "reflecting",
        "+z": "open",
    }
    if "boundary_conditions" in data:
        bcs = json.loads(str(data["boundary_conditions"]))
    return bcs


def _add_static_scene_context(
    plotter,
    grid,
    data,
    lx: float,
    ly: float,
    lz: float,
    dx: float,
):
    bcs = _parse_boundary_conditions(data)
    for face_key, bc_type in bcs.items():
        if bc_type == "reflecting":
            wall_plane = create_reflecting_wall_plane(face_key, lx, ly, lz)
            if wall_plane is not None:
                plotter.add_mesh(
                    wall_plane,
                    color="#ff0000",
                    opacity=0.22,
                    show_edges=True,
                    edge_color="#707078",
                    line_width=1.5,
                    label=f"Reflecting Wall ({face_key})",
                )

    plotter.add_mesh(grid.outline(), color="#888899", line_width=1.0)

    if "source_positions" in data:
        transducer_discs = []
        src_pos = data["source_positions"]
        trans_radius = (
            float(data["transducer_radius"])
            if "transducer_radius" in data
            else 0.0049
        )
        cylinder_height = dx * 0.4
        for center in src_pos:
            tx, ty, tz = center
            disc = pv.Cylinder(
                center=(tx, ty, tz - cylinder_height * 0.5),
                direction=(0, 0, 1),
                radius=trans_radius,
                height=cylinder_height,
                resolution=36,
            )
            transducer_discs.append(disc)

        if transducer_discs:
            merged_transducers = transducer_discs[0].merge(transducer_discs[1:])
            plotter.add_mesh(
                merged_transducers,
                color="#f0b429",
                metallic=0.6,
                roughness=0.3,
                specular=0.8,
                smooth_shading=True,
                label="Transducer Array (z=0)",
            )

    if "target_points" in data:
        target_pts = np.atleast_2d(data["target_points"])
        if len(target_pts) > 0:
            target_spheres = [
                pv.Sphere(radius=dx * 1.5, center=pt)
                for pt in target_pts
            ]
            merged_targets = target_spheres[0].merge(target_spheres[1:])
            plotter.add_mesh(
                merged_targets,
                color="#ff1744",
                opacity=0.3,
                emissive=True,
                smooth_shading=True,
                label="Targets",
            )


def show_method_comparison_scene(
    result_path: str,
    percentile: float = 95.0,
    hide_source_layers: int = 0,
):
    data = np.load(result_path, allow_pickle=True)
    unit = amplitude_unit(data)
    if _resolve_field_key(data, "geometric_amplitude") is None:
        data.close()
        raise ValueError(
            "comparison requires geometric_amplitude (or legacy "
            "baseline_amplitude) in the result file"
        )

    geometric = _copy_scalar_field(
        data,
        "geometric_amplitude",
        hide_source_layers,
    )
    optimized = _copy_scalar_field(
        data,
        "amplitude",
        hide_source_layers,
    )
    dx = float(data["dx"])
    nx, ny, nz = optimized.shape
    lx, ly, lz = (nx - 1) * dx, (ny - 1) * dx, (nz - 1) * dx

    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(dx, dx, dx),
        origin=(0, 0, 0),
    )
    grid.point_data["GeometricAmplitude"] = geometric.flatten(order="F")
    grid.point_data["OptimizedAmplitude"] = optimized.flatten(order="F")

    geometric_min = float(np.nanmin(geometric))
    geometric_max = float(np.nanmax(geometric))
    optimized_min = float(np.nanmin(optimized))
    optimized_max = float(np.nanmax(optimized))
    geometric_level = float(np.nanpercentile(geometric, percentile))
    optimized_level = float(np.nanpercentile(optimized, percentile))

    plotter = pv.Plotter(window_size=[1366, 860])
    plotter.set_background("#16161a", top="#22222a")
    plotter.enable_anti_aliasing("msaa")
    _add_static_scene_context(plotter, grid, data, lx, ly, lz, dx)

    geometric_actor_name = "geometric_red_isosurface"
    optimized_actor_name = "optimized_blue_isosurface"
    plotter.add_mesh(
        grid.contour([geometric_level], scalars="GeometricAmplitude"),
        name=geometric_actor_name,
        color="#ff1744",
        opacity=0.46,
        smooth_shading=True,
        label="Geometric phase",
    )
    plotter.add_mesh(
        grid.contour([optimized_level], scalars="OptimizedAmplitude"),
        name=optimized_actor_name,
        color="#2979ff",
        opacity=0.50,
        smooth_shading=True,
        label="Optimized phase",
    )

    hud_actor = plotter.add_text(
        "Geometric red vs optimized blue\n"
        f"Red level: {geometric_level:.1f} {unit} / max {geometric_max:.1f} {unit}\n"
        f"Blue level: {optimized_level:.1f} {unit} / max {optimized_max:.1f} {unit}",
        position="upper_left",
        font_size=11,
        color="white",
        font="courier",
    )

    levels = {
        "geometric": geometric_level,
        "optimized": optimized_level,
    }

    def refresh_hud():
        hud_actor.SetText(
            0,
            "Geometric red vs optimized blue\n"
            f"Red level: {levels['geometric']:.1f} {unit} / max {geometric_max:.1f} {unit}\n"
            f"Blue level: {levels['optimized']:.1f} {unit} / max {optimized_max:.1f} {unit}",
        )

    def geometric_slider_callback(value):
        levels["geometric"] = float(value)
        plotter.add_mesh(
            grid.contour([float(value)], scalars="GeometricAmplitude"),
            name=geometric_actor_name,
            color="#ff1744",
            opacity=0.46,
            smooth_shading=True,
            show_scalar_bar=False,
        )
        refresh_hud()

    def optimized_slider_callback(value):
        levels["optimized"] = float(value)
        plotter.add_mesh(
            grid.contour([float(value)], scalars="OptimizedAmplitude"),
            name=optimized_actor_name,
            color="#2979ff",
            opacity=0.50,
            smooth_shading=True,
            show_scalar_bar=False,
        )
        refresh_hud()

    plotter.add_slider_widget(
        callback=geometric_slider_callback,
        rng=[geometric_min, geometric_max],
        value=geometric_level,
        title=f"Geometric red threshold ({unit})",
        pointa=(0.04, 0.92),
        pointb=(0.36, 0.92),
        color="#ff1744",
        style="modern",
    )
    plotter.add_slider_widget(
        callback=optimized_slider_callback,
        rng=[optimized_min, optimized_max],
        value=optimized_level,
        title=f"Optimized blue threshold ({unit})",
        pointa=(0.04, 0.84),
        pointb=(0.36, 0.84),
        color="#2979ff",
        style="modern",
    )

    plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
        color="#8888aa",
        xtitle="X (m)",
        ytitle="Y (m)",
        ztitle="Z (m)",
    )
    plotter.add_legend(
        bcolor="#202028",
        border=True,
        size=(0.24, 0.20),
        loc="upper right",
    )
    plotter.camera_position = [
        (lx * 2.2, -ly * 1.8, lz * 2.0),
        (lx * 0.5, ly * 0.5, lz * 0.5),
        (0, 0, 1),
    ]

    print("[PyVista] 启动传统/优化相位 3D 对比视窗...")
    plotter.show()
    data.close()


def show_pyvista_scene(
    result_path: str,
    percentile: float = 80.0,
    use_sq: bool = False,
    field_name: str = "amplitude",
    list_fields: bool = False,
    hide_source_layers: int = 0,
    compare_methods: bool = False,
):
    # 1. 加载仿真数据
    data = np.load(result_path, allow_pickle=True)
    if list_fields:
        print("可视化字段:")
        for field in available_scalar_fields(data):
            print(f"  - {field}")
        data.close()
        return
    if compare_methods:
        data.close()
        show_method_comparison_scene(
            result_path,
            percentile=percentile,
            hide_source_layers=hide_source_layers,
        )
        return
    field, field_title, field_unit = select_scalar_field(
        data,
        field_name,
        use_sq,
    )
    field = np.array(field, dtype=np.float64, copy=True)
    if hide_source_layers > 0:
        layers = min(int(hide_source_layers), field.shape[2])
        field[:, :, :layers] = np.nan

    dx = float(data["dx"])
    nx, ny, nz = field.shape
    lx, ly, lz = (nx - 1) * dx, (ny - 1) * dx, (nz - 1) * dx

    sdf = data["sdf"] if "sdf" in data else None
    src_pos = data["source_positions"] if "source_positions" in data else None
    targets = data["target_points"] if "target_points" in data else None
    trans_radius = float(data["transducer_radius"]) if "transducer_radius" in data else 0.0049

    # 边界条件解析
    bcs = {"-x": "reflecting", "+x": "reflecting", "-y": "reflecting", "+y": "reflecting", "-z": "reflecting", "+z": "open"}
    if "boundary_conditions" in data:
        try:
            bcs = json.loads(str(data["boundary_conditions"]))
        except Exception:
            pass

    # 2. 构建 PyVista 统一结构化网格
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(dx, dx, dx), origin=(0, 0, 0))
    grid.point_data["ScalarField"] = field.flatten(order="F")
    if sdf is not None:
        grid.point_data["SDF"] = sdf.flatten(order="F")

    # 3. 初始化 PyVista 3D 渲染窗口
    plotter = pv.Plotter(window_size=[1366, 860])
    plotter.set_background("#16161a", top="#22222a")
    plotter.enable_anti_aliasing("msaa")

    # -------------------------------------------------------------------------
    # 4. 绘制全反射面 (灰色半透明挡板) 与开放面边界外框
    # -------------------------------------------------------------------------
    for face_key, bc_type in bcs.items():
        if bc_type == "reflecting":
            wall_plane = create_reflecting_wall_plane(face_key, lx, ly, lz)
            if wall_plane is not None:
                plotter.add_mesh(
                    wall_plane,
                    color="#ff0000",            # 修复：设为高质感暗灰色挡板
                    opacity=0.35,
                    show_edges=True,
                    edge_color="#707078",
                    line_width=1.5,
                    label=f"Reflecting Wall ({face_key})",
                )

    # 空间总包围边框
    plotter.add_mesh(grid.outline(), color="#888899", line_width=1.0)

    # -------------------------------------------------------------------------
    # 5. 绘制底面换能器阵列 (金色金属圆盘)
    # -------------------------------------------------------------------------
    if src_pos is not None:
        transducer_discs = []
        cylinder_height = dx * 0.4
        for center in src_pos:
            tx, ty, tz = center
            disc = pv.Cylinder(
                center=(tx, ty, tz - cylinder_height * 0.5),
                direction=(0, 0, 1),
                radius=trans_radius,
                height=cylinder_height,
                resolution=36,
            )
            transducer_discs.append(disc)

        if transducer_discs:
            merged_transducers = transducer_discs[0].merge(transducer_discs[1:])
            plotter.add_mesh(
                merged_transducers,
                color="#f0b429",
                metallic=0.6,
                roughness=0.3,
                specular=0.8,
                smooth_shading=True,
                label="Transducer Array (z=0)",
            )

    # -------------------------------------------------------------------------
    # 6. 绘制 SDF=0 障碍物物理表面 (青色玻璃质感)
    # -------------------------------------------------------------------------
    if sdf is not None and np.min(sdf) <= 0.0 <= np.max(sdf):
        try:
            sdf_contour = grid.contour([0.0], scalars="SDF")
            plotter.add_mesh(
                sdf_contour,
                color="#00e5ff",
                opacity=0.45,
                smooth_shading=True,
                specular=0.9,
                split_sharp_edges=True,
                label="Obstacle (SDF = 0)",
            )
            print("[PyVista] SDF 障碍物几何表面渲染成功。")
        except Exception as e:
            print(f"[PyVista] SDF 提取跳过: {e}")

    # -------------------------------------------------------------------------
    # 7. 绘制多目标焦点 (红色高亮球体)
    # -------------------------------------------------------------------------
    if targets is not None and len(targets) > 0:
        target_pts = np.atleast_2d(targets)
        target_spheres = []
        for pt in target_pts:
            sp = pv.Sphere(radius=dx * 1.5, center=pt)
            target_spheres.append(sp)
        merged_targets = target_spheres[0].merge(target_spheres[1:])
        plotter.add_mesh(
            merged_targets,
            color="#ff1744",
            opacity=0.2,               
            emissive=True,
            smooth_shading=True,
            label="Targets",
        )

    # -------------------------------------------------------------------------
    # 8. 等振幅声场曲面 (Flying Edges 极速硬件光栅化) 与交互滑块
    # -------------------------------------------------------------------------
    raw_min, raw_max = float(np.nanmin(field)), float(np.nanmax(field))
    init_level = float(np.nanpercentile(field, percentile))
    if raw_min == raw_max:
        init_level = raw_min
    else:
        init_level = float(np.clip(init_level, raw_min + 1e-4, raw_max - 1e-4))

    # 初始曲面提取
    init_contour = grid.contour([init_level], scalars="ScalarField")
    
    # 建立右侧全局标尺色标 (Scalar Bar)
    scalar_bar_args = {
        "title": f"Acoustic {field_title} ({field_unit})",
        "vertical": True,
        "position_x": 0.88,
        "position_y": 0.15,
        "height": 0.7,
        "width": 0.05,
        "title_font_size": 11,
        "label_font_size": 9,
    }

    plotter.add_mesh(
        init_contour,
        name="amplitude_isosurface",
        cmap="plasma",
        clim=[raw_min, raw_max],       # 核心修复：色彩严格映射到全局最大与最小振幅区间
        opacity=0.72,
        smooth_shading=True,
        show_scalar_bar=True,
        scalar_bar_args=scalar_bar_args,
    )

    # 左上角 HUD 文字
    hud_actor = plotter.add_text(
        f"{field_title}\nIsosurface Level: {init_level:.1f} {field_unit}\nMax: {raw_max:.1f} {field_unit}",
        position="upper_left",
        font_size=11,
        color="white",
        font="courier",
    )

    # 滑块实时更新回调
    def slider_callback(value):
        new_contour = grid.contour([value], scalars="ScalarField")
        plotter.add_mesh(
            new_contour,
            name="amplitude_isosurface",
            cmap="plasma",
            clim=[raw_min, raw_max],   # 核心修复：滑动更新时继续保持全局颜色映射，不产生局部偏色
            opacity=1,
            smooth_shading=True,
            show_scalar_bar=False,     # 保持复用之前的 Scalar Bar
        )
        active_ratio = float(np.nanmean(field >= value)) * 100.0
        hud_actor.SetText(
            0,
            f"Level: {value:8.1f} {field_unit} ({(value/raw_max)*100:4.1f}%)\n"
            f"Active Volume : {active_ratio:5.2f}%\n"
            f"Field Peak    : {raw_max:8.1f} {field_unit}",
        )

    # 滑动条上下限严格映射至 [raw_min, raw_max]
    plotter.add_slider_widget(
        callback=slider_callback,
        rng=[raw_min, raw_max],
        value=init_level,
        title=f"Iso Threshold ({field_unit})",
        pointa=(0.04, 0.90),
        pointb=(0.32, 0.90),
        color="#00e5ff",
        style="modern",
    )

    # 添加标尺与坐标轴
    plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
        color="#8888aa",
        xtitle="X (m)",
        ytitle="Y (m)",
        ztitle="Z (m)",
    )
    plotter.add_legend(bcolor="#202028", border=True, size=(0.20, 0.20), loc="upper right")
    plotter.camera_position = [(lx * 2.2, -ly * 1.8, lz * 2.0), (lx * 0.5, ly * 0.5, lz * 0.5), (0, 0, 1)]

    print("[PyVista] 启动 GPU 硬件加速 3D 视窗...")
    plotter.show()


def main():
    args = build_parser().parse_args()
    show_pyvista_scene(
        args.result,
        percentile=args.percentile,
        use_sq=args.use_sq,
        field_name=args.field,
        list_fields=args.list_fields,
        hide_source_layers=args.hide_source_layers,
        compare_methods=args.compare_methods,
    )


if __name__ == "__main__":
    main()
