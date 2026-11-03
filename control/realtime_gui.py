"""Realtime SonicSurface focus GUI driven by BEM target-response phases."""

from __future__ import annotations

import math
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any
from urllib.parse import urlsplit

import numpy as np

if __package__:
    from .sonicsurface.client import request_json
else:
    from sonicsurface.client import request_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "control" / "sonicsurface.local.yaml"
DEFAULT_SERVICE_URL = "http://127.0.0.1:8766"

CHANNELS_PER_SIDE = 16
CHANNELS = CHANNELS_PER_SIDE * CHANNELS_PER_SIDE
BOARD_SIDE_M = 0.16
TRANSDUCER_PITCH_M = 0.01
SOUND_SPEED_M_S = 343.0
ULTRASOUND_HZ = 40_000.0
WAVENUMBER = 2.0 * math.pi * ULTRASOUND_HZ / SOUND_SPEED_M_S

DEFAULT_FOCUS = np.asarray([BOARD_SIDE_M * 0.5, BOARD_SIDE_M * 0.5, 0.06])
DEFAULT_STEP_M = 0.002
DEFAULT_UPLOAD_HZ = 30.0
STATUS_INTERVAL_MS = 1000
CONTINUOUS_INTERVAL_MIN_MS = 10
MODE_BEM = "BEM无限正交面"


def _emitter_centers() -> np.ndarray:
    """Emitter centers in the same logical order used by the BEM solver."""
    import json

    data = json.loads(
        (REPOSITORY_ROOT / "src" / "algorithm" / "physics" / "sonicsurface_reference.json")
        .read_text(encoding="utf-8")
    )
    positions = np.asarray(data["positions_vendor_xyz_m"], dtype=np.float64)
    logical_to_device = np.asarray(data["logical_to_device"], dtype=np.int64)
    centers = positions[logical_to_device][:, [0, 2, 1]].copy()
    centers[:, :2] += np.asarray([0.08, 0.08], dtype=np.float64)
    return centers


EMITTER_CENTERS = _emitter_centers()
_VISUAL_MARGIN_M = TRANSDUCER_PITCH_M * 0.5
BOARD_VISUAL_MIN = np.asarray(
    [
        float(EMITTER_CENTERS[:, 0].min() - _VISUAL_MARGIN_M),
        float(EMITTER_CENTERS[:, 1].min() - _VISUAL_MARGIN_M),
    ],
    dtype=np.float64,
)
BOARD_VISUAL_MAX = np.asarray(
    [
        float(EMITTER_CENTERS[:, 0].max() + _VISUAL_MARGIN_M),
        float(EMITTER_CENTERS[:, 1].max() + _VISUAL_MARGIN_M),
    ],
    dtype=np.float64,
)
BOARD_VISUAL_CENTER = 0.5 * (BOARD_VISUAL_MIN + BOARD_VISUAL_MAX)
BOARD_VISUAL_SPAN = float(np.max(BOARD_VISUAL_MAX - BOARD_VISUAL_MIN))


@dataclass
class FocusPoint:
    name: str
    position_m: np.ndarray
    enabled: bool = True


@dataclass
class ReflectorState:
    left_x0: bool = False
    right_xl: bool = False
    front_y0: bool = False
    back_yl: bool = False


def _phase_to_color(phase: float) -> str:
    if not np.isfinite(phase):
        return "#2c2c2c"
    hue = (phase % (2.0 * math.pi)) / (2.0 * math.pi)
    sector = int(hue * 6.0) % 6
    f = hue * 6.0 - sector
    value = 0.95
    chroma = 0.82
    x = chroma * (1.0 - abs(f % 2.0 - 1.0))
    if sector == 0:
        r, g, b = chroma, x, 0.0
    elif sector == 1:
        r, g, b = x, chroma, 0.0
    elif sector == 2:
        r, g, b = 0.0, chroma, x
    elif sector == 3:
        r, g, b = 0.0, x, chroma
    elif sector == 4:
        r, g, b = x, 0.0, chroma
    else:
        r, g, b = chroma, 0.0, x
    m = value - chroma
    return f"#{int((r + m) * 255):02x}{int((g + m) * 255):02x}{int((b + m) * 255):02x}"


class RealtimeFocusGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SonicSurface Realtime Focus")
        self.root.geometry("1180x760")
        self.root.minsize(1040, 680)

        self.focuses = [FocusPoint("Focus 1", DEFAULT_FOCUS.copy())]
        self.selected_index = 0
        self.reflectors = ReflectorState()
        self.current_phases = np.full(CHANNELS, np.nan, dtype=np.float64)
        self.phase_source = "bem"
        self.bem_running = False
        self.bem_pending = True
        self.bem_generation = 0
        self.bem_after_id: str | None = None
        self.bem_last_seconds: float | None = None
        self.last_valid_bem_phases: np.ndarray | None = None
        self.envelope_vertices: np.ndarray | None = None
        self.envelope_faces: np.ndarray | None = None
        self.direct_envelope_vertices: np.ndarray | None = None
        self.direct_envelope_faces: np.ndarray | None = None
        self.envelope_grid: np.ndarray | None = None
        self.envelope_amplitude: np.ndarray | None = None
        self.direct_envelope_amplitude: np.ndarray | None = None
        self.envelope_running = False
        self.last_upload_s = 0.0
        self.server_process: subprocess.Popen[str] | None = None
        self.continuous_upload = False
        self.request_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.status: dict[str, Any] | None = None
        self.dragging_focus = False
        self.dragging_camera = False
        self.panning_camera = False
        self.last_mouse_xy = (0, 0)
        self.camera_yaw = -0.72
        self.camera_pitch = 0.58
        self.camera_zoom = 1.0
        self.camera_pan = np.zeros(2, dtype=np.float64)

        self.service_url_var = tk.StringVar(value=DEFAULT_SERVICE_URL)
        self.config_path_var = tk.StringVar(value=str(DEFAULT_CONFIG))
        self.connection_text_var = tk.StringVar(value="服务未连接")
        self.upload_text_var = tk.StringVar(value="相位已就绪")
        self.envelope_text_var = tk.StringVar(value="包络面未计算")
        self.envelope_threshold_var = tk.DoubleVar(value=80.0)
        self.envelope_threshold_text_var = tk.StringVar(value="阈值 80%")
        self.coord_text_var = tk.StringVar(value="")
        self.step_mm_var = tk.DoubleVar(value=DEFAULT_STEP_M * 1000.0)
        self.upload_hz_var = tk.DoubleVar(value=DEFAULT_UPLOAD_HZ)
        self.reflector_vars = {
            "left_x0": tk.BooleanVar(value=False),
            "right_xl": tk.BooleanVar(value=False),
            "front_y0": tk.BooleanVar(value=False),
            "back_yl": tk.BooleanVar(value=False),
        }

        self._build_ui()
        self._bind_keys()
        self._refresh_focus_list()
        self._recompute_and_draw()
        self._schedule_status_poll()
        self._schedule_queue_drain()
        self._schedule_continuous_upload()
        self._mark_bem_dirty()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6f7")
        style.configure("Panel.TFrame", background="#ffffff", relief="flat")
        style.configure("TLabel", background="#ffffff", foreground="#202326", font=("Segoe UI", 9))
        style.configure("Title.TLabel", background="#ffffff", foreground="#151719", font=("Segoe UI", 13, "bold"))
        style.configure("Status.TLabel", background="#202326", foreground="#ffffff", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9), padding=(10, 6))
        style.configure("Accent.TButton", font=("Segoe UI", 9, "bold"), padding=(10, 6))

        shell = ttk.Frame(self.root, padding=12)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(0, weight=1)
        shell.columnconfigure(1, weight=0)
        shell.rowconfigure(0, weight=1)

        visual_panel = ttk.Frame(shell, style="Panel.TFrame", padding=10)
        visual_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        visual_panel.rowconfigure(0, weight=1)
        visual_panel.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            visual_panel,
            background="#eef2f1",
            highlightthickness=0,
            bd=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._draw_scene())
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_press)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_camera_release)
        self.canvas.bind("<ButtonPress-3>", self._on_camera_press)
        self.canvas.bind("<B3-Motion>", self._on_camera_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_camera_release)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda _event: self._zoom_camera(1.08))
        self.canvas.bind("<Button-5>", lambda _event: self._zoom_camera(1.0 / 1.08))

        side = ttk.Frame(shell, style="Panel.TFrame", padding=12)
        side.grid(row=0, column=1, sticky="ns")
        side.columnconfigure(0, weight=1)

        ttk.Label(side, text="实时焦点控制", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            side,
            text="WASD 平移，Q/E 调高度。左键拖红点，右键旋转视角，滚轮缩放。",
            wraplength=310,
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        self.focus_list = tk.Listbox(side, height=5, exportselection=False, activestyle="dotbox")
        self.focus_list.grid(row=2, column=0, sticky="ew")
        self.focus_list.bind("<<ListboxSelect>>", self._on_focus_select)

        focus_buttons = ttk.Frame(side, style="Panel.TFrame")
        focus_buttons.grid(row=3, column=0, sticky="ew", pady=(8, 12))
        focus_buttons.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(focus_buttons, text="新增", command=self._add_focus).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(focus_buttons, text="启停", command=self._toggle_focus).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(focus_buttons, text="删除", command=self._remove_focus).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        coord_frame = ttk.Frame(side, style="Panel.TFrame")
        coord_frame.grid(row=4, column=0, sticky="ew", pady=(0, 12))
        coord_frame.columnconfigure(1, weight=1)
        ttk.Label(coord_frame, textvariable=self.coord_text_var).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Spinbox(coord_frame, from_=0.1, to=20.0, increment=0.5, textvariable=self.step_mm_var, width=8).grid(
            row=1,
            column=1,
            sticky="e",
            pady=(8, 0),
        )
        ttk.Label(coord_frame, text="步长 mm").grid(row=1, column=0, sticky="w", pady=(8, 0))

        reflector_box = ttk.LabelFrame(side, text="反射面", padding=10)
        reflector_box.grid(row=5, column=0, sticky="ew", pady=(0, 12))
        reflector_box.columnconfigure(0, weight=1)
        labels = [
            ("left_x0", "左侧 x=0"),
            ("right_xl", "右侧 x=L"),
            ("front_y0", "前侧 y=0"),
            ("back_yl", "后侧 y=L"),
        ]
        for row, (key, label) in enumerate(labels):
            ttk.Checkbutton(
                reflector_box,
                text=label,
                variable=self.reflector_vars[key],
                command=lambda key=key: self._on_reflector_change(key),
            ).grid(row=row, column=0, sticky="w", pady=1)
        ttk.Label(
            reflector_box,
            text="BEM: x/y 各最多一个无限挡板",
        ).grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Label(
            reflector_box,
            text="焦点或反射面变化后自动重算 BEM 相位",
        ).grid(row=6, column=0, sticky="w", pady=(10, 0))
        envelope_buttons = ttk.Frame(reflector_box, style="Panel.TFrame")
        envelope_buttons.grid(row=7, column=0, sticky="ew", pady=(10, 0))
        envelope_buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(envelope_buttons, text="计算包络面", command=self._start_envelope_compute).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 4),
        )
        ttk.Button(envelope_buttons, text="清除包络面", command=self._clear_envelope).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(4, 0),
        )
        threshold_row = ttk.Frame(reflector_box, style="Panel.TFrame")
        threshold_row.grid(row=8, column=0, sticky="ew", pady=(8, 0))
        threshold_row.columnconfigure(0, weight=1)
        ttk.Scale(
            threshold_row,
            from_=50.0,
            to=95.0,
            orient=tk.HORIZONTAL,
            variable=self.envelope_threshold_var,
            command=self._on_envelope_threshold_change,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(threshold_row, textvariable=self.envelope_threshold_text_var, width=9).grid(
            row=0,
            column=1,
            sticky="e",
        )
        ttk.Label(reflector_box, textvariable=self.envelope_text_var, wraplength=300).grid(
            row=9,
            column=0,
            sticky="w",
            pady=(6, 0),
        )

        service_box = ttk.LabelFrame(side, text="硬件服务", padding=10)
        service_box.grid(row=6, column=0, sticky="ew", pady=(0, 12))
        service_box.columnconfigure(0, weight=1)
        ttk.Label(service_box, text="服务地址").grid(row=0, column=0, sticky="w")
        ttk.Entry(service_box, textvariable=self.service_url_var).grid(row=1, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(service_box, text="配置文件").grid(row=2, column=0, sticky="w")
        ttk.Entry(service_box, textvariable=self.config_path_var).grid(row=3, column=0, sticky="ew", pady=(2, 8))
        service_buttons = ttk.Frame(service_box, style="Panel.TFrame")
        service_buttons.grid(row=4, column=0, sticky="ew")
        service_buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(service_buttons, text="启动并连接", command=self._start_service).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(service_buttons, text="关闭服务", command=self._stop_service).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(service_box, textvariable=self.connection_text_var).grid(row=5, column=0, sticky="w", pady=(8, 0))

        upload_box = ttk.LabelFrame(side, text="相位上传", padding=10)
        upload_box.grid(row=7, column=0, sticky="ew")
        upload_box.columnconfigure(0, weight=1)
        upload_buttons = ttk.Frame(upload_box, style="Panel.TFrame")
        upload_buttons.grid(row=0, column=0, sticky="ew")
        upload_buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(upload_buttons, text="发送当前相位", style="Accent.TButton", command=self._send_current_phase).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 4),
        )
        ttk.Button(upload_buttons, text="关闭相位", command=self._send_off).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        stream_row = ttk.Frame(upload_box, style="Panel.TFrame")
        stream_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        stream_row.columnconfigure(1, weight=1)
        self.continuous_button = ttk.Button(stream_row, text="开始持续上传", command=self._toggle_continuous_upload)
        self.continuous_button.grid(row=0, column=0, sticky="w")
        ttk.Label(stream_row, text="Hz").grid(row=0, column=2, sticky="e")
        ttk.Spinbox(stream_row, from_=1.0, to=80.0, increment=1.0, textvariable=self.upload_hz_var, width=8).grid(
            row=0,
            column=1,
            sticky="e",
            padx=(8, 4),
        )
        ttk.Label(upload_box, textvariable=self.upload_text_var, wraplength=310).grid(row=2, column=0, sticky="w", pady=(8, 0))

        status_bar = ttk.Label(self.root, textvariable=self.connection_text_var, style="Status.TLabel", anchor="w", padding=(10, 4))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress-w>", lambda _event: self._move_selected(0.0, 1.0, 0.0))
        self.root.bind("<KeyPress-s>", lambda _event: self._move_selected(0.0, -1.0, 0.0))
        self.root.bind("<KeyPress-a>", lambda _event: self._move_selected(-1.0, 0.0, 0.0))
        self.root.bind("<KeyPress-d>", lambda _event: self._move_selected(1.0, 0.0, 0.0))
        self.root.bind("<KeyPress-q>", lambda _event: self._move_selected(0.0, 0.0, -1.0))
        self.root.bind("<KeyPress-e>", lambda _event: self._move_selected(0.0, 0.0, 1.0))

    def _selected_focus(self) -> FocusPoint | None:
        if not self.focuses:
            return None
        self.selected_index = max(0, min(self.selected_index, len(self.focuses) - 1))
        return self.focuses[self.selected_index]

    def _refresh_focus_list(self) -> None:
        self.focus_list.delete(0, tk.END)
        for focus in self.focuses:
            x, y, z = focus.position_m * 1000.0
            state = "on" if focus.enabled else "off"
            self.focus_list.insert(tk.END, f"{focus.name}  {state}  ({x:.1f}, {y:.1f}, {z:.1f}) mm")
        if self.focuses:
            self.focus_list.selection_clear(0, tk.END)
            self.focus_list.selection_set(self.selected_index)
            self.focus_list.activate(self.selected_index)
        self._update_coord_label()

    def _update_coord_label(self) -> None:
        focus = self._selected_focus()
        if focus is None:
            self.coord_text_var.set("没有加强点")
            return
        x, y, z = focus.position_m * 1000.0
        self.coord_text_var.set(f"当前：{focus.name}  x={x:.1f}  y={y:.1f}  z={z:.1f} mm")

    def _on_focus_select(self, _event: tk.Event[Any]) -> None:
        selection = self.focus_list.curselection()
        if selection:
            self.selected_index = int(selection[0])
            self._update_coord_label()
            self._draw_scene()

    def _add_focus(self) -> None:
        base = self._selected_focus()
        if base is None:
            pos = DEFAULT_FOCUS.copy()
        else:
            pos = base.position_m + np.asarray([0.006, 0.0, 0.0])
        pos = self._clamp_position(pos)
        self.focuses.append(FocusPoint(f"Focus {len(self.focuses) + 1}", pos))
        self.selected_index = len(self.focuses) - 1
        self._refresh_focus_list()
        self._mark_bem_dirty()

    def _toggle_focus(self) -> None:
        focus = self._selected_focus()
        if focus is None:
            return
        focus.enabled = not focus.enabled
        self._refresh_focus_list()
        self._mark_bem_dirty()

    def _remove_focus(self) -> None:
        if len(self.focuses) <= 1:
            messagebox.showinfo("保留一个加强点", "至少保留一个加强点。")
            return
        del self.focuses[self.selected_index]
        self.selected_index = max(0, self.selected_index - 1)
        self._refresh_focus_list()
        self._mark_bem_dirty()

    def _move_selected(self, dx: float, dy: float, dz: float) -> None:
        focus = self._selected_focus()
        if focus is None:
            return
        step = max(0.0001, float(self.step_mm_var.get()) / 1000.0)
        focus.position_m = self._clamp_position(focus.position_m + np.asarray([dx, dy, dz]) * step)
        self._refresh_focus_list()
        self._mark_bem_dirty()

    @staticmethod
    def _clamp_position(pos: np.ndarray) -> np.ndarray:
        out = np.asarray(pos, dtype=np.float64).copy()
        out[0] = float(np.clip(out[0], 0.0, BOARD_SIDE_M))
        out[1] = float(np.clip(out[1], 0.0, BOARD_SIDE_M))
        out[2] = float(np.clip(out[2], 0.015, 0.18))
        return out

    def _on_reflector_change(self, changed: str | None = None) -> None:
        if changed == "left_x0" and self.reflector_vars["left_x0"].get():
            self.reflector_vars["right_xl"].set(False)
        elif changed == "right_xl" and self.reflector_vars["right_xl"].get():
            self.reflector_vars["left_x0"].set(False)
        elif changed == "front_y0" and self.reflector_vars["front_y0"].get():
            self.reflector_vars["back_yl"].set(False)
        elif changed == "back_yl" and self.reflector_vars["back_yl"].get():
            self.reflector_vars["front_y0"].set(False)
        for key, var in self.reflector_vars.items():
            setattr(self.reflectors, key, bool(var.get()))
        self._mark_bem_dirty()

    def _recompute_and_draw(self) -> None:
        self._on_reflector_change_values_only()
        active_count = sum(1 for focus in self.focuses if focus.enabled)
        mode = self._mode_label()
        if self.bem_running:
            self.upload_text_var.set(f"BEM 正在后台计算：{active_count} 个加强点，模式={mode}")
        else:
            stale = "，BEM结果已过期" if self.bem_pending else ""
            self.upload_text_var.set(f"当前相位来源=BEM，模式={mode}{stale}")
        self._draw_scene()

    def _on_reflector_change_values_only(self) -> None:
        for key, var in self.reflector_vars.items():
            setattr(self.reflectors, key, bool(var.get()))

    def _mode_label(self) -> str:
        planes = self._bem_planes()
        extra = len(planes) - 1
        return f"BEM无限正交面({extra})"

    def _mark_bem_dirty(self) -> None:
        self.bem_pending = True
        self.bem_generation += 1
        self.envelope_vertices = None
        self.envelope_faces = None
        self.direct_envelope_vertices = None
        self.direct_envelope_faces = None
        self.envelope_grid = None
        self.envelope_amplitude = None
        self.direct_envelope_amplitude = None
        if not self.envelope_running:
            self.envelope_text_var.set("包络面已过期")
        self._recompute_and_draw()
        if self.bem_after_id is not None:
            self.root.after_cancel(self.bem_after_id)
        self.bem_after_id = self.root.after(200, self._start_bem_compute)

    def _active_targets(self) -> np.ndarray:
        active = [focus.position_m for focus in self.focuses if focus.enabled]
        if not active:
            raise ValueError("至少启用一个加强点")
        return np.asarray(active, dtype=np.float64)

    def _bem_planes(self) -> tuple[tuple[int, float], ...]:
        planes: list[tuple[int, float]] = [(2, 0.0)]
        if self.reflectors.left_x0:
            planes.append((0, 0.0))
        elif self.reflectors.right_xl:
            planes.append((0, BOARD_SIDE_M))
        if self.reflectors.front_y0:
            planes.append((1, 0.0))
        elif self.reflectors.back_yl:
            planes.append((1, BOARD_SIDE_M))
        return tuple(planes)

    def _bem_boundary_conditions(self) -> dict[str, str]:
        boundary = {
            "-x": "open",
            "+x": "open",
            "-y": "open",
            "+y": "open",
            "-z": "reflecting",
            "+z": "open",
        }
        if self.reflectors.left_x0:
            boundary["-x"] = "reflecting"
        elif self.reflectors.right_xl:
            boundary["+x"] = "reflecting"
        if self.reflectors.front_y0:
            boundary["-y"] = "reflecting"
        elif self.reflectors.back_yl:
            boundary["+y"] = "reflecting"
        return boundary

    def _field_image_transforms(self) -> list[tuple[np.ndarray, np.ndarray]]:
        planes = self._bem_planes()
        signs_offsets: list[tuple[np.ndarray, np.ndarray]] = []
        for mask in range(2 ** len(planes)):
            sign = np.ones(3, dtype=np.float64)
            offset = np.zeros(3, dtype=np.float64)
            for bit, (axis, position) in enumerate(planes):
                if (mask >> bit) & 1:
                    sign[axis] = -1.0
                    offset[axis] = 2.0 * float(position)
            signs_offsets.append((sign, offset))
        return signs_offsets

    def _bem_scene_raw(self, targets: np.ndarray) -> dict[str, Any]:
        return {
            "mode": "phase_optimization",
            "algorithm": "adjoint",
            "physics": {"sound_speed": SOUND_SPEED_M_S, "medium_density": 1.21},
            "transducer_specs": {
                "frequency": ULTRASOUND_HZ,
                "array_n": 16,
                "diameter": 0.009,
                "pitch": 0.0105,
                "geometry": "sonicsurface_reference",
                "reference_anchor_m": [0.08, 0.08],
                "source_radial_order": 16,
                "source_angular_order": 64,
            },
            "domain": {
                "origin_m": [0.0, 0.0, 0.0],
                "box_size": [BOARD_SIDE_M, BOARD_SIDE_M, BOARD_SIDE_M],
                "grid_size": [17, 17, 17],
            },
            "boundary_conditions": self._bem_boundary_conditions(),
            "targets": [{"point": target.tolist()} for target in targets],
            "obstacles": [],
            "solver": {
                "backend": "bem",
                "bem_rtol": 1e-2,
                "bem_source_orders": [2, 8],
                "bem_near_source_orders": [4, 16],
            },
            "training": {
                "phase_basis_file": str(REPOSITORY_ROOT / "outputs" / "realtime_gui_bem_basis.npy"),
                "load_basis_to_gpu": False,
                "show_loss_curve": False,
            },
            "io": {
                "output_file": str(REPOSITORY_ROOT / "outputs" / "realtime_gui_bem_result.npz"),
                "auto_visualize": False,
            },
        }

    def _start_bem_compute(self) -> None:
        if self.bem_running:
            self.upload_text_var.set("BEM 已在计算中；将使用最新位置重新触发。")
            return
        try:
            targets = self._active_targets()
        except ValueError as exc:
            self.upload_text_var.set(str(exc))
            return
        self._on_reflector_change_values_only()
        planes = self._bem_planes()
        generation = self.bem_generation
        self.bem_running = True
        self.upload_text_var.set(
            f"BEM 计算中：targets={len(targets)} planes={len(planes)}"
        )
        self.root.update_idletasks()

        started = time.perf_counter()
        try:
            import yaml
            src_path = str(REPOSITORY_ROOT / "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            from algorithm.config import SimulationConfig
            from algorithm.bem.scene_solver import BemSceneSolver

            raw = self._bem_scene_raw(targets)
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
                yaml.safe_dump(raw, handle, sort_keys=False)
                config_path = handle.name
            solver = BemSceneSolver(SimulationConfig.from_yaml(config_path))
            try:
                rows = solver.target_response(targets)
                combined = np.sum(np.conj(rows), axis=0)
                phases = np.mod(np.angle(combined), 2.0 * np.pi)
                surface_elements = int(solver.bem.n)
            finally:
                solver.close()
            self._handle_request_result(
                "bem",
                {
                    "generation": generation,
                    "phases": phases,
                    "seconds": time.perf_counter() - started,
                    "surface_elements": surface_elements,
                    "reflectors": len(planes) - 1,
                    "targets": len(targets),
                },
            )
        except Exception as exc:
            self._handle_request_result("bem", exc)

    def _start_envelope_compute(self) -> None:
        if self.envelope_running:
            self.envelope_text_var.set("包络面正在计算")
            return
        if self.bem_pending or not np.isfinite(self.current_phases).all():
            self.envelope_text_var.set("等待 BEM 相位更新完成后再计算包络面")
            return
        try:
            targets = self._active_targets()
        except ValueError as exc:
            self.envelope_text_var.set(str(exc))
            return

        phases = self.current_phases.copy()
        direct_phases = self._direct_distance_phases(targets)
        transforms = self._field_image_transforms()
        threshold = float(self.envelope_threshold_var.get())
        self.envelope_running = True
        self.envelope_text_var.set("包络面后台计算中")

        def worker() -> None:
            started = time.perf_counter()
            try:
                grid, amplitude = self._sample_envelope_amplitude(targets, phases, transforms)
                _grid_direct, direct_amplitude = self._sample_envelope_amplitude(targets, direct_phases, transforms)
                vertices, faces = self._envelope_mesh_from_amplitude(grid, amplitude, threshold)
                direct_vertices, direct_faces = self._envelope_mesh_from_amplitude(grid, direct_amplitude, threshold)
                self.request_queue.put(
                    (
                        "envelope",
                        {
                            "grid": grid,
                            "amplitude": amplitude,
                            "vertices": vertices,
                            "faces": faces,
                            "direct_amplitude": direct_amplitude,
                            "direct_vertices": direct_vertices,
                            "direct_faces": direct_faces,
                            "seconds": time.perf_counter() - started,
                        },
                    )
                )
            except Exception as exc:
                self.request_queue.put(("envelope", exc))

        threading.Thread(target=worker, daemon=True).start()

    def _direct_distance_phases(self, targets: np.ndarray) -> np.ndarray:
        combined = np.zeros(CHANNELS, dtype=np.complex128)
        for target in np.atleast_2d(targets):
            distances = np.linalg.norm(EMITTER_CENTERS - target[None, :], axis=1)
            distances = np.maximum(distances, 1e-6)
            response = np.exp(1j * WAVENUMBER * distances) / distances
            combined += np.conj(response)
        return np.mod(np.angle(combined), 2.0 * np.pi)

    def _sample_envelope_amplitude(
        self,
        targets: np.ndarray,
        phases: np.ndarray,
        transforms: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        center = targets.mean(axis=0)
        radius = 0.028
        samples = 31
        axes = [
            np.linspace(max(0.0, center[0] - radius), min(BOARD_SIDE_M, center[0] + radius), samples),
            np.linspace(max(0.0, center[1] - radius), min(BOARD_SIDE_M, center[1] + radius), samples),
            np.linspace(max(0.012, center[2] - radius), min(0.16, center[2] + radius), samples),
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        points = grid.reshape(-1, 3)
        drive = np.exp(1j * phases).astype(np.complex128)
        field = np.zeros(len(points), dtype=np.complex128)
        chunk = 1024
        for start in range(0, len(points), chunk):
            block = points[start:start + chunk]
            value = np.zeros(len(block), dtype=np.complex128)
            for sign, offset in transforms:
                sources = EMITTER_CENTERS * sign + offset
                r = np.linalg.norm(block[:, None, :] - sources[None, :, :], axis=2)
                r = np.maximum(r, 1e-5)
                value += (np.exp(1j * WAVENUMBER * r) / r) @ drive
            field[start:start + chunk] = value
        amplitude = np.abs(field).reshape((samples, samples, samples))
        return grid, amplitude

    def _envelope_mesh_from_amplitude(
        self,
        grid: np.ndarray,
        amplitude: np.ndarray,
        threshold_percent: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        peak = float(amplitude.max())
        if peak <= 0.0 or not np.isfinite(peak):
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
        threshold = (
            float(self.envelope_threshold_var.get())
            if threshold_percent is None
            else float(threshold_percent)
        ) / 100.0
        level = threshold * peak
        if level <= float(amplitude.min()) or level >= peak:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
        from skimage.measure import marching_cubes

        axes_x = grid[:, 0, 0, 0]
        axes_y = grid[0, :, 0, 1]
        axes_z = grid[0, 0, :, 2]
        spacing = (
            float(axes_x[1] - axes_x[0]),
            float(axes_y[1] - axes_y[0]),
            float(axes_z[1] - axes_z[0]),
        )
        vertices, faces, _normals, _values = marching_cubes(
            amplitude.astype(np.float32),
            level=float(level),
            spacing=spacing,
        )
        vertices = vertices + np.asarray([axes_x[0], axes_y[0], axes_z[0]], dtype=np.float64)
        faces = faces.astype(np.int64)
        if len(faces) > 1400:
            order = np.linspace(0, len(faces) - 1, 1400).astype(np.int64)
            faces = faces[order]
        return np.asarray(vertices, dtype=np.float64), faces

    def _on_envelope_threshold_change(self, _value: str | None = None) -> None:
        threshold = float(self.envelope_threshold_var.get())
        self.envelope_threshold_text_var.set(f"阈值 {threshold:.0f}%")
        if self.envelope_grid is None or self.envelope_amplitude is None:
            return
        self.envelope_vertices, self.envelope_faces = self._envelope_mesh_from_amplitude(
            self.envelope_grid,
            self.envelope_amplitude,
        )
        if self.direct_envelope_amplitude is not None:
            self.direct_envelope_vertices, self.direct_envelope_faces = self._envelope_mesh_from_amplitude(
                self.envelope_grid,
                self.direct_envelope_amplitude,
            )
        self.envelope_text_var.set(
            f"{threshold:.0f}%包络面：BEM {0 if self.envelope_faces is None else len(self.envelope_faces)} / "
            f"直距 {0 if self.direct_envelope_faces is None else len(self.direct_envelope_faces)} 个三角面"
        )
        self._draw_scene()

    def _clear_envelope(self) -> None:
        self.envelope_vertices = None
        self.envelope_faces = None
        self.direct_envelope_vertices = None
        self.direct_envelope_faces = None
        self.envelope_grid = None
        self.envelope_amplitude = None
        self.direct_envelope_amplitude = None
        self.envelope_running = False
        self.envelope_text_var.set("包络面已清除")
        self._draw_scene()

    def _scene_scale(self) -> float:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        return (
            min((width - 220.0) / BOARD_VISUAL_SPAN, (height - 180.0) / (BOARD_VISUAL_SPAN + 0.10))
            * self.camera_zoom
        )

    def _camera_center(self) -> tuple[float, float]:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        return width * 0.48 + self.camera_pan[0], height * 0.60 + self.camera_pan[1]

    def _camera_axes(self) -> tuple[np.ndarray, np.ndarray]:
        yaw = self.camera_yaw
        pitch = self.camera_pitch
        horizontal = np.asarray([math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float64)
        vertical = np.asarray(
            [
                math.sin(yaw) * math.sin(pitch),
                math.cos(yaw) * math.sin(pitch),
                -math.cos(pitch),
            ],
            dtype=np.float64,
        )
        return horizontal, vertical

    def _project(self, point: np.ndarray) -> tuple[float, float]:
        scale = self._scene_scale()
        center_x, center_y = self._camera_center()
        horizontal, vertical = self._camera_axes()
        relative = np.asarray(point, dtype=np.float64) - np.asarray(
            [BOARD_VISUAL_CENTER[0], BOARD_VISUAL_CENTER[1], 0.0],
            dtype=np.float64,
        )
        sx = center_x + float(np.dot(relative, horizontal)) * scale
        sy = center_y + float(np.dot(relative, vertical)) * scale
        return sx, sy

    def _draw_scene(self) -> None:
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.create_rectangle(0, 0, width, height, fill="#eef2f1", outline="")
        self.canvas.create_text(24, 24, anchor="w", text="SonicSurface 实时相位场", fill="#1e2527", font=("Segoe UI", 16, "bold"))
        self.canvas.create_text(
            24,
            50,
            anchor="w",
            text=f"{self._mode_label()} / 显示=实际相位 / {self._phase_display_state()}",
            fill="#536062",
            font=("Segoe UI", 10),
        )

        corners = [
            np.asarray([BOARD_VISUAL_MIN[0], BOARD_VISUAL_MIN[1], 0.0]),
            np.asarray([BOARD_VISUAL_MAX[0], BOARD_VISUAL_MIN[1], 0.0]),
            np.asarray([BOARD_VISUAL_MAX[0], BOARD_VISUAL_MAX[1], 0.0]),
            np.asarray([BOARD_VISUAL_MIN[0], BOARD_VISUAL_MAX[1], 0.0]),
        ]
        projected = [self._project(corner) for corner in corners]
        self.canvas.create_polygon(*sum(projected, ()), fill="#d7dfdc", outline="#7f8b8b", width=2)

        self._draw_reflectors()
        self._draw_envelope()
        self._draw_emitters()
        self._draw_focuses()
        self._draw_phase_legend(width, height)

    def _draw_reflectors(self) -> None:
        planes = []
        h = 0.12
        if self.reflectors.left_x0:
            planes.append(
                [
                    np.asarray([0.0, BOARD_VISUAL_MIN[1], 0.0]),
                    np.asarray([0.0, BOARD_VISUAL_MAX[1], 0.0]),
                    np.asarray([0.0, BOARD_VISUAL_MAX[1], h]),
                    np.asarray([0.0, BOARD_VISUAL_MIN[1], h]),
                ]
            )
        if self.reflectors.right_xl:
            planes.append(
                [
                    np.asarray([BOARD_SIDE_M, BOARD_VISUAL_MIN[1], 0.0]),
                    np.asarray([BOARD_SIDE_M, BOARD_VISUAL_MAX[1], 0.0]),
                    np.asarray([BOARD_SIDE_M, BOARD_VISUAL_MAX[1], h]),
                    np.asarray([BOARD_SIDE_M, BOARD_VISUAL_MIN[1], h]),
                ]
            )
        if self.reflectors.front_y0:
            planes.append(
                [
                    np.asarray([BOARD_VISUAL_MIN[0], 0.0, 0.0]),
                    np.asarray([BOARD_VISUAL_MAX[0], 0.0, 0.0]),
                    np.asarray([BOARD_VISUAL_MAX[0], 0.0, h]),
                    np.asarray([BOARD_VISUAL_MIN[0], 0.0, h]),
                ]
            )
        if self.reflectors.back_yl:
            planes.append(
                [
                    np.asarray([BOARD_VISUAL_MIN[0], BOARD_SIDE_M, 0.0]),
                    np.asarray([BOARD_VISUAL_MAX[0], BOARD_SIDE_M, 0.0]),
                    np.asarray([BOARD_VISUAL_MAX[0], BOARD_SIDE_M, h]),
                    np.asarray([BOARD_VISUAL_MIN[0], BOARD_SIDE_M, h]),
                ]
            )
        for plane in planes:
            pts = [self._project(point) for point in plane]
            self.canvas.create_polygon(
                *sum(pts, ()),
                fill="#c4d4dc",
                outline="#4b748c",
                width=2,
                stipple="gray50",
            )

    def _phase_display_state(self) -> str:
        if self.bem_running:
            return "正在计算，暂显示上一帧"
        if self.bem_pending:
            return "等待重算，暂显示上一帧"
        if not np.isfinite(self.current_phases).all():
            return "尚无有效相位"
        return "已更新"

    def _draw_envelope(self) -> None:
        if (
            (self.envelope_vertices is None or self.envelope_faces is None or len(self.envelope_faces) == 0)
            and (
                self.direct_envelope_vertices is None
                or self.direct_envelope_faces is None
                or len(self.direct_envelope_faces) == 0
            )
        ):
            return

        self._draw_envelope_mesh(
            self.direct_envelope_vertices,
            self.direct_envelope_faces,
            fill="#ff4a4a",
            outline="#8e1f1f",
        )
        self._draw_envelope_mesh(
            self.envelope_vertices,
            self.envelope_faces,
            fill="#ffffff",
            outline="#5f7f91",
        )

    def _draw_envelope_mesh(
        self,
        vertices: np.ndarray | None,
        faces: np.ndarray | None,
        fill: str,
        outline: str,
    ) -> None:
        if vertices is None or faces is None or len(faces) == 0:
            return
        horizontal, vertical = self._camera_axes()
        view_dir = np.cross(horizontal, vertical)
        relative = vertices - np.asarray(
            [BOARD_VISUAL_CENTER[0], BOARD_VISUAL_CENTER[1], 0.0],
            dtype=np.float64,
        )
        vertex_depth = relative @ view_dir
        face_depth = vertex_depth[faces].mean(axis=1)
        for face_index in np.argsort(face_depth):
            triangle = vertices[faces[face_index]]
            points_2d = [self._project(point) for point in triangle]
            self.canvas.create_polygon(
                *sum(points_2d, ()),
                fill=fill,
                outline=outline,
                width=0.5,
                stipple="gray50",
            )

    def _draw_emitters(self) -> None:
        radius = max(3.0, min(self.canvas.winfo_width(), self.canvas.winfo_height()) / 165.0)
        for index, center in enumerate(EMITTER_CENTERS):
            sx, sy = self._project(center)
            fill = _phase_to_color(float(self.current_phases[index]))
            self.canvas.create_oval(
                sx - radius,
                sy - radius,
                sx + radius,
                sy + radius,
                fill=fill,
                outline="#273033",
                width=0.5,
            )

    def _draw_focuses(self) -> None:
        for index, focus in enumerate(self.focuses):
            sx, sy = self._project(focus.position_m)
            color = "#d92121" if focus.enabled else "#8c8c8c"
            outline = "#ffffff" if index == self.selected_index else "#532222"
            self.canvas.create_line(sx, sy, sx, self._project(np.asarray([focus.position_m[0], focus.position_m[1], 0.0]))[1], fill="#a0a8aa", dash=(3, 4))
            self.canvas.create_oval(sx - 8, sy - 8, sx + 8, sy + 8, fill=color, outline=outline, width=2)
            self.canvas.create_text(sx + 12, sy - 12, anchor="w", text=focus.name, fill="#222629", font=("Segoe UI", 9, "bold"))

    def _draw_phase_legend(self, width: int, height: int) -> None:
        x0 = width - 210
        y0 = height - 48
        steps = 80
        for i in range(steps):
            phase = 2.0 * math.pi * i / max(1, steps - 1)
            color = _phase_to_color(phase)
            self.canvas.create_rectangle(x0 + i * 2, y0, x0 + i * 2 + 2, y0 + 12, fill=color, outline="")
        self.canvas.create_text(x0, y0 - 8, anchor="w", text="actual phase, cyclic", fill="#536062", font=("Segoe UI", 9))
        self.canvas.create_text(x0, y0 + 18, anchor="w", text="0", fill="#536062", font=("Segoe UI", 8))
        self.canvas.create_text(x0 + steps * 2, y0 + 18, anchor="e", text="2π ≡ 0", fill="#536062", font=("Segoe UI", 8))
        self.canvas.create_text(x0, y0 + 36, anchor="w", text="envelope: white=BEM  red=direct distance", fill="#536062", font=("Segoe UI", 8))

    def _canvas_to_board_xy(self, sx: float, sy: float, z: float) -> np.ndarray | None:
        scale = self._scene_scale()
        center_x, center_y = self._camera_center()
        horizontal, vertical = self._camera_axes()
        screen = np.asarray([(sx - center_x) / scale, (sy - center_y) / scale], dtype=np.float64)
        rhs = screen - np.asarray([horizontal[2] * z, vertical[2] * z])
        matrix = np.asarray(
            [
                [horizontal[0], horizontal[1]],
                [vertical[0], vertical[1]],
            ],
            dtype=np.float64,
        )
        try:
            dx, dy = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            return None
        x = dx + BOARD_VISUAL_CENTER[0]
        y = dy + BOARD_VISUAL_CENTER[1]
        if (
            BOARD_VISUAL_MIN[0] - 0.01 <= x <= BOARD_VISUAL_MAX[0] + 0.01
            and BOARD_VISUAL_MIN[1] - 0.01 <= y <= BOARD_VISUAL_MAX[1] + 0.01
        ):
            return self._clamp_position(np.asarray([x, y, z], dtype=np.float64))
        return None

    def _on_canvas_press(self, event: tk.Event[Any]) -> None:
        self.last_mouse_xy = (int(event.x), int(event.y))
        if int(getattr(event, "state", 0)) & 0x0001:
            self.panning_camera = True
            self.dragging_focus = False
            return
        focus = self._selected_focus()
        if focus is None:
            return
        sx, sy = self._project(focus.position_m)
        self.dragging_focus = (event.x - sx) ** 2 + (event.y - sy) ** 2 <= 20 ** 2

    def _on_canvas_drag(self, event: tk.Event[Any]) -> None:
        if self.panning_camera:
            self._pan_to_event(event)
            return
        if not self.dragging_focus:
            return
        focus = self._selected_focus()
        if focus is None:
            return
        pos = self._canvas_to_board_xy(float(event.x), float(event.y), float(focus.position_m[2]))
        if pos is None:
            return
        focus.position_m = pos
        self._refresh_focus_list()
        self._mark_bem_dirty()

    def _on_canvas_release(self, _event: tk.Event[Any]) -> None:
        self.dragging_focus = False
        self.panning_camera = False

    def _on_camera_press(self, event: tk.Event[Any]) -> None:
        self.dragging_camera = True
        self.last_mouse_xy = (int(event.x), int(event.y))

    def _on_camera_drag(self, event: tk.Event[Any]) -> None:
        if not self.dragging_camera:
            return
        last_x, last_y = self.last_mouse_xy
        dx = int(event.x) - last_x
        dy = int(event.y) - last_y
        self.last_mouse_xy = (int(event.x), int(event.y))
        self.camera_yaw += dx * 0.008
        self.camera_pitch = float(np.clip(self.camera_pitch - dy * 0.006, 0.12, 1.35))
        self._draw_scene()

    def _on_pan_press(self, event: tk.Event[Any]) -> None:
        self.panning_camera = True
        self.last_mouse_xy = (int(event.x), int(event.y))

    def _on_pan_drag(self, event: tk.Event[Any]) -> None:
        if self.panning_camera:
            self._pan_to_event(event)

    def _pan_to_event(self, event: tk.Event[Any]) -> None:
        last_x, last_y = self.last_mouse_xy
        dx = int(event.x) - last_x
        dy = int(event.y) - last_y
        self.last_mouse_xy = (int(event.x), int(event.y))
        self.camera_pan += np.asarray([dx, dy], dtype=np.float64)
        self._draw_scene()

    def _on_camera_release(self, _event: tk.Event[Any]) -> None:
        self.dragging_camera = False
        self.panning_camera = False

    def _on_mouse_wheel(self, event: tk.Event[Any]) -> None:
        direction = 1.0 if int(event.delta) > 0 else -1.0
        self._zoom_camera(1.10 if direction > 0 else 1.0 / 1.10)

    def _zoom_camera(self, factor: float) -> None:
        self.camera_zoom = float(np.clip(self.camera_zoom * factor, 0.45, 3.0))
        self._draw_scene()

    def _service_url(self) -> str:
        return self.service_url_var.get().strip().rstrip("/")

    def _post_request(self, label: str, method: str, path: str, payload: dict[str, Any] | None = None) -> None:
        service_url = self._service_url()

        def worker() -> None:
            try:
                result = request_json(service_url, method, path, payload, timeout=1.5)
                self.request_queue.put((label, result))
            except Exception as exc:
                self.request_queue.put((label, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _send_current_phase(self) -> None:
        if self.phase_source != "bem" or self.bem_pending or not np.isfinite(self.current_phases).all():
            self.continuous_upload = False
            self.continuous_button.configure(text="开始持续上传")
            self.upload_text_var.set("BEM 相位尚未更新完成，暂不发送。")
            return
        payload = {
            "phases_rad": self.current_phases.tolist(),
            "source": f"realtime-gui:{self.phase_source}:{self._mode_label()}",
        }
        self._post_request("send", "POST", "/v1/pattern", payload)

    def _send_off(self) -> None:
        self._post_request("off", "POST", "/v1/off", {"source": "realtime-gui:off"})

    def _toggle_continuous_upload(self) -> None:
        if (
            not self.continuous_upload
            and (self.phase_source != "bem" or self.bem_pending or not np.isfinite(self.current_phases).all())
        ):
            self.upload_text_var.set("BEM 相位尚未更新完成，暂不持续上传。")
            return
        self.continuous_upload = not self.continuous_upload
        self.continuous_button.configure(text="停止持续上传" if self.continuous_upload else "开始持续上传")
        self.upload_text_var.set("持续上传已启动" if self.continuous_upload else "持续上传已停止")

    def _schedule_continuous_upload(self) -> None:
        if self.continuous_upload:
            now = time.perf_counter()
            hz = max(1.0, min(80.0, float(self.upload_hz_var.get())))
            interval = 1.0 / hz
            if now - self.last_upload_s >= interval:
                self.last_upload_s = now
                self._send_current_phase()
            delay_ms = max(CONTINUOUS_INTERVAL_MIN_MS, int(interval * 500.0))
        else:
            delay_ms = 100
        self.root.after(delay_ms, self._schedule_continuous_upload)

    def _start_service(self) -> None:
        if self.server_process is not None and self.server_process.poll() is None:
            self.upload_text_var.set("服务进程已经由 GUI 启动")
            return
        url = urlsplit(self._service_url())
        if url.scheme != "http" or not url.hostname:
            messagebox.showerror("服务地址错误", "服务地址必须类似 http://127.0.0.1:8766")
            return
        port = int(url.port or 80)
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "control" / "server.py"),
            "--config",
            self.config_path_var.get().strip(),
            "--host",
            url.hostname,
            "--port",
            str(port),
            "--live",
        ]
        try:
            self.server_process = subprocess.Popen(
                command,
                cwd=str(REPOSITORY_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            messagebox.showerror("启动失败", str(exc))
            return
        self.upload_text_var.set(f"正在启动服务：{url.hostname}:{port}")
        self.root.after(800, self._poll_status_once)

    def _stop_service(self) -> None:
        self.continuous_upload = False
        self.continuous_button.configure(text="开始持续上传")
        self._send_off()
        if self.server_process is not None and self.server_process.poll() is None:
            self.server_process.terminate()
            self.upload_text_var.set("已请求关闭 GUI 启动的服务")
        else:
            self.upload_text_var.set("已发送关闭相位；当前没有 GUI 启动的服务进程")

    def _poll_status_once(self) -> None:
        self._post_request("status", "GET", "/v1/status")

    def _schedule_status_poll(self) -> None:
        self._poll_status_once()
        self.root.after(STATUS_INTERVAL_MS, self._schedule_status_poll)

    def _schedule_queue_drain(self) -> None:
        while True:
            try:
                label, result = self.request_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_request_result(label, result)
        self.root.after(50, self._schedule_queue_drain)

    def _handle_request_result(self, label: str, result: Any) -> None:
        if isinstance(result, Exception):
            if label == "status":
                self.status = None
                self.connection_text_var.set(f"未连接：{result}")
            elif label == "bem":
                self.bem_running = False
                self.upload_text_var.set(f"BEM 计算失败：{result}")
            elif label == "envelope":
                self.envelope_running = False
                self.envelope_text_var.set(f"包络面计算失败：{result}")
            else:
                self.upload_text_var.set(f"{label} 失败：{result}")
            return
        if label == "status":
            self.status = dict(result)
            live = bool(result.get("live"))
            serial = bool(result.get("serial_connected"))
            stream = result.get("stream") or {}
            queued = stream.get("frames_queued", 0)
            written = stream.get("frames_written", 0)
            superseded = stream.get("frames_superseded", 0)
            mode = "live" if live else "dry-run"
            link = "串口已连接" if serial else "串口未连接"
            self.connection_text_var.set(
                f"服务已连接：{mode}，{link}，queued={queued} written={written} dropped={superseded}"
            )
            return
        if label == "bem":
            self.bem_running = False
            generation = int(result.get("generation", -1))
            if generation != self.bem_generation:
                self.bem_pending = True
                self.upload_text_var.set("BEM 计算完成，但焦点或反射面已改变；正在重算。")
                self._mark_bem_dirty()
                return
            phases = np.asarray(result["phases"], dtype=np.float64)
            if phases.shape != (CHANNELS,) or not np.isfinite(phases).all():
                self.bem_pending = True
                self.upload_text_var.set("BEM 返回的相位无效；请检查反射面配置。")
                return
            self.current_phases = phases
            self.phase_source = "bem"
            self.bem_pending = False
            self.bem_last_seconds = float(result.get("seconds", 0.0))
            delta_text = ""
            if self.last_valid_bem_phases is not None and self.last_valid_bem_phases.shape == phases.shape:
                delta = np.angle(np.exp(1j * (phases - self.last_valid_bem_phases)))
                delta_text = (
                    f"，相位变化均值 {np.mean(np.abs(np.rad2deg(delta))):.1f}°"
                    f"/最大 {np.max(np.abs(np.rad2deg(delta))):.1f}°"
                )
            self.last_valid_bem_phases = phases.copy()
            self.upload_text_var.set(
                "BEM 相位已更新："
                f"{result.get('targets', '?')} 个目标，"
                f"{result.get('reflectors', '?')} 个正交反射面，"
                f"{self.bem_last_seconds:.2f}s"
                f"{delta_text}"
            )
            self._draw_scene()
            return
        if label == "envelope":
            self.envelope_running = False
            self.envelope_grid = np.asarray(result.get("grid"), dtype=np.float64)
            self.envelope_amplitude = np.asarray(result.get("amplitude"), dtype=np.float64)
            self.envelope_vertices = np.asarray(result.get("vertices", []), dtype=np.float64).reshape(-1, 3)
            self.envelope_faces = np.asarray(result.get("faces", []), dtype=np.int64).reshape(-1, 3)
            self.direct_envelope_amplitude = np.asarray(result.get("direct_amplitude"), dtype=np.float64)
            self.direct_envelope_vertices = np.asarray(result.get("direct_vertices", []), dtype=np.float64).reshape(-1, 3)
            self.direct_envelope_faces = np.asarray(result.get("direct_faces", []), dtype=np.int64).reshape(-1, 3)
            threshold = float(self.envelope_threshold_var.get())
            self.envelope_text_var.set(
                f"{threshold:.0f}%包络面：BEM {len(self.envelope_faces)} / "
                f"直距 {len(self.direct_envelope_faces)} 个三角面，{float(result.get('seconds', 0.0)):.2f}s"
            )
            self._draw_scene()
            return
        if label == "send":
            transmitted = result.get("transmitted", False)
            queued = result.get("queued", False)
            self.upload_text_var.set(f"当前相位已提交 transmitted={transmitted} queued={queued}")
        elif label == "off":
            self.upload_text_var.set("已发送关闭相位")
        else:
            self.upload_text_var.set(f"{label}: {result}")

    def close(self) -> None:
        self.continuous_upload = False
        if self.server_process is not None and self.server_process.poll() is None:
            try:
                self._send_off()
                self.server_process.terminate()
            except Exception:
                pass


def main() -> None:
    root = tk.Tk()
    app = RealtimeFocusGui(root)

    def on_close() -> None:
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
