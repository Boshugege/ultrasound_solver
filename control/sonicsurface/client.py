"""Command-line client for the local SonicSurface service."""

from __future__ import annotations

import argparse
import http.client
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .phase_source import load_phase_vector
from .protocol import CHANNELS


def request_json(
    service_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """One JSON request to the service.

    On Windows, connect() with a timeout often waits ~16 ms even on 127.0.0.1;
    with one connection per frame that capped streaming at ~70 frames/s. A local
    service is therefore connected without a timeout, which is applied only to
    the request and response.
    """
    data = (
        None
        if payload is None
        else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    )
    url = urlsplit(service_url)
    if url.scheme != "http" or not url.hostname:
        raise ValueError(f"服务地址必须是 http://主机:端口，实际为 {service_url!r}")
    local = url.hostname in {"127.0.0.1", "localhost", "::1"}
    connection = http.client.HTTPConnection(
        url.hostname, url.port or 80, timeout=None if local else timeout
    )
    try:
        connection.connect()
        connection.sock.settimeout(timeout)
        connection.request(
            method,
            url.path.rstrip("/") + path,
            body=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        response = connection.getresponse()
        text = response.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"无法连接 SonicSurface 服务：{exc}") from exc
    finally:
        connection.close()
    if response.status >= 400:
        raise RuntimeError(f"服务返回 HTTP {response.status}：{text}")
    return json.loads(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="向 SonicSurface 服务提交 256 通道相位。")
    parser.add_argument(
        "--service-url",
        default="http://127.0.0.1:8765",
        help="服务地址",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="查询服务状态")
    subparsers.add_parser("off", help="关闭全部通道")

    send = subparsers.add_parser("send", help="发送 .npy/.npz 或标准结果目录")
    send.add_argument("source", type=Path)
    send.add_argument("--npz-key", help="指定 NPZ 中的相位键")

    uniform = subparsers.add_parser("uniform", help="向所有通道发送相同相位")
    uniform.add_argument("phase_rad", type=float)

    modulate = subparsers.add_parser(
        "modulate",
        help="以可调波群频率对相位图案进行开关调制",
    )
    modulate.add_argument(
        "source",
        type=Path,
        nargs="?",
        help=".npy/.npz 或标准结果目录",
    )
    modulate.add_argument(
        "--uniform-phase",
        type=float,
        help="不读取文件，使用全阵元相同相位（rad）",
    )
    modulate.add_argument(
        "--frequency",
        type=float,
        default=200.0,
        help="波群包络频率 Hz，范围 1..500（默认 200）",
    )
    modulate.add_argument(
        "--duty",
        type=float,
        default=0.5,
        help="发射占空比，范围 0.05..0.95（默认 0.5）",
    )
    modulate.add_argument(
        "--duration",
        type=float,
        help="持续秒数；不提供时持续运行，直到 off",
    )
    modulate.add_argument("--npz-key", help="指定 NPZ 中的相位键")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "status":
            result = request_json(args.service_url, "GET", "/v1/status")
        elif args.command == "off":
            result = request_json(
                args.service_url,
                "POST",
                "/v1/off",
                {"source": "client:off"},
            )
        elif args.command == "uniform":
            if not np.isfinite(args.phase_rad):
                parser.error("phase_rad 必须是有限数值")
            phases = np.full(CHANNELS, args.phase_rad, dtype=np.float64)
            result = request_json(
                args.service_url,
                "POST",
                "/v1/pattern",
                {
                    "phases_rad": phases.tolist(),
                    "source": f"client:uniform:{args.phase_rad}",
                },
            )
        elif args.command == "send":
            phases, source, key = load_phase_vector(args.source, args.npz_key)
            source_label = str(source) if key is None else f"{source}#{key}"
            result = request_json(
                args.service_url,
                "POST",
                "/v1/pattern",
                {"phases_rad": phases.tolist(), "source": source_label},
            )
        else:
            if (args.source is None) == (args.uniform_phase is None):
                parser.error("modulate 必须指定相位来源或 --uniform-phase，且只能指定一个")
            if args.uniform_phase is not None:
                if not np.isfinite(args.uniform_phase):
                    parser.error("--uniform-phase 必须是有限数值")
                phases = np.full(CHANNELS, args.uniform_phase, dtype=np.float64)
                source_label = f"client:uniform:{args.uniform_phase}"
            else:
                phases, source, key = load_phase_vector(args.source, args.npz_key)
                source_label = str(source) if key is None else f"{source}#{key}"
            result = request_json(
                args.service_url,
                "POST",
                "/v1/modulation",
                {
                    "phases_rad": phases.tolist(),
                    "frequency_hz": args.frequency,
                    "duty_cycle": args.duty,
                    "duration_s": args.duration,
                    "source": source_label,
                },
            )
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"错误：{exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
