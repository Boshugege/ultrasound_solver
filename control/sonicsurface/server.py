"""Local HTTP service that owns the SonicSurface serial connection."""

from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from .config import load_config
from .controller import SonicSurfaceController
from .protocol import CHANNELS


LOGGER = logging.getLogger("sonicsurface.server")
MAX_REQUEST_BYTES = 64 * 1024


def _respond(
    handler: BaseHTTPRequestHandler,
    status: HTTPStatus,
    payload: dict[str, Any],
) -> None:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _parse_phases(raw: object) -> np.ndarray:
    if not isinstance(raw, list) or len(raw) != CHANNELS:
        raise ValueError("phases_rad 必须是长度为 256 的 JSON 数组")
    normalized = [np.nan if value is None else value for value in raw]
    try:
        return np.asarray(normalized, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases_rad 只能包含数值或 null") from exc


def create_server(
    controller: SonicSurfaceController,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        # Headers and body go out in two writes; without TCP_NODELAY the second
        # can wait for the client's delayed ACK.
        disable_nagle_algorithm = True

        def log_message(self, *_: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/status":
                _respond(self, HTTPStatus.OK, controller.status())
                return
            _respond(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {
                "/v1/pattern",
                "/v1/modulation",
                "/v1/frequency-sweep",
                "/v1/off",
            }:
                _respond(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("请求体大小无效")
                body = self.rfile.read(length)
                payload = json.loads(body or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("请求体必须是 JSON 对象")
                source = str(payload.get("source", "client"))
                if self.path == "/v1/pattern":
                    result = controller.submit(
                        _parse_phases(payload.get("phases_rad")),
                        source=source,
                        wait=bool(payload.get("wait", False)),
                    )
                    # Per-frame lines would flood the console when streaming;
                    # /v1/status carries the counters.
                    LOGGER.debug(
                        "相位已接收 source=%s transmitted=%s sha256=%s",
                        source,
                        result["transmitted"],
                        result["payload_sha256"],
                    )
                elif self.path == "/v1/modulation":
                    result = controller.start_modulation(
                        _parse_phases(payload.get("phases_rad")),
                        frequency_hz=payload.get("frequency_hz"),
                        duty_cycle=payload.get("duty_cycle", 0.5),
                        duration_s=payload.get("duration_s"),
                        source=source,
                    )
                    LOGGER.info(
                        "波群调制已接收 source=%s frequency_hz=%s "
                        "duty_cycle=%s duration_s=%s transmitted=%s",
                        source,
                        result["frequency_hz"],
                        result["duty_cycle"],
                        result["duration_s"],
                        result["transmitted"],
                    )
                elif self.path == "/v1/frequency-sweep":
                    result = controller.start_frequency_sweep(
                        _parse_phases(payload.get("phases_rad")),
                        min_frequency_hz=payload.get("min_frequency_hz"),
                        max_frequency_hz=payload.get("max_frequency_hz"),
                        sweep_period_s=payload.get("sweep_period_s"),
                        duty_cycle=payload.get("duty_cycle", 0.5),
                        duration_s=payload.get("duration_s"),
                        source=source,
                    )
                    LOGGER.info(
                        "包络频率扫描已接收 source=%s range=%s..%sHz "
                        "period=%ss duty=%s duration_s=%s transmitted=%s",
                        source,
                        result["min_frequency_hz"],
                        result["max_frequency_hz"],
                        result["sweep_period_s"],
                        result["duty_cycle"],
                        result["duration_s"],
                        result["transmitted"],
                    )
                else:
                    result = controller.off(source=source)
                    LOGGER.warning(
                        "全通道关闭 source=%s transmitted=%s",
                        source,
                        result["transmitted"],
                    )
                _respond(self, HTTPStatus.OK, result)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                LOGGER.warning("拒绝请求 path=%s error=%s", self.path, exc)
                _respond(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except (IOError, OSError, RuntimeError) as exc:
                LOGGER.exception("串口发送失败")
                _respond(
                    self,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": f"串口发送失败：{exc}"},
                )

    return ThreadingHTTPServer((host, port), Handler)


def serve(
    config_path: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    live: bool = False,
) -> None:
    controller = SonicSurfaceController(load_config(config_path), live=live)
    server = create_server(controller, host=host, port=port)
    actual_host, actual_port = server.server_address[:2]
    LOGGER.info(
        "SonicSurface 服务启动 http://%s:%s live=%s",
        actual_host,
        actual_port,
        live,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("收到中断，服务停止")
    finally:
        server.server_close()
        controller.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 SonicSurface 相位控制服务。")
    parser.add_argument("--config", required=True, help="SonicSurface YAML 配置")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--live",
        action="store_true",
        help="真实打开串口；缺省为 dry-run，不访问硬件",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING"),
        help="DEBUG 会逐帧记录相位提交",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    serve(args.config, host=args.host, port=args.port, live=args.live)


if __name__ == "__main__":
    main()
