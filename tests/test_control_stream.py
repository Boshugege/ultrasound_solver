"""SonicSurface control service with a fake serial port: latest-only streaming and the client."""

from pathlib import Path
import sys
import threading
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "control"))

from sonicsurface.config import load_config  # noqa: E402
from sonicsurface.controller import SonicSurfaceController  # noqa: E402
from sonicsurface.protocol import PHASE_OFF, START_PHASES  # noqa: E402

CONFIG = ROOT / "control" / "sonicsurface.example.yaml"


class FakeSerial:
    """Records writes; a write can be held open to keep the port busy."""

    def __init__(self, *args, **kwargs):
        self.writes = []
        self.release = threading.Event()
        self.release.set()
        self.writing = threading.Event()

    def write(self, data):
        self.writing.set()
        self.release.wait(5.0)
        self.writes.append((time.perf_counter(), bytes(data)))
        return len(data)

    def flush(self):
        pass

    def close(self):
        pass


def phases(value):
    return np.full(256, value, dtype=float)


class ControlStreamTests(unittest.TestCase):
    def setUp(self):
        self.controller = SonicSurfaceController(load_config(CONFIG), live=True, serial_factory=FakeSerial)
        self.port = self.controller._serial

    def tearDown(self):
        self.port.release.set()
        self.controller.close()

    def frames(self):
        return [data for _, data in self.port.writes if data[0] == START_PHASES]

    def wait_idle(self):
        for _ in range(500):
            status = self.controller.status()["stream"]
            if not status["pending"] and status["frames_written"] + status["frames_superseded"] >= status["frames_queued"]:
                return
            time.sleep(0.01)
        self.fail("writer did not drain")

    def test_newest_pattern_replaces_one_not_yet_written(self):
        self.port.release.clear()
        self.controller.submit(phases(0.0))
        self.assertTrue(self.port.writing.wait(2.0))  # first frame occupies the port
        for value in (1.0, 2.0, 3.0):
            result = self.controller.submit(phases(value))
            self.assertTrue(result["transmitted"] and result["queued"])
        self.port.release.set()
        self.wait_idle()
        frames = self.frames()
        self.assertEqual(len(frames), 2)
        self.assertNotEqual(frames[0], frames[1])
        stream = self.controller.status()["stream"]
        self.assertEqual((stream["frames_queued"], stream["frames_written"], stream["frames_superseded"]), (4, 2, 2))

    def test_off_is_written_after_any_pattern_and_discards_queued_ones(self):
        self.port.release.clear()
        self.controller.submit(phases(0.0))
        self.assertTrue(self.port.writing.wait(2.0))
        self.controller.submit(phases(1.0))  # queued behind the busy port
        threading.Timer(0.2, self.port.release.set).start()
        self.controller.off()
        frames = self.frames()
        self.assertEqual(len(frames), 2)
        self.assertTrue(all(byte == PHASE_OFF for byte in frames[-1][1:-1]))

    def test_wait_returns_after_the_frame_is_written(self):
        result = self.controller.submit(phases(0.5), wait=True)
        self.assertTrue(result["transmitted"])
        self.assertNotIn("queued", result)
        self.assertEqual(len(self.frames()), 1)


class ClientServerTests(unittest.TestCase):
    def test_round_trip_and_errors(self):
        from sonicsurface.client import request_json
        from sonicsurface.server import create_server

        controller = SonicSurfaceController(load_config(CONFIG), live=False)  # dry-run
        server = create_server(controller, host="127.0.0.1", port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            self.assertFalse(request_json(url, "GET", "/v1/status")["live"])
            result = request_json(url, "POST", "/v1/pattern", {"phases_rad": [0.0] * 256})
            self.assertTrue(result["accepted"])
            self.assertFalse(result["transmitted"])
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                request_json(url, "POST", "/v1/pattern", {"phases_rad": [0.0]})
        finally:
            server.shutdown()
            server.server_close()
            controller.close()
        with self.assertRaisesRegex(RuntimeError, "无法连接"):
            request_json(url, "GET", "/v1/status")


if __name__ == "__main__":
    unittest.main()
