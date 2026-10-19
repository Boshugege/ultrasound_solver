"""SonicSurface control service: the on/off envelope follows the requested frequency."""

from pathlib import Path
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "control"))

from sonicsurface.config import load_config  # noqa: E402
from sonicsurface.controller import SonicSurfaceController  # noqa: E402

SWAP = bytes((253,))


class RecordingSerial:
    def __init__(self, *args, **kwargs):
        self.writes = []

    def write(self, data):
        self.writes.append((time.perf_counter(), bytes(data)))
        return len(data)

    def flush(self):
        pass

    def close(self):
        pass


class EnvelopeTests(unittest.TestCase):
    def test_envelope_follows_the_requested_frequency(self):
        controller = SonicSurfaceController(load_config(ROOT / "control" / "sonicsurface.example.yaml"),
                                            live=True, serial_factory=RecordingSerial)
        try:
            controller.start_modulation(np.zeros(256), frequency_hz=200.0, duty_cycle=0.5)
            time.sleep(1.0)
            controller.off()
            swaps = np.array([t for t, data in controller._serial.writes if data == SWAP])
        finally:
            controller.close()
        gaps = np.diff(swaps) * 1e3
        # 200 Hz, 50 % duty: a swap every 2.5 ms (Event.wait timing gave ~65 Hz on Windows).
        self.assertGreater(len(swaps), 0.9 * 400)
        self.assertLess(abs(np.median(gaps) - 2.5), 0.3)


if __name__ == "__main__":
    unittest.main()
