"""SonicSurface control service: configurable wait after reopening the serial port."""

from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "control"))

from sonicsurface.config import load_config  # noqa: E402
from sonicsurface.controller import SonicSurfaceController  # noqa: E402

EXAMPLE = (ROOT / "control" / "sonicsurface.example.yaml").read_text(encoding="utf-8")


def config_with(settle):
    text = EXAMPLE.replace("reopen_settle_s: 2.0", f"reopen_settle_s: {settle}")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        handle.write(text)
    try:
        return load_config(handle.name)
    finally:
        Path(handle.name).unlink()


class FailOnceSerial:
    """The first port fails its first write, like a stalled USB bridge."""

    opened = 0

    def __init__(self, *args, **kwargs):
        FailOnceSerial.opened += 1
        self.fail = FailOnceSerial.opened == 1

    def write(self, data):
        if self.fail:
            self.fail = False
            raise OSError("write timeout")
        return len(data)

    def flush(self):
        pass

    def close(self):
        pass


class ReopenSettleTests(unittest.TestCase):
    def test_default_and_validation(self):
        self.assertEqual(load_config(ROOT / "control" / "sonicsurface.example.yaml").reopen_settle_s, 2.0)
        self.assertEqual(config_with(0.2).reopen_settle_s, 0.2)
        with self.assertRaises(ValueError):
            config_with(-1)

    def test_reopen_waits_the_configured_time(self):
        FailOnceSerial.opened = 0
        controller = SonicSurfaceController(config_with(0.05), live=True, serial_factory=FailOnceSerial)
        try:
            started = time.perf_counter()
            # off() writes a complete frame synchronously (submit() may only queue it).
            self.assertTrue(controller.off()["transmitted"])
            elapsed = time.perf_counter() - started
            self.assertEqual(controller.status()["serial_recoveries"], 1)
        finally:
            controller.close()
        self.assertGreaterEqual(elapsed, 0.05)
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    unittest.main()
