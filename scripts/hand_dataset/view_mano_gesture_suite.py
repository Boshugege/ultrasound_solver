"""Open the interactive BEM viewer for a selected FreiHAND/MANO gesture."""

from __future__ import annotations

import argparse
from pathlib import Path

from view_ok_hand_bem_demo import _build_plotter, _load_demo


ROOT = Path(__file__).resolve().parents[2]
GESTURES = ("ok", "open", "fist", "point")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gesture", choices=GESTURES, default="ok")
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "mano_gestures"))
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--screenshot", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_root) / args.gesture / "bem"
    data = _load_demo(output_dir)
    data["scene_name"] = f"MANO {args.gesture}"
    off_screen = bool(args.screenshot)
    plotter = _build_plotter(data, float(args.threshold), off_screen=off_screen)
    if args.screenshot:
        screenshot = Path(args.screenshot)
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(screenshot), auto_close=True)
        print(screenshot)
    else:
        plotter.show(title=f"MANO {args.gesture} BEM interactive viewer")


if __name__ == "__main__":
    main()
