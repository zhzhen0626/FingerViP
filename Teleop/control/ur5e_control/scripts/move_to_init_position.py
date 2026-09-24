#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
UR5E_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIG = (
    UR5E_DIR.parents[1]
    / "args"
    / "robot_args"
    / "arm_args"
    / "ur5e_rapid.yaml"
)

if str(UR5E_DIR) not in sys.path:
    sys.path.insert(0, str(UR5E_DIR))

from ur5e_controller import UR5eController


def load_init_qpos(config_path: Path) -> np.ndarray:
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    init_qpos = cfg["init_qpos"]
    order = cfg.get("ur5e_command_order", init_qpos.keys())
    joints = [float(init_qpos[name]) for name in order]

    if len(joints) != 6:
        raise ValueError("UR5e expects exactly 6 joints.")

    return np.array(joints, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move the UR5e arm to the configured initial joint position using moveJ."
    )
    parser.add_argument("--robot_ip", type=str, default="")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--velocity", type=float, default=0.15)
    parser.add_argument("--acceleration", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_qpos = load_init_qpos(Path(args.config).expanduser().resolve())
    print(f"move_to_init_position: target joints {target_qpos}")

    with UR5eController(
        robot_ip=args.robot_ip,
        init_qpos=target_qpos,
        auto_reset=False,
        use_servo=False,
    ) as controller:
        controller.set_joint_positions(
            target_qpos,
            acceleration=args.acceleration,
            velocity=args.velocity,
            async_move=False,
        )
        print("move_to_init_position: moveJ command sent.")

    print("move_to_init_position: completed.")


if __name__ == "__main__":
    main()

