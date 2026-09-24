"""Run the Sapien simulator using joint commands received through ROS.

Run from Teleop/:
    python sim_node.py --teleop_args args/teleop_args.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from control import SimSapien as Sim
from utils.load_args import _load_yaml


class SimulatorSystem:
    """Synchronise Sapien simulator with incoming joint commands."""

    def __init__(self, teleop_args_path: str | Path) -> None:
        # ------------------------------------------------------------------
        # Load configuration
        self.teleop_args = _load_yaml(Path(teleop_args_path))
        robot_name = f"{self.teleop_args['robot_arm']}_{self.teleop_args['robot_hand']}"

        # ------------------------------------------------------------------
        # Initialize ROS first (needed for logging)
        rospy.init_node("simulator_system", anonymous=True)
        
        # ------------------------------------------------------------------
        # Try to load assembly (arm + hand) URDF first, fallback to hand-only URDF
        arm_name = self.teleop_args['robot_arm'].lower()
        hand_name = self.teleop_args['robot_hand'].lower()
        
        # Check for assembly URDF
        assembly_urdf = Path(
            f"./assets/robots/assembly/{arm_name}_{hand_name}/"
            f"{arm_name}_{hand_name}_right_hand.urdf"
        )
        
        # Fallback to hand-only URDF
        hand_only_urdf = Path(
            f"./assets/robots/hands/{hand_name}_hand/"
            f"{hand_name}_hand_right.urdf"
        )
        
        # Use assembly URDF if it exists, otherwise use hand-only URDF
        if assembly_urdf.exists():
            urdf_path = str(assembly_urdf)
            rospy.loginfo(f"[sim] Using assembly URDF: {urdf_path}")
        elif hand_only_urdf.exists():
            urdf_path = str(hand_only_urdf)
            rospy.loginfo(f"[sim] Using hand-only URDF: {urdf_path}")
        else:
            raise FileNotFoundError(
                f"URDF file not found. Checked:\n"
                f"  - {assembly_urdf}\n"
                f"  - {hand_only_urdf}"
            )
        
        self.sim = Sim(urdf_path)
        
        self.sim_joint_names: List[str] = self.sim.sapien_joint_names

        # Joint command buffer and mapping indices
        self._cmd: np.ndarray = np.zeros(len(self.sim_joint_names))
        self._teleop2sim: Optional[np.ndarray] = None  # lazy lookup

        self._init_ros()

    # -------------------------------------------------------------- ROS
    def _init_ros(self) -> None:
        rospy.Subscriber(
            self.teleop_args["joint_command_topic"],
            JointState,
            self._joint_angle_cb,
            queue_size=1,
        )
        rospy.loginfo("[sim] ROS node initialised")

    def _joint_angle_cb(self, msg: JointState) -> None:
        if self._teleop2sim is None:
            self._teleop2sim = np.array(
                [msg.name.index(n) for n in self.sim_joint_names], dtype=int
            )
            rospy.loginfo_once(f"Mapping tele‑op → sim: {self._teleop2sim.tolist()}")

        teleop_pos = np.array(msg.position)
        self._cmd = teleop_pos[self._teleop2sim]

    # ------------------------------------------------------------- Main
    def run(self) -> None:
        rospy.loginfo("[sim] Ready — spinning …")
        # Sapien internally runs at its own fixed‑step; we simply drive it
        # with the most recent joint command each tick until ROS shutdown.
        rate = rospy.Rate(100)  # 100 Hz visual update
        while not rospy.is_shutdown():
            # Render / physics step & capture cameras if needed
            _ = self.sim.step(self._cmd)  # (left_img, right_img)
            rate.sleep()
        rospy.loginfo("[sim] Shutdown complete")


# ------------------------------------------------------------------------
# CLI entry‑point
# ------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Sapien simulator ROS node")
    p.add_argument(
        "--teleop_args",
        type=str,
        help="Path to teleop_args YAML configuration file",
        default="./args/teleop_args.yaml",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sim_system = SimulatorSystem(args.teleop_args)
    try:
        sim_system.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
