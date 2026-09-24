#!/usr/bin/env python3
"""Read and print the current UR5e robot state."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
UR5E_DIR = SCRIPT_DIR.parent

if str(UR5E_DIR) not in sys.path:
    sys.path.insert(0, str(UR5E_DIR))

from ur5e_node import UR5eNode


# Joint names.
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def format_array(arr: np.ndarray, precision: int = 4) -> str:
    """Format an array as a readable string."""
    return "[" + ", ".join(f"{x:.{precision}f}" for x in arr) + "]"


def format_pose(pose: np.ndarray, precision: int = 4) -> str:
    """Format a pose as a readable string."""
    if len(pose) == 6:
        return f"Position: [{pose[0]:.{precision}f}, {pose[1]:.{precision}f}, {pose[2]:.{precision}f}], " \
               f"Orientation: [{pose[3]:.{precision}f}, {pose[4]:.{precision}f}, {pose[5]:.{precision}f}]"
    return format_array(pose, precision)


def print_section(title: str, width: int = 80):
    """Print a section heading."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title: str):
    """Print a subsection heading."""
    print(f"\n--- {title} ---")


def read_all_robot_info(robot_ip: str) -> None:
    """Read and print all available robot information."""
    print(f"\nConnecting to the UR5e robot: {robot_ip}")
    print("Reading robot state...\n")
    arm_node: UR5eNode | None = None

    try:
        # Create the UR5eNode instance.
        arm_node = UR5eNode(robot_ip=robot_ip, rtde_frequency=125)

        if not arm_node.rtde_r.isConnected():
            print(f"Error: Could not connect to the UR5e robot ({robot_ip})")
            return

        print("✓ Connected successfully!\n")

        # Read basic feedback.
        feedback = arm_node.get_feedback()
        rtde_r = arm_node.rtde_r

        # 1. Joint information.
        print_section("Joint state")
        
        # Actual joint positions.
        joint_positions = feedback["joint_positions"]
        print_subsection("Actual joint positions (radians)")
        for i, (name, pos) in enumerate(zip(JOINT_NAMES, joint_positions)):
            print(f"  {name:25s}: {pos:8.4f} rad ({np.degrees(pos):7.2f}°)")
        
        # Actual joint velocities.
        joint_velocities = feedback["joint_velocities"]
        print_subsection("Actual joint velocities (rad/s)")
        for i, (name, vel) in enumerate(zip(JOINT_NAMES, joint_velocities)):
            print(f"  {name:25s}: {vel:8.4f} rad/s")
        
        # Joint currents.
        joint_current = feedback["joint_current"]
        print_subsection("Joint currents (A)")
        for i, (name, cur) in enumerate(zip(JOINT_NAMES, joint_current)):
            print(f"  {name:25s}: {cur:8.4f} A")
        
        # Try to read target joint positions.
        try:
            target_q = np.asarray(rtde_r.getTargetQ(), dtype=np.float64)
            print_subsection("Target joint positions (radians)")
            for i, (name, pos) in enumerate(zip(JOINT_NAMES, target_q)):
                print(f"  {name:25s}: {pos:8.4f} rad ({np.degrees(pos):7.2f}°)")
        except Exception as e:
            print_subsection("Target joint positions")
            print(f"  Could not read: {e}")
        
        # Try to read target joint velocities.
        try:
            target_qd = np.asarray(rtde_r.getTargetQd(), dtype=np.float64)
            print_subsection("Target joint velocities (rad/s)")
            for i, (name, vel) in enumerate(zip(JOINT_NAMES, target_qd)):
                print(f"  {name:25s}: {vel:8.4f} rad/s")
        except Exception as e:
            print_subsection("Target joint velocities")
            print(f"  Could not read: {e}")
        
        # Try to read joint torques.
        try:
            joint_torques = np.asarray(rtde_r.getActualTorque(), dtype=np.float64)
            print_subsection("Joint torques (Nm)")
            for i, (name, torque) in enumerate(zip(JOINT_NAMES, joint_torques)):
                print(f"  {name:25s}: {torque:8.4f} Nm")
        except Exception as e:
            print_subsection("Joint torques")
            print(f"  Could not read: {e}")
        
        # 2. TCP and end-effector information.
        print_section("TCP/end-effector state")
        
        # Actual TCP pose.
        tcp_pose = feedback["tcp_pose"]
        print_subsection("Actual TCP pose")
        print(f"  {format_pose(tcp_pose)}")
        print(f"  Position (x, y, z): [{tcp_pose[0]:.4f}, {tcp_pose[1]:.4f}, {tcp_pose[2]:.4f}] m")
        print(f"  Orientation (rx, ry, rz): [{tcp_pose[3]:.4f}, {tcp_pose[4]:.4f}, {tcp_pose[5]:.4f}] rad")
        
        # TCP speed.
        eef_speed = feedback["eef_speed"]
        print_subsection("Actual TCP velocity")
        print(f"  Linear velocity (vx, vy, vz): [{eef_speed[0]:.4f}, {eef_speed[1]:.4f}, {eef_speed[2]:.4f}] m/s")
        print(f"  Angular velocity (wx, wy, wz): [{eef_speed[3]:.4f}, {eef_speed[4]:.4f}, {eef_speed[5]:.4f}] rad/s")
        
        # Try to read the target TCP pose.
        try:
            target_tcp_pose = np.asarray(rtde_r.getTargetTCPPose(), dtype=np.float64)
            print_subsection("Target TCP pose")
            print(f"  {format_pose(target_tcp_pose)}")
        except Exception as e:
            print_subsection("Target TCP pose")
            print(f"  Could not read: {e}")
        
        # Try to read target TCP speed.
        try:
            target_tcp_speed = np.asarray(rtde_r.getTargetTCPSpeed(), dtype=np.float64)
            print_subsection("Target TCP velocity")
            print(f"  Linear velocity (vx, vy, vz): [{target_tcp_speed[0]:.4f}, {target_tcp_speed[1]:.4f}, {target_tcp_speed[2]:.4f}] m/s")
            print(f"  Angular velocity (wx, wy, wz): [{target_tcp_speed[3]:.4f}, {target_tcp_speed[4]:.4f}, {target_tcp_speed[5]:.4f}] rad/s")
        except Exception as e:
            print_subsection("Target TCP velocity")
            print(f"  Could not read: {e}")
        
        # Try to read TCP force.
        try:
            tcp_force = np.asarray(rtde_r.getActualTCPForce(), dtype=np.float64)
            print_subsection("TCP Force/Torque")
            print(f"  Force (Fx, Fy, Fz): [{tcp_force[0]:.4f}, {tcp_force[1]:.4f}, {tcp_force[2]:.4f}] N")
            print(f"  Torque (Mx, My, Mz): [{tcp_force[3]:.4f}, {tcp_force[4]:.4f}, {tcp_force[5]:.4f}] Nm")
        except Exception as e:
            print_subsection("TCP Force/Torque")
            print(f"  Could not read: {e}")
        
        # 3. Robot status.
        print_section("Robot system state")
        
        # Connection status.
        print_subsection("Connection status")
        print(f"  RTDE Connection status: {'connected' if rtde_r.isConnected() else 'disconnected'}")
        
        # Try to read the safety mode.
        try:
            safety_mode = rtde_r.getSafetyMode()
            print_subsection("Safety mode")
            safety_modes = {
                1: "NORMAL",
                2: "REDUCED",
                3: "PROTECTIVE_STOP",
                4: "RECOVERY",
                5: "SAFEGUARD_STOP",
                6: "SYSTEM_EMERGENCY_STOP",
                7: "ROBOT_EMERGENCY_STOP",
                8: "VIOLATION",
                9: "FAULT",
                10: "AUTOMATIC_MODE_SAFEGUARD_STOP",
                11: "SYSTEM_THREE_POSITION_ENABLING_STOP"
            }
            mode_name = safety_modes.get(safety_mode, f"Unknown mode ({safety_mode})")
            print(f"  Safety mode code: {safety_mode}")
            print(f"  Safety mode name: {mode_name}")
        except Exception as e:
            print_subsection("Safety mode")
            print(f"  Could not read: {e}")
        
        # Try to read whether a program is running.
        try:
            robot_mode = rtde_r.getRobotMode()
            print_subsection("Robot mode")
            robot_modes = {
                0: "NO_CONTROLLER",
                1: "DISCONNECTED",
                2: "CONFIRM_SAFETY",
                3: "BOOTING",
                4: "POWER_OFF",
                5: "POWER_ON",
                6: "IDLE",
                7: "BACKDRIVE",
                8: "RUNNING",
                9: "UPDATING_FIRMWARE"
            }
            mode_name = robot_modes.get(robot_mode, f"Unknown mode ({robot_mode})")
            print(f"  Robot mode code: {robot_mode}")
            print(f"  Robot mode name: {mode_name}")
        except Exception as e:
            print_subsection("Robot mode")
            print(f"  Could not read: {e}")
        
        # Try to read the program state.
        try:
            program_state = rtde_r.getRobotStatus()
            print_subsection("Program state")
            print(f"  Program state: {program_state}")
        except Exception as e:
            print_subsection("Program state")
            print(f"  Could not read: {e}")
        
        # Try to read the timestamp.
        try:
            timestamp = rtde_r.getTimestamp()
            print_subsection("Timestamp")
            print(f"  Data timestamp: {timestamp:.6f} s")
        except Exception as e:
            print_subsection("Timestamp")
            print(f"  Could not read: {e}")
        
        # 4. Additional information.
        print_section("Additional information")
        
        # Try to read tool acceleration.
        try:
            tool_acc = np.asarray(rtde_r.getActualToolAccelerometer(), dtype=np.float64)
            print_subsection("Tool accelerometer")
            print(f"  Acceleration (ax, ay, az): [{tool_acc[0]:.4f}, {tool_acc[1]:.4f}, {tool_acc[2]:.4f}] m/s²")
        except Exception as e:
            print_subsection("Tool accelerometer")
            print(f"  Could not read: {e}")
        
        # Try to read the main voltage.
        try:
            voltage = rtde_r.getActualMainVoltage()
            print_subsection("Main voltage")
            print(f"  Voltage: {voltage:.2f} V")
        except Exception as e:
            print_subsection("Main voltage")
            print(f"  Could not read: {e}")
        
        # Try to read robot voltage.
        try:
            robot_voltage = rtde_r.getActualRobotVoltage()
            print_subsection("Robot voltage")
            print(f"  Voltage: {robot_voltage:.2f} V")
        except Exception as e:
            print_subsection("Robot voltage")
            print(f"  Could not read: {e}")
        
        # Try to read robot current.
        try:
            robot_current = rtde_r.getActualRobotCurrent()
            print_subsection("Robot current")
            print(f"  Current: {robot_current:.4f} A")
        except Exception as e:
            print_subsection("Robot current")
            print(f"  Could not read: {e}")
        
        # Try to read joint temperatures.
        try:
            joint_temps = np.asarray(rtde_r.getActualJointTemperatures(), dtype=np.float64)
            print_subsection("Joint temperatures (°C)")
            for i, (name, temp) in enumerate(zip(JOINT_NAMES, joint_temps)):
                print(f"  {name:25s}: {temp:6.2f} °C")
        except Exception as e:
            print_subsection("Joint temperatures")
            print(f"  Could not read: {e}")
        
        # Try to read digital inputs.
        try:
            digital_inputs = rtde_r.getActualDigitalInputBits()
            print_subsection("Digital inputs")
            print(f"  Digital input bits: {digital_inputs}")
        except Exception as e:
            print_subsection("Digital inputs")
            print(f"  Could not read: {e}")
        
        # Try to read digital outputs.
        try:
            digital_outputs = rtde_r.getActualDigitalOutputBits()
            print_subsection("Digital outputs")
            print(f"  Digital output bits: {digital_outputs}")
        except Exception as e:
            print_subsection("Digital outputs")
            print(f"  Could not read: {e}")
        
        # Try to read analog inputs.
        try:
            analog_inputs = rtde_r.getStandardAnalogInput0()
            print_subsection("Analog input 0")
            print(f"  Value: {analog_inputs:.4f}")
        except Exception as e:
            print_subsection("Analog input 0")
            print(f"  Could not read: {e}")

        print("\n" + "=" * 80)
        print("✓ Finished reading robot state")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError: Exception while reading robot state: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Always close the RTDE connection to release the interface.
        if arm_node is not None:
            arm_node.close()
            print("RTDE connection closed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read and print the current UR5e robot state"
    )
    parser.add_argument(
        "--robot_ip",
        type=str,
        default="",
        help="UR5e IP address; supply --robot_ip before connecting."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    read_all_robot_info(args.robot_ip)


if __name__ == "__main__":
    main()

