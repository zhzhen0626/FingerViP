"""robot_node.py
---------------------------------
High-level wrapper for FingerViP teleoperation.

This module provides an integrated control system for UR5e robotic arm
and RAPID Hand, including:
- High-frequency control (125Hz for servoJ)
- Message timeout detection
- Command deduplication
- Comprehensive data collection and publishing
- Expected input units: radians (as per UR5e RTDE standard)

USAGE
-----
$ python robot_node.py path/to/robot_args.yaml

The YAML file layout is expected to be identical to that
accepted by ``utils.load_args.load_robot_args`` in the
original project.

"""

from __future__ import annotations

import argparse
import math
import time
import select
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from base_robot_system import BaseRobotSystem
from control.rapid_hand_control.rapid_node import MotorFeedbackError

import os
import datetime
from enum import Enum
import threading

# State machine for interaction flow
class SystemState(Enum):
    """System state for multi-session data collection."""
    IDLE = "idle"  # Initial state, waiting for 't' to start teleop
    TELEOP_READY = "teleop_ready"  # Waiting for teleop messages
    TELEOP_ACTIVE = "teleop_active"  # Teleop active, waiting for 's' to start collection
    DATA_COLLECTION = "data_collection"  # Collecting data, waiting for 'q' to stop
    SAVING = "saving"  # Saving data and resetting


class RobotSystem(BaseRobotSystem):
    """Handle FingerViP teleoperation with a UR5e arm and RAPID Hand."""

    def __init__(self, robot_args_path: str | Path) -> None:
        # Initialize base class (handles robot controllers, I/O, alignment, reset)
        super().__init__(robot_args_path)
        
        # Set stdin availability
        self._stdin_available = sys.stdin.isatty()
        
        # State management (teleoperation specific)
        self.state = SystemState.IDLE
        self.teleop_active = False
        self._waiting_for_save_confirmation = False
        
        # Camera display threads
        self._camera_display_thread: Optional[threading.Thread] = None
        
        # Load display configuration from display_config (with backward compatibility)
        display_cfg = self.robot_args.get("display_config", {})
        self._show_camera_display = display_cfg.get("enable_fingertip_camera", 
                                                      self.robot_args.get("show_camera_display", False))  # Backward compatibility
        self._show_third_view_camera_display = display_cfg.get("enable_third_view_camera",
                                                                self.robot_args.get("show_third_view_camera_display", False))  # Backward compatibility

        # Start camera display
        self._start_camera_display()
        
        # Override save_data to include tv_rgb
        self.save_data = {
            "rgb": [],
            "tv_rgb": [],  # third view RealSense camera
            "timestamp": [],
            "joint_pos": [],
            "joint_vel": [],
            "joint_current": [],
            "control": [],
        }


    # ------------------------------------------------------------------ ROS
    def _init_ros(self) -> None:
        """Initialize ROS node with teleoperation subscriber."""
        super()._init_ros()
        rospy.Subscriber(
            self.robot_args["joint_command_topic"],
            JointState,
            self._joint_angle_cb,
            queue_size=1,
        )

    def _start_camera_display(self) -> None:
        """Start camera display in separate threads if enabled."""
        display_threads = []
        
        # Fingertip camera display (legacy support)
        if self._show_camera_display:
            try:
                fingertip_camera = getattr(self.rapid_hand._hand, 'fingertip_camera', None)
                if fingertip_camera is not None and hasattr(fingertip_camera, 'display'):
                    def fingertip_display_loop():
                        try:
                            # Use unified display method with exit flag for ROS compatibility
                            fingertip_camera.display(check_exit_flag=lambda: rospy.is_shutdown())
                        except Exception as exc:
                            rospy.logerr(f"[robot] Fingertip camera display error: {exc}")
                    
                    thread = threading.Thread(target=fingertip_display_loop, daemon=True)
                    thread.start()
                    display_threads.append(thread)
                    rospy.loginfo("[robot] Fingertip camera display started.")
            except Exception as exc:
                rospy.logwarn(f"[robot] Failed to access fingertip camera: {exc}")
        
        # RealSense cameras
        try:
            rapid_node = self.rapid_hand._hand
            
            # Third view camera
            if self._show_third_view_camera_display:
                if hasattr(rapid_node, 'third_view_camera') and rapid_node.third_view_camera is not None:
                    cam = rapid_node.third_view_camera
                    if hasattr(cam, 'display'):
                        def third_view_display_loop(camera=cam):
                            try:
                                camera.display(window_name="RealSense Third View Camera")
                            except Exception as exc:
                                rospy.logerr(f"[robot] Third view camera display error: {exc}")
                        
                        thread = threading.Thread(target=third_view_display_loop, daemon=True)
                        thread.start()
                        display_threads.append(thread)
                        rospy.loginfo("[robot] Third view camera display started.")
        
        except Exception as exc:
            rospy.logwarn(f"[robot] Failed to access RealSense cameras: {exc}")
        
        if display_threads:
            self._camera_display_thread = display_threads[0]  # Keep first thread reference
            rospy.loginfo(f"[robot] Camera display started ({len(display_threads)} camera(s)).")
        elif self._show_third_view_camera_display:
            rospy.logwarn("[robot] Camera display requested but no cameras available.")

    # ---------------------------------------------------------- Callbacks
    def _joint_angle_cb(self, msg: JointState) -> None:
        """Convert incoming tele‑op message into robot command arrays.
        
        Features:
        - Joint mapping extraction
        - Timestamp tracking for timeout detection
        - Expected input units: radians
        """
        # Safety check: ensure state is initialized
        if not hasattr(self, 'state'):
            return

        if getattr(self, "_save_pending", False):
            return
        
        # Process messages when teleop is active OR when waiting for first message (TELEOP_READY)
        if not self.teleop_active and self.state != SystemState.TELEOP_READY:
            return
        
        teleop_names = msg.name
        
        # Build mapping lazily on first message
        if self._teleop2arm is None:
            self._teleop2arm = np.array(
                [teleop_names.index(j) for j in self.arm_joint_order], dtype=int
            )
            rospy.loginfo(f"[robot] Arm mapping: {self._teleop2arm.tolist()}")

        if self._teleop2rapid is None:
            self._teleop2rapid = np.array(
                [teleop_names.index(j) for j in self.rapid_joint_order], dtype=int
            )
            rospy.loginfo(f"[robot] RapidHand mapping: {self._teleop2rapid.tolist()}")
        
        # Mark teleop as active when we receive first message
        if not self.teleop_active:
            # A callback already in flight must not restart a stopped session.
            if self.state != SystemState.TELEOP_READY:
                return
            self.teleop_active = True
            self.state = SystemState.TELEOP_ACTIVE
            rospy.loginfo("[robot] ✓ Teleoperation active!")

        # Extract joint positions
        teleop_pos = np.array(msg.position)
        arm_incoming = teleop_pos[self._teleop2arm]
        rapid_incoming = teleop_pos[self._teleop2rapid]
        
        # Sanity check for unreasonable values (warn but don't modify)
        # Expected units: radians
        max_val = np.max(np.abs(arm_incoming))
        if max_val > 2 * math.pi:
            rospy.logwarn_throttle(
                10.0,
                f"[robot] Warning: Arm joint angle out of expected range (max={max_val:.2f} rad). "
                "Expected values in radians, range typically [-2π, 2π]."
            )
        
        # Update command buffers
        self.arm_cmd_target = arm_incoming
        self.rapid_cmd_target = rapid_incoming
        # Update timestamp for timeout detection
        self.last_message_time = rospy.Time.now().to_sec()
        
        rospy.logdebug(f"[robot] Received - Arm: {arm_incoming}, Hand: {rapid_incoming}")

    # -------------------------------------------------------------- Helpers

    def _start_teleoperation(self) -> None:
        """Start teleoperation mode. Reset sync flags and wait for messages."""
        if getattr(self, "_save_pending", False):
            rospy.logwarn("[robot] Unsaved recording retained. Press 'q' to retry saving or discard it first.")
            return
        if self.teleop_active:
            rospy.logwarn("[robot] Teleoperation is already active.")
            return
        
        if self.state == SystemState.SAVING:
            rospy.logwarn("[robot] Cannot start teleoperation while saving.")
            return
        
        rospy.loginfo("[robot] Starting teleoperation...")
        rospy.loginfo("[robot] Waiting for /teleoperation/joint_angles to start publishing...")
        
        # Reset sync flags for new teleop session (arm only, no sync for hand)
        self.initial_sync_complete = False
        self._initial_sync_warned = False
        
        # Clear command buffers
        self.arm_cmd_target = None
        self.rapid_cmd_target = None
        self.arm_cmd = None
        self.rapid_cmd = None
        self.last_sent_arm_cmd = None
        self.last_sent_rapid_cmd = None
        self.last_message_time = None
        
        # Update state: if data collection is also active, keep DATA_COLLECTION state
        if not self.data_collection_active:
            self.state = SystemState.TELEOP_READY
        self.teleop_active = False  # Will be set to True when first message arrives
    
    def _activate_data_collection(self) -> None:
        """Start data collection after teleoperation has received its first command."""
        if getattr(self, "_save_pending", False):
            rospy.logwarn("[robot] Unsaved recording retained. Press 'q' to retry saving or discard it first.")
            return
        if not self.teleop_active:
            rospy.logwarn("[robot] Start teleoperation with 't' and wait for 'Teleoperation active' before pressing 's'.")
            return
        if self.data_collection_active:
            rospy.logwarn("[robot] Data collection is already active.")
            return
        
        if self.state == SystemState.SAVING:
            rospy.logwarn("[robot] Cannot start data collection while saving.")
            return

        self._collection_count += 1
        
        # Use existing session directory (created at node startup)
        if self._session_dir:
            if self.datasaver:
                self.datasaver.frame_dir = self._session_dir
            
            if self._zarr_enabled:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.zarr_output_path = os.path.join(self._session_dir, f"{timestamp}.zarr")
        else:
            save_cfg = self.robot_args.get("save_config", {})
            if not save_cfg.get("save_robot_data", False) and not save_cfg.get("save_robot_data_zarr", False):
                rospy.logwarn("[robot] Data saving is disabled. No data will be saved.")

        for key in self.save_data:
            self.save_data[key].clear()

        self.data_collection_active = True
        self.state = SystemState.DATA_COLLECTION
        
        rospy.loginfo(f"[robot] ✓ Data collection started (Collection #{self._collection_count}).")
        if self._session_dir:
            rospy.loginfo(f"[robot] Data will be saved to: {self._session_dir}")
        rospy.loginfo("[robot] Press 'q' to stop data collection and save.")
    
    def _ask_save_confirmation(self) -> bool:
        """Ask user whether to save collected data.
        
        Returns:
            bool: True if user wants to save, False otherwise.
        """
        # Check if there's any data to save
        if not self.save_data["rgb"]:
            rospy.loginfo("[robot] No data collected, skipping save.")
            return False

        if not self._stdin_available:
            # If stdin is not available, default to saving
            rospy.logwarn("[robot] Cannot prompt for confirmation (stdin not available). Saving by default.")
            return True
        
        # Use print() instead of rospy.loginfo() for prompt (rospy.loginfo doesn't support end parameter)
        print("[robot] Save collected data? (y/n): ", end="", flush=True)
        
        # Wait for user input (blocking)
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    rospy.logwarn("[robot] Input closed. Not saving.")
                    return False
                user_input = line.strip().lower()
                if user_input == "y":
                    return True
                elif user_input == "n":
                    return False
                else:
                    print("[robot] Please enter 'y' or 'n': ", end="", flush=True)
            except (EOFError, KeyboardInterrupt):
                rospy.logwarn("[robot] Input interrupted. Not saving.")
                return False
    
    def _stop_and_reset_impl(self) -> None:
        """
        Stop current operation (data collection and/or teleoperation) and reset.
        
        Handles all scenarios:
        1. Only data collection active: Ask to save, then reset
        2. Only teleoperation active: Stop teleoperation, then reset
        3. Both active: Ask to save, stop both, then reset
        """
        # Check if there's anything to stop
        if not self.data_collection_active and not self.teleop_active and not getattr(self, "_save_pending", False):
            rospy.logwarn("[robot] Nothing to stop.")
            return
        
        if self.state == SystemState.SAVING:
            rospy.logwarn("[robot] Already saving, please wait...")
            return
        
        # Determine what we're stopping
        is_data_collection = self.data_collection_active or getattr(self, "_save_pending", False)
        is_teleop = self.teleop_active
        
        # Log what we're stopping
        if is_data_collection and is_teleop:
            rospy.loginfo("[robot] Stopping data collection and teleoperation...")
        elif is_data_collection:
            rospy.loginfo("[robot] Stopping data collection...")
        elif is_teleop:
            rospy.loginfo("[robot] Stopping teleoperation...")
        
        # Stop teleoperation control first (prevents new servoJ commands)
        # Critical: Stop servoJ control before calling reset() to avoid control conflict
        if is_teleop:
            rospy.loginfo("[robot] Stopping servoJ control...")
            try:
                self.arm.stop_servo()
                # Brief wait to ensure servoJ fully releases control
                time.sleep(0.1)
                rospy.logdebug("[robot] servoJ control released.")
            except Exception as exc:
                rospy.logwarn(f"[robot] Error stopping servoJ: {exc}")
        
        # Stop data collection and teleoperation flags
        self.teleop_active = False
        self.data_collection_active = False
        
        # Ask user whether to save data if we were collecting data
        should_save = False
        if is_data_collection:
            should_save = self._ask_save_confirmation()
            
            if should_save:
                self.state = SystemState.SAVING
                rospy.loginfo("[robot] Saving collected data...")
                
                # Save zarr format if enabled
                saved = self._save_to_zarr()
                if self._zarr_enabled and self.save_data["rgb"] and not saved:
                    self._save_pending = True
                    self.state = SystemState.IDLE
                    rospy.logerr("[robot] Recording remains in memory. Fix the storage issue, then press 'q' to retry or discard. Do not exit the node.")
                    return
                
                # Note: PklSaver data is already saved frame by frame during collection
                save_cfg = self.robot_args.get("save_config", {})
                if self.datasaver and not save_cfg.get("save_robot_data_zarr", False):
                    if self._session_dir:
                        rospy.loginfo(f"[robot] Pkl data saved to: {self._session_dir}")
                
                # Report save status
                if self._zarr_enabled and self.zarr_output_path:
                    if os.path.exists(self.zarr_output_path):
                        rospy.loginfo(f"[robot] ✓ Zarr data saved: {self.zarr_output_path}")
                    else:
                        rospy.logwarn("[robot] Zarr save may have failed (file not found).")
                
                if self._session_dir:
                    rospy.loginfo(f"[robot] ✓ Data collection #{self._collection_count} saved successfully.")
                    rospy.loginfo(f"[robot] Session directory: {self._session_dir}")
                elif not self._zarr_enabled:
                    rospy.logwarn("[robot] No data was saved (data saving disabled).")
            else:
                rospy.loginfo("[robot] Data collection discarded (not saved).")
                # Clear save_data buffers
                for key in self.save_data:
                    self.save_data[key].clear()
        
        self._save_pending = False

        # Reset to initial position (now safe to use moveJ)
        self._reset_robots()
        
        # Clear command buffers
        self.arm_cmd_target = None
        self.rapid_cmd_target = None
        self.arm_cmd = None
        self.rapid_cmd = None
        self.last_sent_arm_cmd = None
        self.last_sent_rapid_cmd = None
        self.last_message_time = None
        
        # Clear zarr output path (will be set on next collection)
        self.zarr_output_path = None
        
        # Return to idle state
        self.state = SystemState.IDLE
        rospy.loginfo("[robot] ✓ Ready for next collection.")
        rospy.loginfo("[robot] Press 't' to start teleoperation or 's' to start data collection.")

    def _send_robot_commands(self) -> None:
        """Send robot commands with timeout detection and command deduplication."""
        # Use base class method with teleop state check
        super()._send_robot_commands(lambda: (
            self.teleop_active
            and not getattr(self, "_save_pending", False)
            and self.state in (SystemState.TELEOP_ACTIVE, SystemState.DATA_COLLECTION)
        ))

    def _poll_user_input(self) -> None:
        """Listen for user input to control the interaction flow."""
        if not self._stdin_available:
            return
        
        # Don't process input while saving (waiting for save confirmation is handled in _stop_and_reset)
        if self.state == SystemState.SAVING:
            return

        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if ready:
            user_input = sys.stdin.readline().strip().lower()
            
            if user_input == "t":
                self._start_teleoperation()
            elif user_input == "s":
                self._activate_data_collection()
            elif user_input == "q":
                self._stop_and_reset()  # Can stop data collection and/or teleoperation
            elif user_input:
                self._print_help_message()
    
    def _print_help_message(self) -> None:
        """Print help message based on current state."""
        rospy.loginfo("[robot] Commands:")
        if not self.teleop_active:
            rospy.loginfo("[robot]   't' - Start teleoperation")
        if not self.data_collection_active:
            rospy.loginfo("[robot]   's' - Start data collection")
        if self.teleop_active or self.data_collection_active:
            rospy.loginfo("[robot]   'q' - Stop current operation(s) and reset")
        
        if self.state == SystemState.SAVING:
            rospy.loginfo("[robot] Saving data, please wait...")
        elif self.teleop_active and self.data_collection_active:
            rospy.loginfo("[robot] Status: Teleoperation active, Data collection active")
        elif self.teleop_active:
            rospy.loginfo("[robot] Status: Teleoperation active")
        elif self.data_collection_active:
            rospy.loginfo("[robot] Status: Data collection active")

    def _handle_robot_data(self) -> None:
        """Handle robot data collection with tv_rgb support."""
        if not self.data_collection_active:
            return

        real_arm = self.arm.get_arm_data()
        real_hand = self.rapid_hand.get_hand_data()

        # Use base class method with tv_rgb enabled
        super()._collect_robot_data(real_arm, real_hand, include_tv_rgb=True)

    def _save_to_zarr(self) -> bool:
        """Save collected data to zarr format with tv_rgb support."""
        # Use base class method with tv_rgb enabled
        return super()._save_to_zarr(include_tv_rgb=True)

    def _shutdown(self) -> None:
        """Shutdown handler. Save any active collection and stop robots."""
        # ROS and the control loop can both request shutdown.
        if getattr(self, "_shutdown_started", False):
            return
        if getattr(self, "_stop_in_progress", False):
            self._shutdown_requested = True
            rospy.logwarn("[robot] Shutdown requested. Waiting for the current stop/save/reset to finish.")
            return
        self._shutdown_started = True

        # If we have active operations, stop them first
        # This will ask user to save if data collection is active
        if self.data_collection_active or self.teleop_active or getattr(self, "_save_pending", False):
            rospy.loginfo("[robot] Shutdown requested. Stopping current operation...")
            self._stop_and_reset()

        if getattr(self, "_save_pending", False):
            rospy.logerr("[robot] Shutdown save failed. The recording is still only in memory and will be lost when this process exits.")
        
        # Delete session directory if empty
        if self._session_dir and os.path.exists(self._session_dir):
            try:
                if not os.listdir(self._session_dir):
                    os.rmdir(self._session_dir)
                    rospy.loginfo(f"[robot] Removed empty session directory: {self._session_dir}")
            except Exception as exc:
                rospy.logdebug(f"[robot] Could not remove session directory: {exc}")
        
        # Stop camera display
        if self._camera_display_thread and self._camera_display_thread.is_alive():
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        
        # Call base class shutdown
        super()._shutdown()

    # ----------------------------------------------------------------- Main
    def run(self) -> None:
        """Main control loop with configurable rate (default 125Hz for servoJ)."""
        rospy.loginfo(f"[robot] Ready — starting control loop at {self.control_rate_hz} Hz")
        rate = rospy.Rate(self.control_rate_hz)
        rospy.on_shutdown(self._shutdown)
        
        # Print initial instructions
        self._print_initial_instructions()
        
        try:
            while not rospy.is_shutdown():
                self._poll_user_input()
                self._send_robot_commands()
                if self._should_collect_data:
                    try:
                        self._handle_robot_data()
                    except TimeoutError as exc:
                        rospy.logerr(f"[robot] Camera timeout: {exc}. Stopping current operation.")
                        self._stop_and_reset()
                    except MotorFeedbackError as exc:
                        rospy.logerr(f"[robot] {exc}. Stopping current operation.")
                        self._stop_and_reset()
                rate.sleep()
        finally:
            self._shutdown()
    
    def _print_initial_instructions(self) -> None:
        """Print initial instructions for user interaction."""
        rospy.loginfo("=" * 60)
        rospy.loginfo("[robot] System initialized and ready!")
        rospy.loginfo("=" * 60)
        save_cfg = self.robot_args.get("save_config", {})
        if save_cfg.get("save_robot_data", False) or save_cfg.get("save_robot_data_zarr", False):
            root_dir = save_cfg.get("root_dir", "./data")
            task_name = save_cfg.get("task_dir", "test_data")
            rospy.loginfo(f"[robot] Data will be saved to: {root_dir}/{task_name}/MMDD_HHMMSS")
        rospy.loginfo("[robot] Interaction flow:")
        rospy.loginfo("[robot]   't' - Start teleoperation and wait for 'Teleoperation active'")
        rospy.loginfo("[robot]   's' - Start data collection after teleoperation is active")
        rospy.loginfo("[robot]   'q' - Stop current operation(s) and reset")
        rospy.loginfo("[robot] Note: Start teleoperation before starting data collection")
        rospy.loginfo("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    # Resolve default configuration paths relative to this script.
    default_config = Path(__file__).parent / "args" / "teleop_args.yaml"
    
    parser = argparse.ArgumentParser(
        description="FingerViP teleoperation ROS node (UR5e + RAPID Hand)"
    )
    parser.add_argument(
        "robot_args",
        type=str,
        nargs='?',  # Make the positional argument optional.
        default=str(default_config),
        help=f"Path to robot_args YAML configuration file (default: {default_config})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot_system = RobotSystem(args.robot_args)
    try:
        robot_system.run()
    except rospy.ROSInterruptException:
        pass  # Graceful exit on Ctrl‑C within  ROS


if __name__ == "__main__":  # pragma: no cover
    main()
