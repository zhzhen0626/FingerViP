"""base_robot_system.py
---------------------------------
Base class for FingerViP robot control using a UR5e arm and RAPID Hand.

This module provides common functionality shared between teleoperation
and policy evaluation robot nodes.
"""

from __future__ import annotations

import time
import os
import datetime
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import rospy
import zarr

from control import RapidHandController, UR5eController
from utils.load_args import load_robot_args


class BaseRobotSystem:
    """Base class for FingerViP robot control using a UR5e arm and RAPID Hand.
    
    Provides common functionality for:
    - Robot controller initialization
    - Command preparation and execution
    - Data collection and saving
    - Robot reset and shutdown
    """

    def __init__(self, robot_args_path: str | Path) -> None:
        """Initialize base robot system.
        
        Args:
            robot_args_path: Path to robot configuration YAML file
        """
        self.robot_args, self.arm_cfg, self.hand_cfg = load_robot_args(robot_args_path)

        # Extract control parameters with defaults
        robot_ip = self.robot_args.get("robot_ip", "")
        
        # Advanced control parameters
        self.control_rate_hz = self.robot_args.get("control_rate_hz", 125.0)
        self.message_timeout = self.robot_args.get("message_timeout", 1.0)
        self.command_threshold = self.robot_args.get("command_threshold", 1e-3)

        # Sync configuration
        sync_cfg = self.robot_args.get("initial_sync", {})
        self.initial_sync_tolerance = np.deg2rad(sync_cfg.get("tolerance_deg", 1.0))
        self.initial_sync_max_step = np.deg2rad(sync_cfg.get("max_step_deg", 2.0))
        self.normal_max_step = np.deg2rad(sync_cfg.get("normal_max_step_deg", 2.5))
        
        # Hand step limiting (optional, default: disabled)
        self.enable_hand_step_limiting = sync_cfg.get("enable_hand_step_limiting", False)
        self.hand_initial_tolerance = sync_cfg.get("hand_tolerance", 0.05)
        self.hand_initial_max_step = sync_cfg.get("hand_max_step", 0.05)
        self.hand_normal_max_step = sync_cfg.get("hand_normal_max_step", 0.08)
        
        # UR5e controller with servoJ support (RTDE)
        use_servo = self.robot_args.get("use_servo", True)
        servo_speed = self.robot_args.get("servo_speed", 0.5)
        servo_acceleration = self.robot_args.get("servo_acceleration", 0.3)
        servo_lookahead_time = self.robot_args.get("servo_lookahead_time", 0.15)
        servo_gain = self.robot_args.get("servo_gain", 150)
        
        # Alignment configuration
        self.alignment_cfg = self.robot_args.get("alignment_config", {})
        self.alignment_enabled = self.alignment_cfg.get("enable", True)
        self.alignment_arm_velocity = self.alignment_cfg.get("arm_velocity", 0.1)
        self.alignment_arm_acc = self.alignment_cfg.get("arm_acceleration", 0.1)
        # self.alignment_hand_velocity = self.alignment_cfg.get("hand_velocity", 0.3)  # Disabled: only aligning arm
        self.alignment_wait_sec = self.alignment_cfg.get("wait_after_move_sec", 0.5)

        # Initialize ROS node FIRST before any rospy.loginfo calls
        self._init_ros()

        rospy.loginfo(f"[robot] Initializing UR5e at {robot_ip} (servoJ: {use_servo})")
        self.arm = UR5eController(
            robot_ip=robot_ip,
            init_qpos=None,  # Capture the arm pose at connection as the reset target.
            auto_reset=False,  # Keep movement in the existing reset stage.
            use_servo=use_servo,
            control_dt=1.0 / self.control_rate_hz,
            servo_speed=servo_speed,
            servo_acceleration=servo_acceleration,
            servo_lookahead_time=servo_lookahead_time,
            servo_gain=servo_gain,
        )
        
        # RapidHand controller
        self.rapid_hand = RapidHandController(**self.hand_cfg)

        # Joint name ordering & command buffers
        self.arm_joint_order: List[str] = self.arm_cfg.get(
            "arm_command_order", self.arm_cfg.get("ur5e_command_order", [])
        )
        self.rapid_joint_order: List[str] = self.hand_cfg["joint_command_order_list"]
        
        # Initialize command arrays
        self.arm_cmd: Optional[np.ndarray] = None
        self.rapid_cmd: Optional[np.ndarray] = None
        self.arm_cmd_target: Optional[np.ndarray] = None
        self.rapid_cmd_target: Optional[np.ndarray] = None
        self.last_sent_arm_cmd: Optional[np.ndarray] = None
        self.last_sent_rapid_cmd: Optional[np.ndarray] = None
        
        # Timestamp tracking for timeout detection
        self.last_message_time: Optional[float] = None

        # Mapping from tele‑op indices → robot indices (lazily initialised)
        self._teleop2arm: Optional[np.ndarray] = None
        self._teleop2rapid: Optional[np.ndarray] = None

        # Optional data saver / publisher helpers (unified interface)
        self.datasaver: Optional[object] = None  # PklSaver | None
        self.publisher: Optional[object] = None  # RobotDataPublisher | None
        
        # State management
        self.data_collection_active = False
        self._stdin_available = True  # Will be set by subclasses
        self.initial_sync_complete = False
        self.initial_hand_sync_complete = False
        self._initial_sync_warned = False
        self._initial_hand_sync_warned = False
        self._alignment_done = False
        self._aligned_arm_pose: Optional[np.ndarray] = None
        # self._aligned_hand_pose: Optional[np.ndarray] = None  # Disabled: only aligning arm
        self._zarr_enabled = False
        self.zarr_output_path: Optional[str] = None
        
        # Session directory (created once per node startup)
        self._session_dir: Optional[str] = None
        self._collection_count = 0

        # Initialize I/O helpers
        self._init_io_helpers()
        self._perform_initial_alignment()
        
        # Reset to initial position on startup
        rospy.loginfo("[robot] Resetting arm and hand to initial positions...")
        try:
            self.arm.reset()
            rospy.loginfo("[robot] ✓ Arm reset complete.")
        except Exception as exc:
            rospy.logerr(f"[robot] Failed to reset arm: {exc}")
        
        try:
            self.rapid_hand.reset()
            rospy.loginfo("[robot] ✓ Hand reset complete.")
        except Exception as exc:
            rospy.logerr(f"[robot] Failed to reset hand: {exc}")
    
        # Failed saves remain pending until retried or explicitly discarded.
        self._save_pending = False
        self._stop_in_progress = False
        self._shutdown_requested = False
        self._shutdown_started = False

        # Initialize save_data structure (subclasses can override)
        self.save_data = {
            "rgb": [],
            "timestamp": [],
            "joint_pos": [],
            "joint_vel": [],
            "joint_current": [],
            "control": [],
        }

    # ------------------------------------------------------------------ ROS
    def _init_ros(self) -> None:
        """Initialize ROS node. Subclasses should override to add subscribers."""
        rospy.init_node("robot_system", anonymous=True)
        rospy.loginfo("[robot] ROS node initialised")

    # ---------------------------------------------------------------- IO‑helpers
    def _init_io_helpers(self) -> None:
        """Initialize I/O helpers (PklSaver, RobotDataPublisher, session directory)."""
        save_cfg = self.robot_args.get("save_config", {})
        publish_cfg = self.robot_args.get("publish_config", {})

        self._zarr_enabled = save_cfg.get("save_robot_data_zarr", False)
        self.zarr_output_path = None

        # Create session directory once per node startup (if saving is enabled)
        save_robot_data = save_cfg.get("save_robot_data", False)
        save_robot_data_zarr = save_cfg.get("save_robot_data_zarr", False)
        
        if save_robot_data or save_robot_data_zarr:
            root_dir = save_cfg.get("root_dir", "./data")
            task_name = save_cfg.get("task_dir", "test_data")
            time_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
            self._session_dir = os.path.abspath(os.path.join(root_dir, task_name, time_str))
            os.makedirs(self._session_dir, exist_ok=True)
            rospy.loginfo(f"[robot] Session directory created: {self._session_dir}")

        # Initialize PklSaver if save_robot_data is enabled
        if save_robot_data:
            from data.scripts import PklSaver
            
            if self._session_dir:
                self.datasaver = PklSaver(
                    task_name=task_name,
                    root_dir=root_dir
                )
                self.datasaver.frame_dir = self._session_dir

        if publish_cfg.get("publish_robot_data", False):
            from data.scripts import RobotDataPublisher

            robot_data_topic = publish_cfg.get("robot_data_topic", {})
            self.publisher = RobotDataPublisher(
                rate_hz=50,
                init_node=False,
                **robot_data_topic,
            )
            self.publisher.start()

    def _perform_initial_alignment(self) -> None:
        """Move the physical devices to a known alignment pose before teleop begins."""
        if not self.alignment_enabled:
            rospy.loginfo("[robot] Alignment step disabled via configuration.")
            return

        arm_pose = self.alignment_cfg.get("arm_joint_positions")
        if arm_pose is not None:
            try:
                arm_pose_arr = np.asarray(arm_pose, dtype=float)
                if arm_pose_arr.shape != (len(self.arm_joint_order),):
                    rospy.logwarn(
                        "[robot] alignment_config.arm_joint_positions has incorrect length; "
                        f"expected {len(self.arm_joint_order)}, got {arm_pose_arr.shape}"
                    )
                else:
                    rospy.loginfo("[robot] Moving UR5e to alignment pose before teleoperation.")
                    self.arm.move_to_joint_positions(
                        arm_pose_arr,
                        acceleration=self.alignment_arm_acc,
                        velocity=self.alignment_arm_velocity,
                        async_move=False,
                    )
                    self._aligned_arm_pose = arm_pose_arr
            except Exception as exc:
                rospy.logerr(f"[robot] Failed to move UR5e to alignment pose: {exc}")

        # Hand alignment disabled - only aligning arm
        # hand_pose = self.alignment_cfg.get("hand_joint_positions")
        # if hand_pose is not None:
        #     try:
        #         hand_pose_arr = np.asarray(hand_pose, dtype=float)
        #         if hand_pose_arr.shape != (len(self.rapid_joint_order),):
        #             rospy.logwarn(
        #                 "[robot] alignment_config.hand_joint_positions has incorrect length; "
        #                 f"expected {len(self.rapid_joint_order)}, got {hand_pose_arr.shape}"
        #             )
        #         else:
        #             rospy.loginfo("[robot] Moving RapidHand to alignment pose before teleoperation.")
        #             self.rapid_hand.control_hand_qpos(hand_pose_arr)
        #             self._aligned_hand_pose = hand_pose_arr
        #     except Exception as exc:
        #         rospy.logerr(f"[robot] Failed to move RapidHand to alignment pose: {exc}")

        if self.alignment_wait_sec > 0.0:
            rospy.logdebug(f"[robot] Waiting {self.alignment_wait_sec:.2f}s after alignment motion.")
            time.sleep(self.alignment_wait_sec)

        self._alignment_done = True

    # -------------------------------------------------------------- Helpers
    @property
    def _should_collect_data(self) -> bool:
        """Check if data collection should be performed."""
        return self.data_collection_active

    @staticmethod
    def _limit_step(reference: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
        """Limit the step size between reference and target.
        
        Args:
            reference: Current reference value
            target: Target value
            max_step: Maximum allowed step size
            
        Returns:
            Limited target value
        """
        if max_step is None or max_step <= 0:
            return target

        delta = target - reference
        max_delta = np.max(np.abs(delta))
        if max_delta <= max_step:
            return target

        return reference + np.clip(delta, -max_step, max_step)

    def _prepare_arm_command(self) -> Optional[np.ndarray]:
        """Prepare arm command with initial sync and step limiting.
        
        Returns:
            Prepared arm command array, or None if no target available
        """
        if self.arm_cmd_target is None:
            return None

        target = np.asarray(self.arm_cmd_target, dtype=float)

        # Initial sync phase
        if not self.initial_sync_complete:
            try:
                arm_feedback = self.arm.get_arm_data()
                current = np.asarray(arm_feedback["joint_positions"], dtype=float)
            except Exception as exc:
                rospy.logerr_throttle(5.0, f"[robot] Could not read UR feedback for initial sync: {exc}")
                return None

            delta = target - current
            max_delta = np.max(np.abs(delta))
            if max_delta <= self.initial_sync_tolerance:
                self.initial_sync_complete = True
                rospy.loginfo("[robot] ✓ Arm initial sync complete.")
                return target

            if not self._initial_sync_warned:
                rospy.logwarn(
                    "[robot] Teleop arm target far from UR pose. Ramping by %.2f deg per cycle.",
                    np.rad2deg(self.initial_sync_max_step),
                )
                self._initial_sync_warned = True

            return current + np.clip(delta, -self.initial_sync_max_step, self.initial_sync_max_step)

        # Normal operation: step limiting
        reference = self.arm_cmd if self.arm_cmd is not None else self.last_sent_arm_cmd
        if reference is None:
            return target

        return self._limit_step(reference, target, self.normal_max_step)

    def _prepare_hand_command(self, enable_initial_sync: bool = False) -> Optional[np.ndarray]:
        """Prepare hand command with optional initial sync and step limiting.
        
        Args:
            enable_initial_sync: Whether to perform initial sync (default: False)
            
        Returns:
            Prepared hand command array, or None if no target available
        """
        if self.rapid_cmd_target is None:
            return None

        target = np.asarray(self.rapid_cmd_target, dtype=float)

        # If hand step limiting is disabled, return target directly
        if not self.enable_hand_step_limiting:
            return target

        # Initial sync phase (if enabled)
        if enable_initial_sync and not self.initial_hand_sync_complete:
            try:
                hand_feedback = self.rapid_hand.get_hand_data()
                current = np.asarray(hand_feedback["joint_positions"], dtype=float)
            except Exception as exc:
                rospy.logerr_throttle(5.0, f"[robot] Could not read RapidHand feedback for initial sync: {exc}")
                return target

            delta = target - current
            max_delta = np.max(np.abs(delta))
            if max_delta <= self.hand_initial_tolerance:
                self.initial_hand_sync_complete = True
                return target

            if not self._initial_hand_sync_warned:
                rospy.logwarn(
                    "[robot] Teleop hand target far from RapidHand pose at startup. Ramping by %.3f units per cycle.",
                    self.hand_initial_max_step,
                )
                self._initial_hand_sync_warned = True

            return current + np.clip(delta, -self.hand_initial_max_step, self.hand_initial_max_step)

        # Normal operation: step limiting
        reference = self.rapid_cmd if self.rapid_cmd is not None else self.last_sent_rapid_cmd
        if reference is None:
            return target

        return self._limit_step(reference, target, self.hand_normal_max_step)

    def _send_robot_commands(self, state_check_func) -> None:
        """Send robot commands with timeout detection and command deduplication.
        
        Args:
            state_check_func: Function that returns True if commands should be sent
        """
        # Check if we should send commands
        if not state_check_func():
            return
        
        # Check if we have received any commands yet
        if self.arm_cmd_target is None or self.rapid_cmd_target is None:
            rospy.logdebug_throttle(5.0, "[robot] Waiting for first command...")
            return
        
        self.arm_cmd = self._prepare_arm_command()
        self.rapid_cmd = self._prepare_hand_command()

        if self.arm_cmd is None or self.rapid_cmd is None:
            return
        
        # Check for message timeout
        current_time = rospy.Time.now().to_sec()
        if self.last_message_time is not None:
            time_since_last_msg = current_time - self.last_message_time
            if time_since_last_msg > self.message_timeout:
                rospy.logwarn_throttle(
                    5.0,
                    f"[robot] No messages received for {time_since_last_msg:.2f}s "
                    f"(timeout={self.message_timeout}s). Skipping command."
                )
                return
        
        # Command deduplication - check if command changed significantly
        send_arm = True
        send_hand = True
        
        if self.last_sent_arm_cmd is not None:
            arm_change = np.max(np.abs(self.arm_cmd - self.last_sent_arm_cmd))
            if arm_change < self.command_threshold:
                send_arm = False
        
        if self.last_sent_rapid_cmd is not None:
            hand_change = np.max(np.abs(self.rapid_cmd - self.last_sent_rapid_cmd))
            if hand_change < self.command_threshold:
                send_hand = False
        
        # Send commands if changed significantly
        try:
            if send_arm:
                self.arm.control_arm_qpos(self.arm_cmd)
                self.last_sent_arm_cmd = self.arm_cmd.copy()
                rospy.logdebug(f"[robot] Sent arm command: {self.arm_cmd}")
            
            if send_hand:
                self.rapid_hand.control_hand_qpos(self.rapid_cmd)
                self.last_sent_rapid_cmd = self.rapid_cmd.copy()
                rospy.logdebug(f"[robot] Sent hand command: {self.rapid_cmd}")
                
        except Exception as e:
            rospy.logerr_throttle(5.0, f"[robot] Error sending commands: {e}")

    def _stop_and_reset(self) -> None:
        """Finish stop/save/reset before handling a reentrant shutdown request."""
        if getattr(self, "_stop_in_progress", False):
            rospy.logwarn("[robot] Stop/save/reset is already in progress. Please wait.")
            return

        self._stop_in_progress = True
        try:
            self._stop_and_reset_impl()
        finally:
            self._stop_in_progress = False
            if getattr(self, "_shutdown_requested", False):
                self._shutdown_requested = False
                self._shutdown()

    def _stop_and_reset_impl(self) -> None:
        """Implement the node-specific stop, save and reset sequence."""
        raise NotImplementedError

    def _reset_robots(self) -> None:
        """Reset robots to initial positions."""
        rospy.loginfo("[robot] Resetting arm and hand to initial positions...")
        try:
            self.arm.reset()
            rospy.loginfo("[robot] ✓ Arm reset complete.")
        except Exception as exc:
            rospy.logerr(f"[robot] Failed to reset arm: {exc}")
        
        try:
            self.rapid_hand.reset()
            rospy.loginfo("[robot] ✓ Hand reset complete.")
        except Exception as exc:
            rospy.logerr(f"[robot] Failed to reset hand: {exc}")

    def _collect_robot_data(self, real_arm: dict, real_hand: dict, include_tv_rgb: bool = False) -> None:
        """Collect robot data and save to buffers.
        
        Args:
            real_arm: Arm data dictionary
            real_hand: Hand data dictionary
            include_tv_rgb: Whether to include third view RGB data
        """
        # Store five fingertip views in thumb, index, middle, ring, little order.
        self.save_data["rgb"].append(real_hand["fingertip_rgb"].copy())

        # Save third view RGB if enabled
        if include_tv_rgb:
            if "tv_rgb" in self.save_data:
                self.save_data["tv_rgb"].append(real_hand["third_view_rgb"])
            else:
                # Initialize tv_rgb list if it doesn't exist
                self.save_data["tv_rgb"] = [real_hand["third_view_rgb"]]
        
        # Save other data
        self.save_data["timestamp"].append(time.time())
        self.save_data["joint_pos"].append(
            np.concatenate([real_arm["joint_positions"], real_hand["joint_positions"]])
        )
        self.save_data["joint_vel"].append(
            np.concatenate([real_arm["joint_velocities"], real_hand["joint_velocities"]])
        )
        self.save_data["joint_current"].append(
            np.concatenate([real_arm["joint_current"], real_hand["joint_current"]])
        )
    
        # Save control commands
        # Safeguard: ensure command arrays exist and are 1-D before concatenation
        if self.arm_cmd is not None and self.rapid_cmd is not None:
            arm_cmd = np.atleast_1d(self.arm_cmd)
            rapid_cmd = np.atleast_1d(self.rapid_cmd)
            self.save_data["control"].append(np.concatenate([arm_cmd, rapid_cmd]))
        else:
            total_cmd_size = len(self.arm_joint_order) + len(self.rapid_joint_order)
            self.save_data["control"].append(np.zeros(total_cmd_size))
        
        # Save to PklSaver if enabled
        if self.datasaver:
            # Build safe command for saving
            if self.arm_cmd is not None and self.rapid_cmd is not None:
                arm_cmd = np.atleast_1d(self.arm_cmd)
                rapid_cmd = np.atleast_1d(self.rapid_cmd)
                command = np.concatenate([arm_cmd, rapid_cmd])
            else:
                command = np.concatenate([
                    np.zeros(len(self.arm_joint_order)),
                    np.zeros(len(self.rapid_joint_order)),
                ])

            save_cfg = self.robot_args.get("save_config", {})
            if not save_cfg.get("save_robot_data_zarr", False):
                self.datasaver.save_frame(
                    real_arm_data=real_arm,
                    real_hand_data=real_hand,
                    command=command,
                )
            rospy.logdebug("[robot] Frame saved")

        # Publish data if enabled
        if self.publisher:
            self.publisher.update_data(
                joint_angles=np.concatenate(
                    [real_arm["joint_positions"], real_hand["joint_positions"]]
                ),
                joint_names=self.arm_joint_order + self.rapid_joint_order,
                image=real_hand["third_view_rgb"][0],
            )

    def _save_to_zarr(self, zarr_path: Optional[str] = None, include_tv_rgb: bool = False) -> bool:
        """Save a complete episode atomically; return False if it could not be saved.

        Writing is synchronous, with no timeout: all frames and the final rename
        complete before success is returned.
        
        Args:
            zarr_path: Path to save zarr file (uses self.zarr_output_path if None)
            include_tv_rgb: Whether to include third view RGB dataset
        """
        if not self._zarr_enabled:
            rospy.logdebug("[robot] Zarr saving is disabled. Skipping zarr save.")
            return False

        path = zarr_path if zarr_path is not None else self.zarr_output_path
        if not path:
            rospy.logwarn("[robot] Zarr output path not initialised. Skipping zarr save.")
            return False

        if not self.save_data["rgb"]:
            rospy.logwarn("[robot] No data collected, skipping zarr save.")
            return False

        # Check if path already exists (avoid overwrite)
        if os.path.exists(path):
            rospy.logwarn(f"[robot] Zarr path already exists: {path}")
            rospy.logwarn("[robot] Skipping zarr save to avoid overwrite.")
            return False
        
        rospy.loginfo(f"[robot] Saving to Zarr path: {path}")

        try:
            num_frames = len(self.save_data["rgb"])
            required = ["timestamp", "joint_pos", "joint_vel", "joint_current", "control"]
            if include_tv_rgb:
                required.append("tv_rgb")
            for key in required:
                if len(self.save_data.get(key, [])) != num_frames:
                    raise ValueError(f"{key} must have one entry per RGB frame ({num_frames}).")
            first_rgb_frame = np.asarray(self.save_data["rgb"][0], dtype=np.uint8)
            num_cams, img_dim1, img_dim2, channels = first_rgb_frame.shape
            if num_cams != 5 or channels != 3:
                raise ValueError("Expected five fingertip RGB images per frame.")

            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            # A sibling staging directory keeps failed writes out of episode discovery.
            with tempfile.TemporaryDirectory(prefix=".fingervip-recording-", dir=parent) as staging:
                staged_path = os.path.join(staging, "episode.zarr")
                store = zarr.DirectoryStore(staged_path)
                root = zarr.group(store=store, overwrite=True)

                # Compressor
                compressor = zarr.Blosc(cname="zstd", clevel=5)

                # Create empty datasets and write frames or chunks without building large arrays.
                rgb_ds = root.create_dataset(
                    "rgb",
                    shape=(num_frames, num_cams, img_dim1, img_dim2, 3),
                    chunks=(1, num_cams, img_dim1, img_dim2, 3),
                    compressor=compressor,
                    dtype=np.uint8,
                )

                timestamp_ds = root.create_dataset(
                    "timestamp",
                    shape=(num_frames,),
                    chunks=(min(1024, num_frames),),
                    compressor=compressor,
                    dtype=np.float64,
                )

                joint_pos_ds = root.create_dataset(
                    "joint_pos",
                    shape=(num_frames, len(self.save_data["joint_pos"][0])),
                    chunks=(min(1024, num_frames), len(self.save_data["joint_pos"][0])),
                    compressor=compressor,
                    dtype=np.float32,
                )

                joint_vel_ds = root.create_dataset(
                    "joint_vel",
                    shape=(num_frames, len(self.save_data["joint_vel"][0])),
                    chunks=(min(1024, num_frames), len(self.save_data["joint_vel"][0])),
                    compressor=compressor,
                    dtype=np.float32,
                )

                joint_current_ds = root.create_dataset(
                    "joint_current",
                    shape=(num_frames, len(self.save_data["joint_current"][0])),
                    chunks=(min(1024, num_frames), len(self.save_data["joint_current"][0])),
                    compressor=compressor,
                    dtype=np.float32,
                )

                control_ds = root.create_dataset(
                    "control",
                    shape=(num_frames, len(self.save_data["control"][0])),
                    chunks=(min(1024, num_frames), len(self.save_data["control"][0])),
                    compressor=compressor,
                    dtype=np.float32,
                )

                # Optional third-view RGB data.
                if include_tv_rgb and "tv_rgb" in self.save_data and self.save_data["tv_rgb"]:
                    num_tv_frames = len(self.save_data["tv_rgb"])
                    # Infer tv_rgb image dimensions from first frame
                    # tv_rgb data has shape (1, dim1, dim2, 3) - squeeze the batch dimension
                    first_tv_frame = np.asarray(self.save_data["tv_rgb"][0], dtype=np.uint8)
                    if first_tv_frame.ndim == 4 and first_tv_frame.shape[0] == 1:
                        first_tv_frame = first_tv_frame.squeeze(axis=0)
                    tv_img_dim1, tv_img_dim2 = first_tv_frame.shape[0], first_tv_frame.shape[1]
                    tv_rgb_ds = root.create_dataset(
                        "tv_rgb",
                        shape=(num_tv_frames, tv_img_dim1, tv_img_dim2, 3),
                        chunks=(1, tv_img_dim1, tv_img_dim2, 3),
                        compressor=compressor,
                        dtype=np.uint8,
                    )
                else:
                    tv_rgb_ds = None

                # Write one frame at a time to avoid a single large array allocation.
                for i in range(num_frames):
                    # Preserve the source image axes: (num_cams, dim1, dim2, 3).
                    rgb_frame = np.asarray(self.save_data["rgb"][i], dtype=np.uint8)
                    rgb_ds[i] = rgb_frame

                    timestamp_ds[i] = float(self.save_data["timestamp"][i])
                    joint_pos_ds[i] = np.asarray(self.save_data["joint_pos"][i], dtype=np.float32)
                    joint_vel_ds[i] = np.asarray(self.save_data["joint_vel"][i], dtype=np.float32)
                    joint_current_ds[i] = np.asarray(self.save_data["joint_current"][i], dtype=np.float32)
                    control_ds[i] = np.asarray(self.save_data["control"][i], dtype=np.float32)

                if tv_rgb_ds is not None:
                    for i in range(len(self.save_data["tv_rgb"])):
                        tv_rgb_frame = np.asarray(self.save_data["tv_rgb"][i], dtype=np.uint8)
                        # Remove the camera batch axis without transposing the image.
                        if tv_rgb_frame.ndim == 4 and tv_rgb_frame.shape[0] == 1:
                            tv_rgb_frame = tv_rgb_frame.squeeze(axis=0)
                        tv_rgb_ds[i] = tv_rgb_frame

                store.close()
                if os.path.exists(path):
                    raise FileExistsError(f"Zarr path already exists: {path}")
                os.rename(staged_path, path)
            rospy.loginfo(f"[robot] ✓ Zarr dataset saved successfully: {path}")
            return True
        except Exception as exc:
            rospy.logerr(f"[robot] Failed to save zarr dataset: {exc}")
            return False

    def _shutdown(self) -> None:
        """Shutdown handler. Stop robots gracefully."""
        try:
            self.arm.stop()
        except Exception:
            pass
        
        try:
            self.rapid_hand.stop()
        except Exception as exc:
            rospy.logdebug(f"[robot] Error stopping hand during shutdown: {exc}")
        
        rospy.loginfo("[robot] Shutdown complete")
