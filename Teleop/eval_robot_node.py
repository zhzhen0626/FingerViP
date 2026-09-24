"""eval_robot_node.py
---------------------------------
High-level wrapper for FingerViP policy evaluation.

This module provides an integrated control system for UR5e robotic arm
and RapidHand using diffusion policy models for autonomous control.

USAGE
-----
$ python eval_robot_node.py path/to/robot_args.yaml path/to/checkpoint

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
import pathlib
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import JointState
import threading
import queue
import yaml

import os
ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR / "Teleop"))
sys.path.append(str(ROOT_DIR / "DiffusionPolicy"))

from base_robot_system import BaseRobotSystem
from control.rapid_hand_control.rapid_node import MotorFeedbackError

import datetime
from typing import Optional, Tuple
from enum import Enum

# Import HWDiffusionPolicyEval from eval_fingervip.py
from diffusion_policy.eval_fingervip import HWDiffusionPolicyEval

# State machine for interaction flow
class SystemState(Enum):
    """System state for policy control and data collection."""
    IDLE = "idle"  # Initial state
    POLICY_CONTROL_ACTIVE = "policy_control_active"  # Policy control active
    DATA_COLLECTION = "data_collection"  # Collecting data
    WAITING_SAVE_CONFIRM = "waiting_save_confirm"  # Waiting for user to confirm save (y/n)
    SAVING = "saving"  # Saving data and resetting

class RobotSystem(BaseRobotSystem):
    """Handle FingerViP policy evaluation with a UR5e arm and RAPID Hand."""

    def __init__(self, robot_args_path: str | Path,
                 ckpt_path: str | Path, # diffusion policy checkpoint path
                 eval_config: Optional[dict] = None,  # Evaluation configuration from eval_args.yaml
                 handbase_transform_mask: np.ndarray = np.ones(5, dtype=int),
                 in_res: tuple = (640, 480), 
                 out_res: tuple = (224, 224),
                 urdf_file_name: str = "ur5e_with_rapid_hand_right_fingercamera",
                 random_disabling: bool = False,    # randomly disable cameras
                 random_disabling_prob: float = 0.1,    # probability of frame being disabled some cameras
                 random_latency: bool = False,      # randomly add latency to cameras
                 random_latency_prob: float = 0.2,  # probability of adding latency to some cameras
                 random_latency_range: Tuple[float, float] = (0, 1.0),   # range of latency to add to cameras
                 store_attention: bool = False,  # store attention maps
                 render: bool = False,
                 num_steps: int = 1  # Number of action steps to execute (range: 1-16)
                ) -> None:
        # Initialize base class (handles robot controllers, I/O, alignment, reset)
        super().__init__(robot_args_path)
        # Store five fingertip images in rgb and the separate third view in tv_rgb.
        self.save_data = {
            "rgb": [],
            "tv_rgb": [],  # third view RealSense camera
            "timestamp": [],
            "joint_pos": [],
            "joint_vel": [],
            "joint_current": [],
            "control": [],
        }

        # Validate and store action step configuration
        if not (1 <= num_steps <= 16):
            raise ValueError(
                f"Invalid num_steps: {num_steps}. Requires: 1 <= num_steps <= 16"
            )
        self.num_steps = num_steps
        # Calculate step indices: always start from 0, end at num_steps-1
        self.action_step_a = 0
        self.action_step_b = num_steps - 1
        self.action_steps_to_execute = list(range(self.action_step_a, self.action_step_b + 1))
        
        # Initialize policy evaluator using HWDiffusionPolicyEval from eval_fingervip.py
        rospy.loginfo("[robot] Initializing diffusion policy evaluator...")
        self.policy_eval = HWDiffusionPolicyEval(
            ckpt_path=ckpt_path,
            handbase_transform_mask=handbase_transform_mask,
            in_res=in_res,
            out_res=out_res,
            urdf_file_name=urdf_file_name,
            random_disabling=random_disabling,
            random_disabling_prob=random_disabling_prob,
            random_latency=random_latency,
            random_latency_prob=random_latency_prob,
            random_latency_range=random_latency_range,
            store_attention=store_attention,
            render=render,
            use_current=True,
            use_third_view=False,
        )
        rospy.loginfo("[robot] ✓ Policy evaluator initialized.")
        
        # Frame counter for policy prediction
        self._frame_idx = 0
        
        # Queue for passing predicted actions from prediction thread to main thread
        self._action_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory buildup
        
        # Queue for storing action sequence to execute (when a < b)
        self._action_sequence_queue: Optional[list] = None
        self._action_sequence_idx = 0
        self._action_step_cycles = 0
        self._is_executing_sequence = False  # Flag to track if we're executing an action sequence
        
        # Thread for policy prediction
        self._prediction_thread = None
        self._prediction_thread_running = False
        self._prediction_lock = threading.Lock()
        
        # Action interpolation for smooth control
        self.policy_inference_interval = max(1, int(self.control_rate_hz / 10))  # Run policy at ~10Hz instead of 125Hz
        self.policy_inference_counter = 0
        self.last_action: Optional[np.ndarray] = None
        self.current_action_target: Optional[np.ndarray] = None
        
        # State management (policy evaluation specific)
        self.state = SystemState.IDLE
        self.policy_control_active = False
        self._start_prompt_announced = False
        self._device_ready_announced = False
        self._stdin_available = sys.stdin.isatty()
        
        # Camera display thread
        self._camera_display_thread = None
        self._camera_display_running = False
        
        # Initialize camera display if enabled
        # Priority: eval_config > robot_args (for backward compatibility)
        eval_config = eval_config or {}
        display_cfg = eval_config.get("display_config", {})
        enable_fingertip_camera = display_cfg.get("enable_fingertip_camera", 
                                                   self.robot_args.get("show_camera_display", False))
        
        if enable_fingertip_camera:
            self._start_camera_display()
        
        # Read auto_start_data_collection configuration
        self.auto_start_data_collection = eval_config.get("auto_start_data_collection", False)
        
        # Track whether any data was saved this session (for cleanup of empty session dir)
        self._saved_any_data_this_session = False

        # Log session directory info
        if self._session_dir:
            rospy.loginfo(f"[robot] Session directory created: {self._session_dir}")
            rospy.loginfo("[robot] All data will be saved to this directory during this session.")



    # ------------------------------------------------------------------ ROS
    def _init_ros(self) -> None:
        """Initialize ROS node (no subscriber for policy evaluation)."""
        super()._init_ros()

    # ---------------------------------------------------------- Callbacks
    def _joint_angle_cb(self, msg: JointState) -> None:
        """Convert incoming tele‑op message into robot command arrays.
        
        Features:
        - Joint mapping extraction
        - Timestamp tracking for timeout detection
        - Expected input units: radians
        """
        teleop_names = msg.name
        
        # Build mapping lazily on first message
        if self._teleop2arm is None:
            self._teleop2arm = np.array(
                [teleop_names.index(j) for j in self.arm_joint_order], dtype=int
            )
            rospy.loginfo(f"[robot] Arm mapping: {self._teleop2arm.tolist()}")
            self._announce_start_prompt_if_ready()

        if self._teleop2rapid is None:
            self._teleop2rapid = np.array(
                [teleop_names.index(j) for j in self.rapid_joint_order], dtype=int
            )
            rospy.loginfo(f"[robot] RapidHand mapping: {self._teleop2rapid.tolist()}")
            self._announce_start_prompt_if_ready()

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
    
    def _joint_angle_policy(self, control_command) -> None:
        arm_joint_count = len(self.arm_joint_order)
        hand_joint_count = len(self.rapid_joint_order)
        all_joint_count = arm_joint_count + hand_joint_count

        # Build mapping lazily on first message
        if self._teleop2arm is None:
            self._teleop2arm = np.arange(arm_joint_count, dtype=int)
            print(f"[robot] Arm mapping: {self._teleop2arm.tolist()}")
            self._announce_start_prompt_if_ready()

        if self._teleop2rapid is None:
            self._teleop2rapid = np.arange(arm_joint_count, all_joint_count, dtype=int)
            print(f"[robot] RapidHand mapping: {self._teleop2rapid.tolist()}")
            self._announce_start_prompt_if_ready()

        # Extract joint positions
        teleop_pos = np.array(control_command).copy()
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

    # -------------------------------------------------------------- Helpers
    def _prepare_hand_command(self) -> Optional[np.ndarray]:
        """Prepare hand command with initial sync enabled for policy evaluation."""
        # Use base class method with initial sync enabled
        return super()._prepare_hand_command(enable_initial_sync=True)

    def _policy_prediction_thread(self) -> None:
        """
        Separate thread for policy prediction using eval_fingervip.py methods.
        
        NOTE: This method is currently not used. Policy prediction happens
        synchronously in _predict_action_async() called from main thread.
        This method is kept for potential future async implementation.
        """
        rospy.loginfo("[robot] Policy prediction thread started.")
        
        while self._prediction_thread_running and not rospy.is_shutdown():
            try:
                # Wait for robot data to be available
                # This will be triggered by the main thread providing data
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                
                # Check if we have data to process
                # The main thread will set this flag when data is ready
                if not self.data_collection_active:
                    continue
                    
            except Exception as e:
                rospy.logerr(f"[robot] Error in policy prediction thread: {e}")
                time.sleep(0.1)
        
        rospy.loginfo("[robot] Policy prediction thread stopped.")
    
    def _predict_action_async(self, real_arm, real_hand, frame_idx) -> None:
        """
        Predict action using eval_fingervip.py methods and put in queue.
        
        This is the ONLY method that calls policy prediction.
        Uses HWDiffusionPolicyEval.predict_actions from eval_fingervip.py.
        No other code should read or load policy predicted actions.
        """
        try:
            # Use HWDiffusionPolicyEval.predict_actions from eval_fingervip.py
            # This is the ONLY way to get policy predictions
            pred_actions = self.policy_eval.predict_actions(real_arm, real_hand, frame_idx)
            
            if pred_actions is not None:
                # Put actions in queue (non-blocking, drop if queue is full)
                try:
                    self._action_queue.put_nowait(pred_actions)
                except queue.Full:
                    rospy.logwarn_throttle(1.0, "[robot] Action queue full, dropping oldest action.")
                    try:
                        # Remove oldest action and add new one
                        _ = self._action_queue.get_nowait()
                        self._action_queue.put_nowait(pred_actions)
                    except queue.Empty:
                        pass
        except KeyboardInterrupt:
            rospy.logwarn("[robot] Policy prediction interrupted")
        except Exception as e:
            rospy.logerr(f"[robot] Error in policy prediction: {e}")

    def _start_policy_control(self) -> None:
        """Start policy control mode (independent of data collection)."""
        if getattr(self, "_save_pending", False):
            rospy.logwarn("[robot] Unsaved recording retained. Press 'q' to retry saving or discard it first.")
            return
        if self.policy_control_active:
            rospy.logwarn("[robot] Policy control is already active.")
            return
        
        if self.state == SystemState.SAVING:
            rospy.logwarn("[robot] Cannot start policy control while saving.")
            return
        
        rospy.loginfo("[robot] Starting policy control...")
        
        # Reset sync flags for new session
        self.initial_sync_complete = False
        self.initial_hand_sync_complete = False
        self._initial_sync_warned = False
        self._initial_hand_sync_warned = False
        
        # Clear command buffers
        self.arm_cmd_target = None
        self.rapid_cmd_target = None
        self.arm_cmd = None
        self.rapid_cmd = None
        self.last_sent_arm_cmd = None
        self.last_sent_rapid_cmd = None
        self.last_message_time = None

        # Reset frame counter for policy prediction
        self._frame_idx = 0
        
        # Reset policy state
        self.policy_eval.policy.reset()
        self.policy_eval.last_obs_dict_np = None
        
        # Reset action interpolation state
        self.policy_inference_counter = 0
        self.last_action = None
        self.current_action_target = None
        self._action_sequence_queue = None
        self._action_sequence_idx = 0
        self._action_step_cycles = 0
        self._is_executing_sequence = False

        # Clear action queue
        while not self._action_queue.empty():
            try:
                self._action_queue.get_nowait()
            except queue.Empty:
                break

        self.policy_control_active = True
        if not self.data_collection_active:
            self.state = SystemState.POLICY_CONTROL_ACTIVE
        rospy.loginfo("[robot] ✓ Policy control started!")
        
        # Auto-start data collection if configured
        if self.auto_start_data_collection and not self.data_collection_active:
            rospy.loginfo("[robot] Auto-starting data collection (auto_start_data_collection=True)...")
            self._activate_data_collection()

    def _activate_data_collection(self) -> None:
        """Start data collection (independent of policy control)."""
        if getattr(self, "_save_pending", False):
            rospy.logwarn("[robot] Unsaved recording retained. Press 'q' to retry saving or discard it first.")
            return
        if self.data_collection_active:
            rospy.logwarn("[robot] Data collection is already active.")
            return
        
        if self.state == SystemState.SAVING:
            rospy.logwarn("[robot] Cannot start data collection while saving.")
            return

        # Increment collection count
        self._collection_count += 1
        
        # Update PklSaver's frame_dir if needed (use fixed session directory)
        if self._session_dir and self.datasaver:
            self.datasaver.frame_dir = self._session_dir

        # Clear data buffers
        for key in self.save_data:
            self.save_data[key].clear()

        self.data_collection_active = True
        self.state = SystemState.DATA_COLLECTION
        rospy.loginfo(f"[robot] ✓ Data collection started (Collection #{self._collection_count}).")
        if self._session_dir:
            rospy.loginfo(f"[robot] Data will be saved to: {self._session_dir}")
        rospy.loginfo("[robot] Press 'q' to stop data collection.")

    def _send_robot_commands(self) -> None:
        """Send robot commands with timeout detection and command deduplication."""
        # Use base class method - send commands when policy control or data collection is active
        super()._send_robot_commands(lambda: self.policy_control_active or self.data_collection_active)

    def _poll_start_signal(self) -> None:
        """Listen for user input to control the interaction flow."""
        # Don't poll input when waiting for save confirmation (handled in _stop_and_reset)
        if self.state == SystemState.WAITING_SAVE_CONFIRM:
            return

        if not self._stdin_available:
            return

        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if ready:
            user_input = sys.stdin.readline().strip().lower()
            if user_input == "c":  # Start policy control
                self._start_policy_control()
            elif user_input == "s":  # Start data collection
                self._activate_data_collection()
            elif user_input == "q":  # Stop current operation(s)
                self._stop_and_reset()
            elif user_input:
                self._print_help_message()
    
    def _stop_and_reset_impl(self) -> None:
        """
        Stop current operation(s) (policy control and/or data collection) and reset.
        
        Stops policy control and/or data collection, asks user if they want to save, then resets robots to initial positions.
        """
        # Check if there's anything to stop
        if not self.policy_control_active and not self.data_collection_active and not getattr(self, "_save_pending", False):
            rospy.logwarn("[robot] Nothing to stop.")
            return
        
        if self.state == SystemState.SAVING:
            rospy.logwarn("[robot] Already saving, please wait...")
            return
        
        # Determine what we're stopping
        is_policy_control = self.policy_control_active
        is_data_collection = self.data_collection_active or getattr(self, "_save_pending", False)
        
        # Log what we're stopping
        if is_policy_control and is_data_collection:
            rospy.loginfo("[robot] Stopping policy control and data collection...")
        elif is_policy_control:
            rospy.loginfo("[robot] Stopping policy control...")
        elif is_data_collection:
            rospy.loginfo("[robot] Stopping data collection...")
        
        # Stop policy control first
        if is_policy_control:
            self.policy_control_active = False
            # Clear action queue
            while not self._action_queue.empty():
                try:
                    self._action_queue.get_nowait()
                except queue.Empty:
                    break
        
        # Stop data collection
        if is_data_collection:
            self.data_collection_active = False
        
        # Stop prediction thread
        self._prediction_thread_running = False
        if self._prediction_thread is not None and self._prediction_thread.is_alive():
            rospy.loginfo("[robot] Waiting for policy prediction thread to stop...")
            self._prediction_thread.join(timeout=2.0)
            if self._prediction_thread.is_alive():
                rospy.logwarn("[robot] Prediction thread did not stop gracefully.")
        
        # Critical: Stop servoJ control before calling reset() to avoid control conflict
        rospy.loginfo("[robot] Stopping servoJ control...")
        try:
            self.arm.stop_servo()
            time.sleep(0.1)
        except Exception as exc:
            rospy.logwarn(f"[robot] Error stopping servoJ: {exc}")
        
        # Ask user if they want to save data (only if data collection was active)
        if is_data_collection:
            self.state = SystemState.WAITING_SAVE_CONFIRM
            print("\n" + "=" * 60)
            print(f"[robot] Collection #{self._collection_count} completed.")
            print("[robot] Save this data? (y/n): ", end="", flush=True)
            
            # Wait for user input
            if self._stdin_available:
                try:
                    user_input = input().strip().lower()
                    if user_input == 'y':
                        saved = self._save_collection_data()
                        if self._zarr_enabled and self.save_data["rgb"] and not saved:
                            self._save_pending = True
                            self.state = SystemState.IDLE
                            rospy.logerr("[robot] Recording remains in memory. Fix the storage issue, then press 'q' to retry or discard. Do not exit the node.")
                            return
                    else:
                        rospy.loginfo("[robot] Data discarded.")
                except (EOFError, KeyboardInterrupt):
                    print("\n[robot] Input interrupted. Data discarded.")
            else:
                # If stdin is not available, skip saving
                rospy.logwarn("[robot] stdin not available. Data discarded.")
        
        self._save_pending = False

        # Reset to initial position
        super()._reset_robots()
        
        # Clear command buffers
        self.arm_cmd_target = None
        self.rapid_cmd_target = None
        self.arm_cmd = None
        self.rapid_cmd = None
        self.last_sent_arm_cmd = None
        self.last_sent_rapid_cmd = None
        self.last_message_time = None
        
        # Return to idle state
        self.state = SystemState.IDLE
        rospy.loginfo("[robot] ✓ Ready for next operation.")
        if self.auto_start_data_collection:
            rospy.loginfo("[robot] Press 'c' to start policy control and data collection.")
        else:
            rospy.loginfo("[robot] Press 'c' to start policy control or 's' to start data collection.")
        if is_data_collection:
            print("=" * 60)
    
    def _save_collection_data(self) -> bool:
        """Save collected data to zarr format with collection number in filename."""
        if not self._zarr_enabled:
            rospy.logwarn("[robot] Zarr saving is disabled.")
            return False
        
        if not self.save_data["rgb"]:
            rospy.logwarn("[robot] No data collected, skipping save.")
            return False
        
        if not self._session_dir:
            rospy.logwarn("[robot] No session directory. Data cannot be saved.")
            return False
        
        # Generate zarr filename with collection number (different from robot_node.py)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        zarr_filename = f"collection_{self._collection_count:03d}_{timestamp}.zarr"
        zarr_output_path = os.path.join(self._session_dir, zarr_filename)
        
        if os.path.exists(zarr_output_path):
            rospy.logwarn(f"[robot] Zarr path already exists: {zarr_output_path}")
            return False
        
        self.state = SystemState.SAVING
        
        # Save the fingertip images and separate third view using the custom path.
        saved = super()._save_to_zarr(zarr_path=zarr_output_path, include_tv_rgb=True)
        if saved:
            self._saved_any_data_this_session = True
        return saved
    
    
    def _print_help_message(self) -> None:
        """Print help message based on current state."""
        rospy.loginfo("[robot] Commands:")
        if not self.policy_control_active:
            if self.auto_start_data_collection:
                rospy.loginfo("[robot]   'c' - Start policy control and data collection")
            else:
                rospy.loginfo("[robot]   'c' - Start policy control")
        if not self.data_collection_active and not self.auto_start_data_collection:
            rospy.loginfo("[robot]   's' - Start data collection")
        if self.policy_control_active or self.data_collection_active:
            rospy.loginfo("[robot]   'q' - Stop current operation(s) and reset")
        
        if self.state == SystemState.WAITING_SAVE_CONFIRM:
            rospy.loginfo("[robot] Waiting for save confirmation (y/n)...")
        elif self.state == SystemState.SAVING:
            rospy.loginfo("[robot] Saving data, please wait...")
        elif self.policy_control_active and self.data_collection_active:
            rospy.loginfo("[robot] Status: Policy control active, Data collection active")
        elif self.policy_control_active:
            rospy.loginfo("[robot] Status: Policy control active")
        elif self.data_collection_active:
            rospy.loginfo("[robot] Status: Data collection active")

    def _collect_and_save_data(self, real_arm, real_hand) -> None:
        """
        Collect robot data and save to buffers.
        This is separated from policy prediction and action execution.
        """
        # Collect the fingertip images and separate third view.
        super()._collect_robot_data(real_arm, real_hand, include_tv_rgb=True)
    
    def _execute_actions_from_queue(self) -> None:
        """
        Execute actions from the prediction queue with interpolation for smooth control.
        Uses actions predicted by eval_fingervip.py methods.
        Executes num_steps actions starting from step 0 (pred_actions[0] to pred_actions[num_steps-1]).
        When num_steps=1, only the first step is executed.
        When num_steps>1, steps 0 to num_steps-1 are executed sequentially.
        
        Only gets new actions from queue when current sequence is fully executed and queue is ready.
        """
        # Load a sequence only after the previous final step has completed.
        if self._action_sequence_queue is None and not self._is_executing_sequence:
            # Get predicted actions from queue (non-blocking)
            try:
                pred_actions = self._action_queue.get_nowait()
                
                if pred_actions is not None:
                    try:
                        # pred_actions from eval_fingervip.py is a numpy array
                        # Shape could be (action_horizon, action_dim) or (action_dim,)
                        if pred_actions.ndim == 2:
                            # Extract actions from step 0 to num_steps-1
                            max_step = self.action_step_b  # This is num_steps - 1
                            if len(pred_actions) <= max_step:
                                rospy.logwarn(
                                    f"[robot] pred_actions has {len(pred_actions)} steps, "
                                    f"but requested {self.num_steps} steps (up to step {max_step}). "
                                    f"Using last available step."
                                )
                                # Fall back to last available action if out of bounds
                                self._action_sequence_queue = [pred_actions[-1]]
                            else:
                                # Extract actions from 0 to num_steps-1 (inclusive)
                                self._action_sequence_queue = [
                                    pred_actions[step] for step in self.action_steps_to_execute
                                ]
                        else:
                            # If it's a single action, use it directly
                            self._action_sequence_queue = [pred_actions]
                        
                        # Set first action target immediately and mark as executing
                        if len(self._action_sequence_queue) > 0:
                            self.current_action_target = self._action_sequence_queue[0]
                            self._action_sequence_idx = 0
                            self._action_step_cycles = 0
                            self._is_executing_sequence = True  # Mark that a sequence is being executed.
                        else:
                            self._action_sequence_queue = None
                            self._action_sequence_idx = 0
                            self._is_executing_sequence = False
                        
                    except Exception as e:
                        rospy.logerr(f"[robot] Error processing actions: {e}")
                        self._action_sequence_queue = None
                        self._action_sequence_idx = 0
                        self._is_executing_sequence = False
            except queue.Empty:
                # No actions available, will use interpolation if we have a target
                pass
        
        # Advance on the cycle after the current target's full interval.
        if self._is_executing_sequence and self._action_step_cycles >= self.policy_inference_interval:
            self._action_sequence_idx += 1
            self.current_action_target = self._action_sequence_queue[self._action_sequence_idx]
            self._action_step_cycles = 0

        # Always execute action (with interpolation between policy updates)
        if self.current_action_target is not None:
            if self.last_action is None:
                # First action, use directly
                current_action = self.current_action_target.copy()
                self.last_action = current_action.copy()
            else:
                # Interpolate between last action and target for smooth control
                steps_remaining = self.policy_inference_interval - self._action_step_cycles
                alpha = 1.0 / max(steps_remaining, 1)
                current_action = self.last_action + alpha * (self.current_action_target - self.last_action)
                self.last_action = current_action.copy()
            
            # Process predicted action to robot commands
            self._joint_angle_policy(current_action)
        elif self.last_action is not None:
            # Use last action if no new target yet (safety check)
            self._joint_angle_policy(self.last_action)

        if self._is_executing_sequence:
            self._action_step_cycles += 1
            final_step = self._action_sequence_idx == len(self._action_sequence_queue) - 1
            if final_step and self._action_step_cycles >= self.policy_inference_interval:
                self._action_sequence_queue = None
                self._action_sequence_idx = 0
                self._action_step_cycles = 0
                self._is_executing_sequence = False
                # Discard queued predictions made before this sequence finished.
                while not self._action_queue.empty():
                    try:
                        self._action_queue.get_nowait()
                    except queue.Empty:
                        break

    def _handle_policy_control(self) -> None:
        """
        Handle policy prediction and action execution (independent of data collection).
        
        This method handles:
        1. Policy prediction: Uses eval_fingervip.py methods to predict actions
        2. Action execution: Executes predicted actions from queue with interpolation
        
        All policy prediction uses HWDiffusionPolicyEval.predict_actions from eval_fingervip.py.
        """
        if not self.policy_control_active:
            return
        
        # Collect robot data for policy prediction
        real_arm = self.arm.get_arm_data()
        real_hand = self.rapid_hand.get_hand_data()
        
        # Policy prediction (using eval_fingervip.py methods)
        # Only predict new actions when:
        # 1. Not currently executing an action sequence, AND
        # 2. Action queue is empty (all previous actions have been consumed)
        # This ensures we execute the current sequence completely before getting new predictions
        self.policy_inference_counter += 1
        should_run_policy = (self.policy_inference_counter % self.policy_inference_interval == 0)
        
        # Only run policy prediction if we're not executing a sequence and queue is empty
        queue_is_empty = self._action_queue.empty()
        ready_for_new_prediction = not self._is_executing_sequence and queue_is_empty
        
        if should_run_policy and ready_for_new_prediction:
            # Use HWDiffusionPolicyEval.predict_actions from eval_fingervip.py
            # This is the ONLY place where policy prediction happens
            self._frame_idx += 1
            self._predict_action_async(real_arm, real_hand, self._frame_idx)
        
        # Execute actions from queue (separated from prediction)
        # Execute predicted actions with interpolation for smooth control
        self._execute_actions_from_queue()

    def _handle_robot_data(self) -> None:
        """
        Handle robot data collection only (independent of policy control).
        
        This method only collects and saves data, without policy prediction or action execution.
        """
        if not self.data_collection_active:
            return
        
        # Collect robot data
        real_arm = self.arm.get_arm_data()
        real_hand = self.rapid_hand.get_hand_data()
        
        # Save data for later analysis
        self._collect_and_save_data(real_arm, real_hand)


    def _shutdown(self) -> None:
        """Shutdown handler. Stop robots gracefully."""
        # ROS and the control loop can both request shutdown.
        if getattr(self, "_shutdown_started", False):
            return
        if getattr(self, "_stop_in_progress", False):
            self._shutdown_requested = True
            rospy.logwarn("[robot] Shutdown requested. Waiting for the current stop/save/reset to finish.")
            return
        self._shutdown_started = True

        # If no data was saved this session, remove the session directory
        if not self._saved_any_data_this_session and self._session_dir and os.path.exists(self._session_dir):
            try:
                if not os.listdir(self._session_dir):
                    os.rmdir(self._session_dir)
                    rospy.loginfo(f"[robot] Removed empty session directory (no data saved): {self._session_dir}")
            except Exception as exc:
                rospy.logdebug(f"[robot] Could not remove session directory: {exc}")

        # Stop camera display
        if self._camera_display_running:
            rospy.loginfo("[robot] Stopping camera display...")
            self._camera_display_running = False
            if self._camera_display_thread is not None and self._camera_display_thread.is_alive():
                self._camera_display_thread.join(timeout=1.0)
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        # If we have active operations, stop them first
        if self.policy_control_active or self.data_collection_active or getattr(self, "_save_pending", False):
            rospy.loginfo("[robot] Shutdown requested. Stopping current operation(s)...")
            self._stop_and_reset()

        if getattr(self, "_save_pending", False):
            rospy.logerr("[robot] Shutdown save failed. The recording is still only in memory and will be lost when this process exits.")
        
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
                self._poll_start_signal()
                self._send_robot_commands()

                try:
                    # Policy control (independent of data collection)
                    if self.policy_control_active:
                        self._handle_policy_control()

                    # Data collection (independent of policy control)
                    if self.data_collection_active:
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
        if self._session_dir:
            rospy.loginfo(f"[robot] Session directory: {self._session_dir}")
            rospy.loginfo("[robot] All data will be saved to this directory during this session.")
        save_cfg = self.robot_args.get("save_config", {})
        if not save_cfg.get("save_robot_data", False) and not save_cfg.get("save_robot_data_zarr", False):
            rospy.logwarn("[robot] Data saving is disabled.")
        rospy.loginfo("[robot] Interaction flow:")
        if self.auto_start_data_collection:
            rospy.loginfo("[robot]   'c' - Start policy control and data collection (auto-start enabled)")
        else:
            rospy.loginfo("[robot]   'c' - Start policy control (execute policy)")
            rospy.loginfo("[robot]   's' - Start data collection (can start anytime)")
        rospy.loginfo("[robot]   'q' - Stop current operation(s) and reset")
        if not self.auto_start_data_collection:
            rospy.loginfo("[robot] Note: Policy control and data collection can be used independently")
        rospy.loginfo("=" * 60)

    def _announce_device_ready(self) -> None:
        if self._device_ready_announced:
            return
        rospy.loginfo("[robot] Devices ready. Waiting for teleoperation joint commands...")
        self._device_ready_announced = True

    def _announce_start_prompt_if_ready(self) -> None:
        if self.data_collection_active or self._start_prompt_announced:
            return

        if self._teleop2arm is None or self._teleop2rapid is None:
            return

        if self._stdin_available:
            rospy.loginfo("[robot] All systems ready. Press 'c' to start policy control or 's' to start data collection.")
        else:
            rospy.logwarn("[robot] Non-interactive terminal detected; automatically starting policy control.")
            self._start_policy_control()
        self._start_prompt_announced = True
    
    def _start_camera_display(self) -> None:
        """Start camera display thread if fingertip cameras are available."""
        if not hasattr(self.rapid_hand, '_hand') or not hasattr(self.rapid_hand._hand, 'fingertip_camera'):
            rospy.logwarn("[robot] Fingertip cameras not available, skipping display.")
            return
        
        if not self.rapid_hand._hand.use_fingertip_camera:
            rospy.logwarn("[robot] Fingertip cameras not enabled, skipping display.")
            return
        
        fingertip_camera = self.rapid_hand._hand.fingertip_camera
        if fingertip_camera is None:
            rospy.logwarn("[robot] Fingertip camera system not initialized, skipping display.")
            return
        
        self._camera_display_running = True
        self._camera_display_thread = threading.Thread(
            target=self._camera_display_loop,
            args=(fingertip_camera,),
            daemon=True
        )
        self._camera_display_thread.start()
        rospy.loginfo("[robot] Camera display started.")
    
    def _camera_display_loop(self, fingertip_camera) -> None:
        """Display loop for fingertip cameras (runs in separate thread)."""
        try:
            # Use unified display method with exit flag for ROS compatibility
            # This leverages all the new features: window arrangement, USB address display, etc.
            fingertip_camera.display(
                check_exit_flag=lambda: not self._camera_display_running or rospy.is_shutdown()
            )
        except Exception as e:
            rospy.logerr(f"[robot] Error in camera display: {e}")


# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def parse_mask(s):
    s = s.replace("[", "").replace("]", "")
    return np.array([int(x) for x in s.split(",")])

def parse_args() -> argparse.Namespace:
    # Resolve default configuration paths relative to this script.
    default_config = ROOT_DIR / "Teleop" / "args" / "teleop_args.yaml"
    default_eval_config = ROOT_DIR / "Teleop" / "args" / "eval_args.yaml"
    default_checkpoint = ""
    
    parser = argparse.ArgumentParser(
        description="FingerViP policy evaluation ROS node (UR5e + RAPID Hand)"
    )
    parser.add_argument(
        "robot_args",
        type=str,
        nargs='?',  # Make the positional argument optional.
        default=str(default_config),
        help=f"Path to robot_args YAML configuration file (default: {default_config})",
    )

    parser.add_argument(
        "checkpoint_path",
        type=str,
        nargs='?',    
        default=default_checkpoint,
        help="Path to diffusion policy checkpoint file (.ckpt), relative to project root",
    )

    parser.add_argument(
        "handbase_transform_mask",
        type=parse_mask,
        nargs='?',
        default=[1, 1, 1, 1, 1]
    )
    
    # Parse command line arguments first
    args = parser.parse_args()
    
    # Load eval_args.yaml to get num_steps, optional checkpoint_path, and display config
    eval_config_path = Path(default_eval_config)
    eval_config = {}
    if eval_config_path.exists():
        try:
            with open(eval_config_path, 'r') as f:
                eval_config = yaml.safe_load(f) or {}
            
            # Get and validate num_steps parameter
            num_steps = eval_config.get('num_steps', 1)
            
            # Validate parameter
            if not (1 <= num_steps <= 16):
                raise ValueError(
                    f"Invalid num_steps in {eval_config_path}: {num_steps}. "
                    f"Requires: 1 <= num_steps <= 16"
                )
            
            args.num_steps = int(num_steps)

            # Optionally override checkpoint_path if provided in eval_args.yaml
            cfg_ckpt = eval_config.get('checkpoint_path')
            if cfg_ckpt and args.checkpoint_path == default_checkpoint:
                args.checkpoint_path = str(cfg_ckpt)
        except Exception as e:
            print(f"[WARN] Failed to load eval_args.yaml: {e}. Using default (num_steps=1)")
            args.num_steps = 1
            # leave checkpoint_path as parsed (CLI or default)
    else:
        # Use default if file doesn't exist
        args.num_steps = 1
    
    # Add eval_config to args for passing to RobotSystem
    args.eval_config = eval_config
    
    return args


def main() -> None:
    args = parse_args()
    checkpoint_path = str(ROOT_DIR / args.checkpoint_path)
    
    rospy.loginfo(f"[robot] Using checkpoint: {checkpoint_path}")
    rospy.loginfo(f"[robot] Action execution: num_steps={args.num_steps} (executing steps 0 to {args.num_steps-1})")
    
    robot_system = RobotSystem(robot_args_path=args.robot_args, 
                               ckpt_path=checkpoint_path,
                               eval_config=args.eval_config,
                               handbase_transform_mask=np.array(args.handbase_transform_mask),
                               in_res=(640, 480),
                               out_res=(224, 224),
                               urdf_file_name="ur5e_with_rapid_hand_right_fingercamera",
                               num_steps=args.num_steps)  
    try:
        robot_system.run()
    except rospy.ROSInterruptException:
        pass  # Graceful exit on Ctrl‑C within  ROS


if __name__ == "__main__":  # pragma: no cover
    main()
