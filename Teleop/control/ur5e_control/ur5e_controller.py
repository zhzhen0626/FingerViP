from __future__ import annotations

from typing import Optional, Union, Dict, List
import warnings
import numpy as np

# Support both relative import (when used as package) and absolute import (when run directly)
try:
    from .ur5e_node import UR5eNode
except ImportError:
    from ur5e_node import UR5eNode

# Constants
_JOINT_DOF = 6


class UR5eController:
    """
    High-level synchronous controller for UR5e robotic arm.

    Args:
        robot_ip (str): IP address of the UR5e robot.
        init_qpos (Optional[Union[np.ndarray, Dict[str, float]]]): Initial joint configuration.
            Can be:
            - np.ndarray: Array of 6 joint angles in radians
            - Dict[str, float]: Dictionary mapping joint names to angles (requires joint_order)
            - None: Uses the built-in joint pose if auto_reset=True, or current position if auto_reset=False
        joint_order (Optional[List[str]]): Joint name order for dictionary init_qpos.
            Required if init_qpos is a dictionary. Default order matches UR5e standard.
        control_dt (float): Control loop timestep (seconds).
        rtde_frequency (int): RTDE communication frequency in Hz.
        auto_reset (bool): Automatically reset to initial position on init.
        use_servo (bool): Use servoJ for real-time control (default: True). If False, uses moveJ.
        servo_speed (float): Servo speed (rad/s) for servoJ.
        servo_acceleration (float): Servo acceleration (rad/s^2) for servoJ.
        servo_lookahead_time (float): Servo lookahead time (s) for servoJ.
        servo_gain (int): Servo gain for servoJ.
    """

    def __init__(
        self,
        robot_ip: str,
        init_qpos: Optional[Union[np.ndarray, Dict[str, float]]] = None,
        joint_order: Optional[List[str]] = None,
        control_dt: float = 0.01,
        rtde_frequency: int = 100,
        auto_reset: bool = True,
        use_servo: bool = True,
        servo_speed: float = 0.2,
        servo_acceleration: float = 0.2,
        servo_lookahead_time: float = 0.1,
        servo_gain: int = 200,
    ) -> None:
        self._arm = UR5eNode(robot_ip=robot_ip, rtde_frequency=rtde_frequency)
        self._control_dt = control_dt
        self._use_servo = use_servo
        self._servo_speed = servo_speed
        self._servo_acceleration = servo_acceleration
        self._servo_lookahead_time = servo_lookahead_time
        self._servo_gain = servo_gain

        # Initialize joint configuration
        if init_qpos is None:
            if auto_reset:
                # Default initial position: reasonable pose for UR5e
                # shoulder_pan: 0.0, shoulder_lift: -1.57, elbow: 1.57,
                # wrist_1: 0.0, wrist_2: 1.57, wrist_3: 0.0
                self._init_qpos = np.array(
                    [0.0, -1.57, 1.57, 0.0, 1.57, 0.0],
                    dtype=np.float64
                )
            else:
                self._init_qpos = self._arm.get_feedback()["joint_positions"]
        elif isinstance(init_qpos, dict):
            # Convert dictionary to array using joint order
            if joint_order is None:
                # Default UR5e joint order
                joint_order = [
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ]
            if len(joint_order) != _JOINT_DOF:
                raise ValueError(f"joint_order must have {_JOINT_DOF} elements, got {len(joint_order)}")
            # Check for missing joints
            missing_joints = [name for name in joint_order if name not in init_qpos]
            if missing_joints:
                raise ValueError(
                    f"Missing joint positions in init_qpos: {missing_joints}. "
                    f"Available joints: {list(init_qpos.keys())}"
                )
            self._init_qpos = np.array(
                [float(init_qpos[name]) for name in joint_order],
                dtype=np.float64
            )
        else:
            # Array format
            self._init_qpos = np.asarray(init_qpos, dtype=np.float64)
            if self._init_qpos.shape != (_JOINT_DOF,):
                raise ValueError(f"init_qpos must have shape ({_JOINT_DOF},), got {self._init_qpos.shape}")

        if auto_reset:
            self.reset()

    def reset(
        self,
        velocity: float = 0.4,
        acceleration: float = 0.2,
    ) -> None:
        """
        Move the arm to its initial joint configuration using moveJ.
        
        Uses moveJ (point-to-point movement) for safe, smooth reset to initial position.
        This bypasses servo control and provides a single smooth trajectory.
        
        Args:
            velocity (float): Movement velocity (rad/s). Default 0.4.
            acceleration (float): Movement acceleration (rad/s^2). Default 0.2 for smooth motion.
        """
        self.move_to_joint_positions(
            self._init_qpos,
            velocity=velocity,
            acceleration=acceleration,
            async_move=False,
        )

    def set_joint_positions(
        self,
        joint_positions: np.ndarray,
        acceleration: float = 0.2,
        velocity: float = 0.2,
        async_move: bool = False,
    ) -> None:
        """
        Command the robot to move to the specified joint configuration.

        Args:
            joint_positions (np.ndarray): Target joint angles (6-element array).
            acceleration (float): Acceleration setting (rad/s^2).
            velocity (float): Velocity setting (rad/s).
            async_move (bool): If True, move asynchronously (non-blocking).
        """
        # Validation is handled by UR5eNode
        if self._use_servo:
            self._arm.servo_to_joint_positions(
                joint_positions,
                speed=self._servo_speed,
                acceleration=self._servo_acceleration,
                dt=self._control_dt,
                lookahead_time=self._servo_lookahead_time,
                gain=self._servo_gain,
            )
        else:
            self._arm.move_to_joint_positions(
                joint_positions,
                acceleration=acceleration,
                velocity=velocity,
                async_move=async_move,
            )

    # Alias for compatibility or convenience
    control_arm_qpos = set_joint_positions

    def move_to_joint_positions(
        self,
        joint_positions: np.ndarray,
        acceleration: float = 0.2,
        velocity: float = 0.2,
        async_move: bool = False,
    ) -> None:
        """
        Blocking moveJ command that bypasses servo control.

        Useful for coarse positioning or alignment where a single smooth
        trajectory is desired before entering high-frequency servo mode.
        """
        # Validation is handled by UR5eNode
        self._arm.move_to_joint_positions(
            joint_positions,
            acceleration=acceleration,
            velocity=velocity,
            async_move=async_move,
        )

    def get_arm_data(self) -> dict:
        """
        Retrieve current joint state and feedback data.

        Returns:
            dict: Contains current joint positions and other sensor feedback.
        """
        return self._arm.get_feedback()

    def stop_servo(self) -> None:
        """
        Stop servoJ control without closing the connection.
        
        This is useful when transitioning from servoJ to moveJ (e.g., for reset).
        Must be called before using moveJ to avoid "Another thread is already controlling the robot" error.
        """
        if self._arm is not None:
            self._arm.stop()

    def stop(self) -> None:
        """Safely stop and shutdown the controller."""
        try:
            if self._arm is not None:
                self._arm.stop()  # Stop robot movement first
        except Exception as e:
            # Log warning but continue with cleanup
            warnings.warn(f"Error stopping arm during shutdown: {e}")
        finally:
            if self._arm is not None:
                self._arm.close()  # Close connections

    def __enter__(self) -> UR5eController:
        """Context manager enter."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit, ensuring safe shutdown."""
        self.stop()
        return False  # Propagate any exceptions
