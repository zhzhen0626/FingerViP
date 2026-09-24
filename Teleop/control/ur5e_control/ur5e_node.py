import numpy as np
import time
from typing import Optional

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False
    RTDEControlInterface = None
    RTDEReceiveInterface = None

# Constants
_JOINT_DOF = 6
_DEFAULT_RTDE_FREQUENCY = 125
_DEFAULT_DECELERATION = 0.5


class UR5eNode:
    """
    Low-level UR5e communication interface using RTDE.

    This class wraps RTDE communication with the UR5e robot,
    providing methods for joint position control and state feedback.
    """

    def __init__(self, robot_ip: str, rtde_frequency: int = _DEFAULT_RTDE_FREQUENCY) -> None:
        """
        Initialize connection to UR5e via RTDE.

        Parameters
        ----------
        robot_ip : str
            IP address of the UR5e robot.
        rtde_frequency : int
            RTDE communication frequency in Hz (default: 100).
        """
        if not RTDE_AVAILABLE:
            raise RuntimeError(
                "rtde_control/rtde_receive modules not available. "
                "Please install ur-rtde: pip install ur-rtde"
            )
        
        self.robot_ip = robot_ip
        self.rtde_frequency = rtde_frequency
        self.rtde_c: Optional[RTDEControlInterface] = None
        self.rtde_r: Optional[RTDEReceiveInterface] = None
        
        self._connect()

    def _connect(self, max_retries: int = 3, retry_delay: float = 1.0) -> None:
        """Establish RTDE connections with retry logic.
        
        Args:
            max_retries: Maximum number of connection attempts (default: 3)
            retry_delay: Delay between retry attempts in seconds (default: 1.0)
        """
        last_exception = None
        for attempt in range(max_retries):
            try:
                self.rtde_c = RTDEControlInterface(self.robot_ip, self.rtde_frequency)
                self.rtde_r = RTDEReceiveInterface(self.robot_ip, self.rtde_frequency)
                
                if self.rtde_r.isConnected():
                    return  # Successfully connected
                else:
                    # Clean up failed connections
                    if self.rtde_c is not None:
                        try:
                            self.rtde_c.disconnect()
                        except Exception:
                            pass
                        self.rtde_c = None
                    if self.rtde_r is not None:
                        try:
                            self.rtde_r.disconnect()
                        except Exception:
                            pass
                        self.rtde_r = None
                    raise RuntimeError(f"Connection established but not connected to UR5e at {self.robot_ip}")
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to connect to UR5e at {self.robot_ip} after {max_retries} attempts: {e}"
                    ) from last_exception

    def _validate_joint_positions(self, joint_positions: np.ndarray) -> np.ndarray:
        """Validate and convert joint positions to the correct format."""
        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        if joint_positions.shape != (_JOINT_DOF,):
            raise ValueError(f"Expected joint_positions shape ({_JOINT_DOF},), got {joint_positions.shape}")
        return joint_positions

    def move_to_joint_positions(
        self,
        joint_positions: np.ndarray,
        acceleration: float = 0.2,
        velocity: float = 0.25,
        async_move: bool = False,
    ) -> None:
        """
        Send a command to move the UR5e to the given joint positions using moveJ.

        Parameters
        ----------
        joint_positions : np.ndarray
            Desired joint angles (6 values, in radians).
        acceleration : float
            Acceleration setting (rad/s^2).
        velocity : float
            Velocity setting (rad/s).
        async_move : bool
            If True, move asynchronously (non-blocking).
        """
        if self.rtde_c is None:
            raise RuntimeError("RTDE control interface not initialized")
        
        joint_positions = self._validate_joint_positions(joint_positions)
        self.rtde_c.moveJ(joint_positions.tolist(), velocity, acceleration, async_move)

    def servo_to_joint_positions(
        self,
        joint_positions: np.ndarray,
        speed: float = 1.2,
        acceleration: float = 0.2,
        dt: float = 0.008,
        lookahead_time: float = 0.1,
        gain: int = 200,
    ) -> None:
        """
        Send a servo command to move the UR5e to the given joint positions using servoJ.
        
        servoJ provides real-time control suitable for high-frequency control loops.
        It requires continuous calls at a consistent rate (typically >125Hz).

        Parameters
        ----------
        joint_positions : np.ndarray
            Desired joint angles (6 values, in radians).
        speed : float
            Movement speed (rad/s).
        acceleration : float
            Movement acceleration (rad/s^2).
        dt : float
            Control period in seconds (e.g., 0.008 for 125Hz, 0.01 for 100Hz).
        lookahead_time : float
            Smooth motion lookahead time in seconds (0.1-0.2s recommended).
        gain : int
            Servo gain (higher value -> stronger control, typically 100-300).
        """
        if self.rtde_c is None:
            raise RuntimeError("RTDE control interface not initialized")
        
        joint_positions = self._validate_joint_positions(joint_positions)
        self.rtde_c.servoJ(joint_positions.tolist(), speed, acceleration, dt, lookahead_time, gain)

    def get_feedback(self) -> dict:
        """
        Return current joint positions and relevant state information.

        Returns
        -------
        dict
            A dictionary containing the current state:
            - joint_positions: np.ndarray (6,) - Current joint angles in radians
            - joint_velocities: np.ndarray (6,) - Current joint velocities in rad/s
            - joint_current: np.ndarray (6,) - Current joint currents in A
            - eef_speed: np.ndarray (6,) - End-effector speed (linear + angular)
            - tcp_pose: np.ndarray (6,) - TCP pose [x, y, z, rx, ry, rz]
            - ee_pos_quat: np.ndarray (7,) - End-effector pose [x, y, z, rx, ry, rz, 0]
            
        Raises
        ------
        RuntimeError
            If RTDE receive interface is not initialized or connection is lost
        """
        if self.rtde_r is None:
            raise RuntimeError("RTDE receive interface not initialized")
        
        if not self.rtde_r.isConnected():
            raise RuntimeError(f"RTDE connection to UR5e at {self.robot_ip} is lost")
        
        try:
            # Batch fetch all data for better performance
            tcp_pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=np.float64)
            
            return {
                "joint_positions": np.asarray(self.rtde_r.getActualQ(), dtype=np.float64),
                "joint_velocities": np.asarray(self.rtde_r.getActualQd(), dtype=np.float64),
                "joint_current": np.asarray(self.rtde_r.getActualCurrent(), dtype=np.float64),
                "eef_speed": np.asarray(self.rtde_r.getActualTCPSpeed(), dtype=np.float64),
                "tcp_pose": tcp_pose,
                "ee_pos_quat": np.concatenate([tcp_pose, [0.0]]),  # [x, y, z, rx, ry, rz, 0]
            }
        except Exception as e:
            # Raise exception instead of returning zeros to allow upper layers to handle errors
            raise RuntimeError(f"Error reading robot state from UR5e at {self.robot_ip}: {e}") from e

    def stop(self) -> None:
        """Stop current robot movement."""
        if self.rtde_c is None:
            return
        
        # Try to stop servo first, then joint movement
        try:
            self.rtde_c.servoStop()
        except Exception:
            pass
        try:
            self.rtde_c.stopJ(_DEFAULT_DECELERATION)
        except Exception:
            pass

    def close(self) -> None:
        """Cleanly close connection to the UR5e."""
        if self.rtde_c is not None:
            try:
                self.rtde_c.stopScript()
                self.rtde_c.disconnect()
            except Exception:
                pass
            self.rtde_c = None
        
        if self.rtde_r is not None:
            try:
                self.rtde_r.disconnect()
            except Exception:
                pass
            self.rtde_r = None

