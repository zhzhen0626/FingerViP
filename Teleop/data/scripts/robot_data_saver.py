import os
import datetime
import numpy as np
import pickle
import threading
from typing import Dict


class PklSaver:
    """
    Asynchronous data saver for arm-hand control systems.
    Saves camera, joint, and command data into timestamped pickle files.

    Args:
        task_name (str): Name of the task directory under `root_dir`.
        root_dir (str): Root directory where data will be saved.
    """

    def __init__(self, task_name: str = "test_data", root_dir: str = "./data"):
        self.task_name = task_name
        self.task_dir = os.path.abspath(os.path.join(root_dir, task_name))
        os.makedirs(self.task_dir, exist_ok=True)
        self.frame_dir = None

        self.hand_data_keys_set = {
            "third_view_rgb",
            "third_view_depth",
            "fingertip_rgb",
            "joint_positions",
            "joint_velocities",
            "joint_current",
        }

        self.arm_data_keys_set = {
            "joint_positions",
            "joint_velocities",
            "joint_current",
            "eef_speed",
            "ee_pos_quat",
        }

    def save_frame(
        self,
        real_arm_data: Dict,
        real_hand_data: Dict,
        command: np.ndarray,
    ):
        """
        Save a frame asynchronously. Performs key validation.

        Args:
            real_arm_data (dict): Dictionary containing arm sensor and kinematic data.
            real_hand_data (dict): Dictionary containing hand sensor data and images.
            command (np.ndarray): Control command data for logging.
        """
        if not self.arm_data_keys_set.issubset(real_arm_data):
            missing = self.arm_data_keys_set - real_arm_data.keys()
            print("[PklSaver] Arm data is missing keys:", missing)
            return

        if not self.hand_data_keys_set.issubset(real_hand_data):
            missing = self.hand_data_keys_set - real_hand_data.keys()
            print("[PklSaver] Hand data is missing keys:", missing)
            return

        if self.frame_dir is None:
            self.start_new_session()

        timestamp = datetime.datetime.now()
        thread = threading.Thread(
            target=self._save_frame_to_file,
            args=(real_arm_data, real_hand_data, command.copy(), timestamp),
            daemon=True
        )
        thread.start()

    def _save_frame_to_file(
        self,
        real_arm_data: Dict,
        real_hand_data: Dict,
        command: np.ndarray,
        timestamp: datetime.datetime,
    ):
        """
        Save combined arm-hand-command data into a timestamped pickle file.

        Args:
            real_arm_data (dict): Arm data dictionary.
            real_hand_data (dict): Hand data dictionary.
            command (np.ndarray): Control command.
            timestamp (datetime.datetime): Timestamp to name the file.
        """
        if self.frame_dir is None:
            raise RuntimeError("Frame directory not initialised. Call start_new_session() before saving data.")

        try:
            os.makedirs(self.frame_dir, exist_ok=True)
            filename = timestamp.isoformat().replace(":", "_").replace(".", "_") + ".pkl"
            save_path = os.path.join(self.frame_dir, filename)

            saved_data = {
                "third_view_rgb": real_hand_data["third_view_rgb"],
                "third_view_depth": real_hand_data["third_view_depth"],
                "fingertip_rgb": real_hand_data["fingertip_rgb"],
                "joint_positions": np.concatenate([
                    real_arm_data["joint_positions"],
                    real_hand_data["joint_positions"]
                ]),
                "joint_velocities": np.concatenate([
                    real_arm_data["joint_velocities"],
                    real_hand_data["joint_velocities"]
                ]),
                "joint_current": np.concatenate([
                    real_arm_data["joint_current"],
                    real_hand_data["joint_current"]
                ]),
                "eef_speed": real_arm_data["eef_speed"],
                "ee_pos_quat": real_arm_data["ee_pos_quat"],
                "control": command,
            }

            with open(save_path, "wb") as f:
                pickle.dump(saved_data, f)

        except Exception as e:
            print(f"[PklSaver] Failed to save frame: {e}")

    def start_new_session(self) -> str:
        """
        Create a new subfolder within the task directory using a fresh timestamp.
        Returns the absolute path to the newly created folder.
        """
        time_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
        self.frame_dir = os.path.join(self.task_dir, time_str)
        os.makedirs(self.frame_dir, exist_ok=True)
        return self.frame_dir
