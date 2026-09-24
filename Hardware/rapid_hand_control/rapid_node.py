import numpy as np
import os
import stat
import threading


class RapidNode:
    """
    Core data aggregation interface for the RapidHand platform.

    - Manages servo (U2D2) initialization and kinematics.
    - Collects PCB tactile data and aligns into 3D positions.
    """

    num_joint: int = 20
    tactile_tensor_shape: tuple[int, int, int] = (5, 12, 8)
    tactile_order_str: list[str] = ["IDIP", "MDIP", "RDIP", "LDIP", "TDIP"]

    def __init__(
        self,
        motor_port: str,
        pcb_port: str,
        tactile_order_dict: dict,
        joint_command_order_list: list,
    ):
        self.use_motor = self._check_port(motor_port, "motor_port")
        self.use_pcb = self._check_port(pcb_port, "pcb_port")
        self.tactile_order_dict = tactile_order_dict

        self.hand_joint_positions = np.zeros(self.num_joint)
        self.hand_joint_velocity = np.zeros(self.num_joint)
        self.hand_joint_current = np.zeros(self.num_joint)
        self.tactile_tensor = np.zeros(self.tactile_tensor_shape)
        self.tactile_tensor_3D = np.zeros((*self.tactile_tensor_shape, 4))

        if self.use_motor:
            self._init_motor(motor_port, joint_command_order_list)

        if self.use_pcb:
            self._init_pcb(pcb_port)

        print("[RapidNode] Initialized successfully.")

    def _init_motor(self, port: str, joint_order: list):
        from servo_driver.rapid_driver import RapidDriver
        from control import PinocchioHandControl

        self.motor_driver = RapidDriver(port=port)
        self.motor_driver.set_pos(np.zeros(self.num_joint))

        self.hand_model = PinocchioHandControl(
            hand_name="rapid_hand", robot_urdf_file="rapid_hand/rapid_hand_right.urdf"
        )

        self.real2pin = [
            joint_order.index(name) for name in self.hand_model.get_joint_names()
        ]

        self.tactile_order_list = [
            self.tactile_order_dict[finger] for finger in self.tactile_order_str
        ]

        self._load_tactile_positions()

    def _load_tactile_positions(self):
        import pandas as pd
        import re

        def parse_cell(cell):
            nums = re.findall(r"-?\d+\.?\d*", cell)
            return [float(x) * 1e-3 for x in nums] + [1]

        filepath = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "tactile_relative_position.xlsx")
        )
        df = pd.read_excel(filepath, header=None)
        single_patch = np.array(
            [[parse_cell(cell) for cell in row] for row in df.values]
        )
        self.tactiles_position_3D_H = np.repeat(
            single_patch[None, :, :], 5, axis=0
        )  # 5*12*8*4
        self.tactiles_position_3D_H[4, :, :, 2] += 12e-3

    def _init_pcb(self, port: str, timeout: float = 1.0):
        from .pcb_comm.serial_comm import PCBSerialInterface

        self.serial_interface = PCBSerialInterface(port=port, timeout=timeout)

    def set_joint_angle(self, data: np.ndarray):
        if self.use_motor:
            self.motor_driver.set_pos(data)
            return None

    def get_data(self):
        """
        Collects all updated sensor and actuator data.
        """
        self.upd_data()
        return {
            "joint_positions": self.hand_joint_positions.copy(),
            "joint_velocities": self.hand_joint_velocity.copy(),
            "joint_current": self.hand_joint_current.copy(),
            "touch": self.tactile_tensor.copy(),
            "touch_3D": self.tactile_tensor_3D.copy(),
        }

    def upd_data(self):
        """
        Core threaded data update. Reads motor states and the tactile matrix.
        """

        def update_motor_state():
            if self.use_motor:
                (
                    self.hand_joint_positions,
                    self.hand_joint_velocity,
                    self.hand_joint_current,
                ) = self.motor_driver.read_pos_vel_cur()
                self.update_tactile_positions()

        def update_tactile_data():
            if self.use_pcb:
                try:
                    tactile_data = (
                        self.serial_interface.trigger_and_read()
                    )  # shape=(5, 96)
                except Exception as e:
                    raise RuntimeError(f"Tactile read failed: {e}")

                # Reshape and reorder tactile data
                self.tactile_tensor = tactile_data.reshape(self.tactile_tensor_shape)
                self.tactile_tensor = np.flip(self.tactile_tensor, axis=(1, 2))[
                    self.tactile_order_list
                ]

        thread1 = threading.Thread(target=update_motor_state)
        thread2 = threading.Thread(target=update_tactile_data)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        self.tactile_tensor_3D[..., -1] = self.tactile_tensor

    def update_tactile_positions(self):
        q = self.hand_joint_positions[self.real2pin]
        self.hand_model.set_current_qpos(q)

        transforms = np.array(
            [
                self.hand_model.get_link_transform(name)
                for name in self.tactile_order_str
            ]
        )
        num_fingers = self.tactile_tensor_shape[0]
        batch = self.tactiles_position_3D_H.reshape(num_fingers, -1, 4)  # (5, 96, 4)
        global_coords = np.einsum("fij,fkj->fki", transforms, batch)
        self.tactile_tensor_3D = global_coords.reshape(*self.tactile_tensor_shape, 4)

    def stop_process(self):
        """Gracefully close any opened devices."""
        if hasattr(self, "serial_interface"):
            self.serial_interface.close()

    @staticmethod
    def _check_port(path: str, label: str) -> bool:
        """
        Check whether the given path exists and is a character device.
        """
        if os.path.exists(path):
            try:
                st = os.stat(path)
                if stat.S_ISCHR(st.st_mode):
                    print(f"[✓] {label}: Found character device at {path}")
                    return True
                else:
                    print(f"[!] {label}: {path} exists but is not a character device")
            except PermissionError as e:
                print(f"[!] {label}: {path} exists but cannot be accessed ({e})")
                return True  # Device exists but permission is denied
        else:
            print(f"[✗] {label}: Path not found at {path}")
        return False
