import numpy as np
import cv2
import time
import os
import stat
import threading


class MotorFeedbackError(RuntimeError):
    """Motor feedback could not be refreshed for the current sample."""


class RapidNode:
    """
    Core data aggregation interface for the RapidHand platform.

    - Manages servo (U2D2) initialization and joint feedback.
    - Retrieves RealSense RGB-D images and fingertip camera frames (optional).
    """

    fingertip_frame_timeout: float = 2.0
    num_joint: int = 20

    def __init__(
        self,
        motor_port: str,
        video_port:str,
        joint_command_order_list: list,
        use_fingertip_camera: bool,
        use_third_view_camera: bool = False,
        third_view_serial_number: str = None,
    ):
        self.image_width = 640
        self.image_height = 480
        self.fingertip_camera_maximum_failure_rate = 0.0

        self.use_motor = self._check_port(motor_port, "motor_port")
        # self.use_fingertip = self._check_port(video_port, "video_port")
        self.use_fingertip_camera = use_fingertip_camera # and self.use_fingertip
        self.use_third_view_camera = use_third_view_camera
        self.third_view_serial_number = third_view_serial_number

        # Debug counters for throttling log messages
        self._camera_update_warn_counter = 0

        self.hand_joint_positions = np.zeros(self.num_joint)
        self.hand_joint_velocity = np.zeros(self.num_joint)
        self.hand_joint_current = np.zeros(self.num_joint)
        self.third_view_rgb_image_array =  np.zeros((1, self.image_width, self.image_height, 3), dtype=np.uint8)
        self.third_view_depth_image_array =  np.zeros((1, self.image_width, self.image_height))
        self.fingertip_rgb_image_array = np.zeros((5, self.image_width, self.image_height, 3), dtype=np.uint8)
        self.fingertip_camera_validation_masks = np.zeros((5,), dtype=np.uint8)

        # Thread locks for camera data arrays
        self.third_view_camera_lock = threading.Lock()

        if self.use_motor:
            self._init_motor(motor_port, joint_command_order_list)

        if self.use_third_view_camera:
            self._init_third_view_camera(type="realsense", serial_number=self.third_view_serial_number)

        if self.use_fingertip_camera:
            self._init_fingertip_camera()

        print("[RapidNode] Initialized successfully.")

    def _init_motor(self, port: str, joint_order: list):
        from .servo_driver.rapid_driver import RapidDriver

        self.motor_driver = RapidDriver(port=port)
        self.motor_driver.set_pos(np.zeros(self.num_joint))

    def _init_third_view_camera(self, type: str = "realsense", serial_number: str = None):
        if type == "realsense":
            from .camera_port.realsensemanager import RealSenseCamera

            self.third_view_camera = RealSenseCamera(serial_number)
            self.third_view_camera.init_camera(width=self.image_width, height=self.image_height)
        else:
            raise ValueError(f"Unsupported camera type: {type}")
        
    def _init_fingertip_camera(self):
        from .fingertip_camera.dicota4kusbcamera import DexterousHandPerception

        self.fingertip_camera = DexterousHandPerception()
        self.fingertip_camera.init_camera()
        self.fingertip_camera.start_all()

    def set_joint_angle(self, data: np.ndarray):
        if self.use_motor:
            self.motor_driver.set_pos(data)
            return None

    def get_data(self):
        """
        Collects all updated sensor and actuator data.
        """
        self.upd_data()
        with self.third_view_camera_lock:
            third_view_rgb = self.third_view_rgb_image_array.copy()
            third_view_depth = self.third_view_depth_image_array.copy()
        return {
            "third_view_rgb": third_view_rgb,
            "third_view_depth": third_view_depth,
            "fingertip_rgb": self.fingertip_rgb_image_array.copy(),
            "fingertip_masks": self.fingertip_camera_validation_masks.copy(),
            "joint_positions": self.hand_joint_positions.copy(),
            "joint_velocities": self.hand_joint_velocity.copy(),
            "joint_current": self.hand_joint_current.copy(),
        }

    def upd_data(self):
        """
        Read motor states and camera images in worker threads.
        """

        motor_errors = []

        def update_motor_state():
            if self.use_motor:
                try:
                    (
                        self.hand_joint_positions,
                        self.hand_joint_velocity,
                        self.hand_joint_current,
                    ) = self.motor_driver.read_pos_vel_cur()
                except Exception as exc:
                    motor_errors.append(exc)

        camera_errors = []

        def update_realsense(camera, label, lock, rgb_attr, depth_attr):
            try:
                if camera is None or not getattr(camera, "running", False):
                    raise TimeoutError(f"{label} RealSense camera is not running")
                bgr, depth = camera.get_latest_frames(device_index=0, timeout=0.5)
                if bgr is None or depth is None:
                    raise TimeoutError(f"{label} RealSense camera returned no fresh frames within 0.5s")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with lock:
                    setattr(self, rgb_attr, np.expand_dims(rgb, axis=0))
                    setattr(self, depth_attr, np.expand_dims(depth, axis=0))
            except Exception as exc:
                # Propagate capture failures through the existing stop/save/reset flow.
                camera_errors.append(TimeoutError(f"{label} RealSense capture failed: {exc}"))

        def update_third_view_camera_data():
            if self.use_third_view_camera:
                update_realsense(getattr(self, "third_view_camera", None), "Third-view",
                                 self.third_view_camera_lock, "third_view_rgb_image_array",
                                 "third_view_depth_image_array")

        def update_fingertip_camera_data(display: bool = False):
            if self.use_fingertip_camera:
                all_frames, _, masks = self.fingertip_camera.get_latest_frames(bgr2rgb=True) # this is a rgb list
                
                deadline = time.monotonic() + self.fingertip_frame_timeout
                while_count = 0
                while np.sum(masks) < self.fingertip_camera.camera_count * (1 - self.fingertip_camera_maximum_failure_rate):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Fingertip cameras missing frames for {self.fingertip_frame_timeout}s: "
                            f"{np.where(masks == 0)[0].tolist()}"
                        )
                    zero_indices = np.where(masks == 0)[0]
                    print(f"[WARN] Cameras missing frames: {zero_indices.tolist()}")

                    print("Retrying RGB capture...")
                    time.sleep(0.5)
                    all_frames, _, masks = self.fingertip_camera.get_latest_frames(bgr2rgb=True)
                    while_count += 1

                if while_count > 0:
                    print("Fingertip camera data updated successfully.")
                
                self.fingertip_rgb_image_array = all_frames
                self.fingertip_camera_validation_masks = masks

                if display:
                    self.fingertip_camera.display()

        def update_fingertip_worker():
            try:
                update_fingertip_camera_data()
            except Exception as exc:
                # Forward worker errors to the caller instead of returning stale frames.
                camera_errors.append(exc)

        thread1 = threading.Thread(target=update_motor_state)
        thread4 = threading.Thread(target=update_fingertip_worker, daemon=True)
        thread5 = threading.Thread(target=update_third_view_camera_data)

        thread1.start()
        thread4.start()
        thread5.start()

        thread1.join()
        thread4.join()
        thread5.join()
        if motor_errors:
            raise MotorFeedbackError(f"Motor feedback failed: {motor_errors[0]}") from motor_errors[0]
        if camera_errors:
            raise camera_errors[0]

    def stop_process(self):
        """Gracefully close any opened devices."""
        # Stop fingertip cameras first
        if hasattr(self, "fingertip_camera") and self.fingertip_camera is not None:
            try:
                self.fingertip_camera.stop_all()
            except Exception as e:
                print(f"[WARN] Error stopping fingertip cameras: {e}")
        
        # Stop third view camera
        if hasattr(self, "third_view_camera") and self.third_view_camera is not None:
            try:
                if hasattr(self.third_view_camera, "stop_camera"):
                    self.third_view_camera.stop_camera()
            except Exception as e:
                print(f"[WARN] Error stopping third view camera: {e}")
        

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
