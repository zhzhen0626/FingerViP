import cv2
import threading
import time
import os
import glob
import csv    # Write timestamp files.
import json   # Write camera metadata.
import sys
from pathlib import Path
from queue import Queue
from typing import Optional, Tuple, List, Dict, Callable
import numpy as np

try:
    from .utils import detect_available_cameras, align_timestamps, load_usb_camera_metas, get_usb_address_by_camera_id
except ImportError:
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from utils import detect_available_cameras, align_timestamps, load_usb_camera_metas, get_usb_address_by_camera_id
# Shared capture start event.
start_event = threading.Event()

class SingleUsbCamera(threading.Thread):
    """Capture thread for one USB camera and its metadata."""
    def __init__(self, cam_meta, save=False, max_queue_size=10, backend=cv2.CAP_V4L2):
        super().__init__(daemon=True)
        self.meta = cam_meta     # Camera ID, name, path, USB port, resolution, and FPS.
        self.cam_id = cam_meta["id"]
        self.width, self.height = cam_meta["res"]
        self.fps_target = cam_meta["fps"]
        self.backend = backend
        self.save = save
        self.running = True

        self.frame = None
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.lock = threading.Lock()
        self.last_frame_time = None
        self.frame_timeout = 2.0
        self.fps = 0.0

        self.timestamps = []  # Per-frame timestamps.
        self.frame_indices = []  # Recorded frame indices.
        self.sync_start = None  # Capture start timestamp.

        # Track consecutive read failures.
        self.failure_count = 0
        self.max_consecutive_failures = 10  # Stop after ten consecutive read failures.
        self.failure_warn_interval = 10  # Log every ten consecutive failures.

        if save:
            os.makedirs("videos", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            filename = f"videos/cam{self.cam_id}_{self.meta['name'].replace(' ','_')}.avi"
            self.writer = cv2.VideoWriter(filename, fourcc, self.fps_target, (self.width, self.height))
            self.outfile = filename  # Recording path used for timestamp export.
        else:
            self.writer = None
            self.outfile = None

    def run(self):
        cap = cv2.VideoCapture(self.cam_id, self.backend)
        time.sleep(0.5)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps_target)

        if not cap.isOpened():
            print(f"[ERROR] Camera {self.cam_id} failed to open")
            self.running = False
            cap.release()
            return

        print(f"[READY] Camera {self.cam_id} ready, waiting for synchronized start...")
        start_event.wait()  # Wait for the shared start signal.
        self.sync_start = time.perf_counter_ns()  # Record the start time for this worker.
        print(f"[INFO] Camera {self.cam_id} capture started!")

        print(f"[INFO] Camera {self.cam_id} ({self.meta['name']}) started @ {self.width}x{self.height} {self.fps_target}fps")
        prev = time.time()
        frame_count = 0
        fps_frame_count = 0

        while self.running:
            ok, frame = cap.read() #  (height, width, 3)
            if not ok:
                self.failure_count += 1

                # Throttle repeated read-failure warnings.
                if self.failure_count % self.failure_warn_interval == 0:
                    print(f"[WARN] Camera {self.cam_id} read failed ({self.failure_count} consecutive failures)")

                # Stop this worker when the failure threshold is reached.
                if self.failure_count >= self.max_consecutive_failures:
                    print(f"[ERROR] Camera {self.cam_id} {self.failure_count} consecutive failures, stopping this camera thread")
                    self.running = False
                    break

                time.sleep(0.05)
                continue

            # Reset the failure count after a successful read.
            self.failure_count = 0
            captured_at = time.monotonic()

            frame_count += 1
            fps_frame_count += 1

            if self.writer:
                self.writer.write(frame)

            # Record the timestamp and frame index.
            # Timestamp relative to this worker start, in nanoseconds.
            t_rel = time.perf_counter_ns() - self.sync_start

            # Discard the oldest frame when the queue is full.
            with self.lock:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame)
                self.frame = frame
                self.last_frame_time = captured_at

            self.timestamps.append(t_rel)
            self.frame_indices.append(frame_count)

            now = time.time()
            if now - prev >= 1.0:
                self.fps = fps_frame_count / (now - prev)
                fps_frame_count = 0
                prev = now

            time.sleep(0.001)

        # Release capture and recording resources.
        try:
            if cap.isOpened():
                cap.release()
        except Exception as e:
            print(f"[WARN] Camera {self.cam_id} error during resource release: {e}")

        try:
            if self.writer is not None:
                self.writer.release()
        except Exception as e:
            print(f"[WARN] Camera {self.cam_id} VideoWriter error during resource release: {e}")

        print(f"[INFO] Camera {self.cam_id} closed")

        if self.save:
            try:
                self.save_timestamps()
            except Exception as e:
                print(f"[WARN] Camera {self.cam_id} failed to save timestamps: {e}")

    def save_timestamps(self):
        """Save frame timestamps as CSV and recording metadata as JSON."""
        base = os.path.splitext(self.outfile)[0]
        csv_path = base + "_timestamps.csv"
        json_path = base + "_meta.json"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for i, ts in zip(self.frame_indices, self.timestamps):
                writer.writerow([i, ts])

        meta_info = {
            "camera_id": self.cam_id,
            "name": self.meta["name"],
            "usb_port": self.meta["usb"],
            "resolution": {"width": self.width, "height": self.height},
            "fps_target": self.fps_target,
            "timestamp_unit": "nanoseconds",
            "frames": len(self.timestamps),
            "timestamp_file": csv_path
        }
        with open(json_path, "w") as f:
            json.dump(meta_info, f, indent=2)
        print(f"[INFO] Timestamps saved to: {csv_path}")

    def get_frame(self):
        with self.lock:
            self._check_frame_health()
            return None if self.frame is None else self.frame.copy()

    def _check_frame_health(self):
        """Reject stopped or stale capture while the frame lock is held."""
        if not self.running or (self.ident is not None and not self.is_alive()):
            raise TimeoutError(f"Fingertip camera {self.cam_id} capture stopped; check the camera connection")
        if self.last_frame_time is not None:
            age = time.monotonic() - self.last_frame_time
            if age >= self.frame_timeout:
                raise TimeoutError(
                    f"Fingertip camera {self.cam_id} has no new frame for {age:.2f}s "
                    f"(limit {self.frame_timeout:.2f}s); check the camera connection"
                )

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the newest frame, raising TimeoutError if capture is unhealthy."""
        with self.lock:
            self._check_frame_health()
            if self.frame_queue.empty():
                return None
            return self.frame_queue.queue[-1].copy()

    def stop(self):
        self.running = False


class DexterousHandPerception:
    """Manage USB cameras, device discovery, and metadata reporting."""

    MAX_CAMERAS = 10
    MAX_QUEUE_SIZE = 5
    def __init__(self,
                 task_name: str = "test_data",
                 root_dir: str = "./data",
                 resolution: tuple = (640, 480),
                 json_path: str ="usbcamera_mapping.json",
                 save: bool=False
                 ):

        self.save = save



        self.fingertip_cameras = {
            "TD_camera": None,
            "ID_camera": None,
            "MD_camera": None,
            "RD_camera": None,
            "LD_camera": None,
        }
        if json_path is None:
            self.cam_metas = detect_available_cameras(self.MAX_CAMERAS)
        else:
            self.cam_metas = load_usb_camera_metas(json_path=json_path)

        if not self.cam_metas:
            raise RuntimeError("❌ No available cameras detected!")
        print("\n=== Detected cameras ===")
        for m in self.cam_metas:
            if resolution:
                m["res"] = resolution
            print(f"  ID:{m['id']}  Model:{m['name']}  Port:{m['usb']}  Resolution:{m['res']}  FPS:{m['fps']}")
        print("=====================\n")

        self.cameras: dict[str, SingleUsbCamera] = {}

    # ---------------------------------------------------
    def init_camera(self):
        """Initialize capture threads for all USB cameras."""
        cameras: Dict[str, SingleUsbCamera] = {}
        for idx, meta in enumerate(self.cam_metas):
            base_name = meta.get("name") or f"Camera{idx}"
            alias = base_name
            if alias in cameras:
                alias = f"{base_name}_{meta['id']}"
            cameras[alias] = SingleUsbCamera(meta, save=self.save, max_queue_size=self.MAX_QUEUE_SIZE)
        self.cameras = cameras
        print(f"[INIT] Initialized {len(self.cameras)} camera threads.")

    def start_all(self):
        for cam in self.cameras.values():
            cam.start()

        time.sleep(1.0)
        print("\n[SYNC] Starting synchronized capture on all cameras!")
        start_event.set()  # Release workers waiting for the start signal.

    def get_latest_frames(self, bgr2rgb: bool = False) -> List[Tuple[int, Optional[np.ndarray]]]:
        """Return frames, camera ID/frame pairs, and validity masks. Frames use (N, W, H, 3)."""
        frames_with_id = []
        frames = []
        masks = np.ones(len(self.cameras), dtype=np.uint8)
        for idx, (alias, cam) in enumerate(self.cameras.items()):
            frame = cam.get_latest_frame()
            if frame is None:
                frame_tp = np.zeros((cam.width, cam.height, 3), dtype=np.uint8)
                masks[idx] = 0  # Mark this camera frame as unavailable.
                print(f"[WARN] {alias}_Camera {cam.cam_id} no frame available")
            else:
                frame_tp = np.transpose(frame, (1, 0, 2))

            if bgr2rgb:
                frame_tp = cv2.cvtColor(frame_tp, cv2.COLOR_BGR2RGB)

            frames_with_id.append((cam.cam_id, frame_tp))
            frames.append(frame_tp)
        frames = np.stack(frames, axis=0)  # (N, W, H, 3)

        return frames, frames_with_id, masks

    def stop_all(self, timeout=5.0):
        """Stop camera workers and wait for them with a timeout."""
        # Request all workers to stop.
        for cam in self.cameras.values():
            cam.stop()

        # Wait for workers with a per-thread timeout.
        for cam in self.cameras.values():
            cam.join(timeout=timeout)
            if cam.is_alive():
                print(f"[WARN] Camera {cam.cam_id} thread did not stop within {timeout} seconds")

        print("[INFO] All cameras stopped.")

    def release(self):
        """Release camera resources by stopping all workers."""
        self.stop_all()
        print("[INFO] All camera resources released.")

    def _get_usb_addresses(self) -> Dict[int, str]:
        """Look up the USB addresses of all active cameras."""
        usb_addresses = {}
        for cam in self.cameras.values():
            usb_addr = get_usb_address_by_camera_id(cam.cam_id)
            if usb_addr:
                usb_addresses[cam.cam_id] = usb_addr
            else:
                usb_addresses[cam.cam_id] = 'N/A'
        return usb_addresses

    def _arrange_windows(self, windows_per_row: int = 3) -> Dict[int, Tuple[int, int]]:
        """Return initial window positions as {camera_id: (x, y)}."""
        window_positions = {}
        for idx, cam in enumerate(self.cameras.values()):
            if cam.width and cam.height:
                row = idx // windows_per_row
                col = idx % windows_per_row
                x = col * (cam.width + 10)  # Horizontal window spacing in pixels.
                y = row * (cam.height + 30)  # Vertical window spacing in pixels.
                window_positions[cam.cam_id] = (x, y)
        return window_positions

    def display(self,
                window_name_prefix: str = "Camera",
                show_usb_address: bool = True,
                arrange_windows: bool = True,
                check_exit_flag: Optional[Callable[[], bool]] = None):
        """Display live fingertip camera frames.

        Exit with Ctrl+C or when check_exit_flag returns True.

        Args:
            window_name_prefix: Prefix for camera window titles.
            show_usb_address: Show USB addresses; enabled by default.
            arrange_windows: Arrange windows automatically; enabled by default.
            check_exit_flag: Optional callback that returns True to exit.
        """

        # Calculate initial window positions.
        window_positions = {}
        positioned_windows = set()  # Position each window only once.
        if arrange_windows:
            window_positions = self._arrange_windows(3)

        # Cache USB addresses when the display starts.
        usb_addresses = {}
        if show_usb_address:
            usb_addresses = self._get_usb_addresses()

        frame_count = 0
        try:
            while True:
                # Check the optional exit callback.
                if check_exit_flag and check_exit_flag():
                    break

                active = False
                for idx, cam in enumerate(self.cameras.values()):
                    frame = cam.get_frame()
                    if frame is None:
                        continue

                    active = True

                    # Build the camera status label.
                    text_line1 = f"{cam.meta['name']} | {cam.width}x{cam.height} | {cam.fps:.1f} fps"
                    cv2.putText(frame, text_line1, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Display the USB address when enabled.
                    if show_usb_address:
                        usb_addr = usb_addresses.get(cam.cam_id, 'N/A')
                        text_line2 = f"USB: {usb_addr}"
                        cv2.putText(frame, text_line2, (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    window_name = f"{window_name_prefix} {cam.cam_id}"
                    cv2.imshow(window_name, frame)

                    # Place new windows once so users can reposition them.
                    if arrange_windows and cam.cam_id in window_positions and cam.cam_id not in positioned_windows:
                        x, y = window_positions[cam.cam_id]
                        cv2.moveWindow(window_name, int(x), int(y))
                        positioned_windows.add(cam.cam_id)

                if not active:
                    # Pause briefly when no frames are available.
                    time.sleep(0.05)

                # Process GUI events; use Ctrl+C to exit.
                cv2.waitKey(1)

                frame_count += 1
                # Periodically yield to reduce CPU usage.
                if frame_count % 30 == 0:
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n[INFO] Interrupt received, closing display.")
        except Exception as e:
            print(f"[ERROR] Error during display: {e}")
        finally:
            try:
                cv2.destroyAllWindows()
            except:
                pass

    def load_bindings(self, json_path="usbcamera_mapping.json"):
        """Read camera USB bindings from a JSON file."""
        with open(json_path, "r") as f:
            bindings = json.load(f)
        print(f"✅ Loaded camera bindings from {json_path}")
        return bindings

    @property
    def camera_count(self):
        return len(self.cameras)


def main():
    system = DexterousHandPerception(save=True)
    system.init_camera()
    system.start_all()

    try:
        system.display()
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received, exiting.")
    finally:
        system.release()

        # Compare frame timestamps after recording.
        csv_files = glob.glob("videos/*_timestamps.csv")
        if len(csv_files) >= 2:
            print("\n=== Automatic frame timestamp alignment analysis ===")
            align_timestamps(csv_files)
            print("=========================\n")


if __name__ == "__main__":
    main()
