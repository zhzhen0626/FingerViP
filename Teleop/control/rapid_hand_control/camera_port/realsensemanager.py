import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading
from queue import Queue
from threading import Lock
from typing import Optional, Tuple
import open3d as o3d

# Global lock for OpenCV window operations to avoid conflicts in multi-threaded environment
_opencv_window_lock = threading.Lock()

class RealSenseCamera:
    """Intel RealSense D435i camera interface with threaded-safe frame handling."""

    MAX_QUEUE_SIZE = 5

    def __init__(self, serial_number=None):
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_queue = Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.color_queue = Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.lock = Lock()
        self.running = False
        self.device = None
        self.serial_number = serial_number
        self.intrinsics = {}
        print("[INFO] RealSenseCamera instance created.")

    # --------------------------------------------------------
    def init_camera(self, width: int = 640, height: int = 480, fps: int = 30):
        """Initialize D435i and configure RGB + Depth streams."""
        print("[INFO] Initializing RealSense D435i camera...")
        ctx = rs.context()
        if len(ctx.devices) == 0:
            raise RuntimeError("No RealSense device detected.")
        
        camera_sn = self.serial_number
        # select target camera
        def find_device_by_sn(ctx, serial):
            for dev in ctx.devices:
                if dev.get_info(rs.camera_info.serial_number) == serial:
                    return dev
            return None
        
        # Try to find device without reset first
        target_dev = None
        if camera_sn is not None:
            target_dev = find_device_by_sn(ctx, camera_sn)
        
        # Only reset if device not found or if explicitly needed
        # Hardware reset can interfere with other cameras, so we try without reset first
        need_reset = False
        if camera_sn is not None:
            if target_dev is None:
                print(f"[WARN] Device {camera_sn} not found. Will try hardware reset.")
                need_reset = True
            else:
                # Device found, try without reset first
                print(f"[INFO] Device {camera_sn} found. Attempting initialization without reset...")
        
        if need_reset:
            # reset camera only if device not found
            target = find_device_by_sn(ctx, camera_sn)
            if target is None:
                raise RuntimeError(f"Device {camera_sn} not found and cannot reset.")
            print(f"[INFO] Resetting device {camera_sn}...")
            target.hardware_reset()

            # ---------- Wait for re-enumeration ----------
            print("[INFO] Waiting for device to reconnect after reset...")
            time.sleep(4.0)  # Allow time for the camera to recover.

            target_dev = None
            for i in range(12):
                ctx = rs.context()
                if camera_sn is not None:
                    target_dev = find_device_by_sn(ctx, camera_sn)
                    if target_dev:
                        print(f"[INFO] Device {camera_sn} reconnected after reset.")
                        break
                time.sleep(0.5)
            else:
                raise RuntimeError("Device did not reconnect after reset.")
        else:
            # Use the device we found without reset
            if target_dev is None:
                raise RuntimeError(f"Device {camera_sn} not found.")
            print(f"[INFO] Using device {camera_sn} without reset (to avoid interfering with other cameras).")

        # ---------- Setup pipeline ----------
        self.device = target_dev # ctx.devices[0]
        name = self.device.get_info(rs.camera_info.name)
        serial = self.device.get_info(rs.camera_info.serial_number)
        # Update self.serial_number to match the actual device serial
        self.serial_number = serial
        print(f"[INFO] Found device: {name} (Serial: {serial})")
        if camera_sn and camera_sn != serial:
            print(f"[WARN] Serial number mismatch! Expected: {camera_sn}, Got: {serial}")

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Explicitly specify device by serial number to avoid conflicts
        if serial:
            self.config.enable_device(serial)

        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)

        # Start pipeline with retry mechanism for device busy errors
        max_retries = 5  # Increased retries for multi-camera scenarios
        retry_delay = 1.0  # Start with shorter delay
        for attempt in range(max_retries):
            try:
                self.pipeline.start(self.config)
                print(f"[INFO] Pipeline started successfully for device {serial} (attempt {attempt + 1})")
                break
            except RuntimeError as e:
                error_msg = str(e).lower()
                if ("busy" in error_msg or "resource" in error_msg or "in use" in error_msg) and attempt < max_retries - 1:
                    print(f"[WARN] Device {serial} busy or in use, retrying in {retry_delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    print(f"[ERROR] Failed to start pipeline for device {serial}: {e}")
                    raise RuntimeError(f"Failed to start RealSense pipeline after {max_retries} attempts: {e}")
        
        self.running = True
        time.sleep(2.0)
        print("[INFO] RealSense D435i streaming started.")

        # Get intrinsics
        profile = self.pipeline.get_active_profile()
        
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"[INFO] Depth scale: {self.depth_scale:.6f} meters per unit")

        color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth_intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        # Extract intrinsics attributes manually (pyrealsense2 intrinsics objects don't have __dict__)
        def extract_intrinsics(intr):
            return {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
                "model": intr.model,
                "coeffs": list(intr.coeffs) if hasattr(intr, 'coeffs') else [],
            }
        
        self.intrinsics = {
            "color": extract_intrinsics(color_intr),
            "depth": extract_intrinsics(depth_intr),
        }
        print("[INFO] Camera intrinsics loaded.")

    # --------------------------------------------------------
    def _capture_frames(self, timeout_ms=10000):
        """Capture latest aligned frames and push them into queues."""
        if not self.running:
            return False

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                return False

            with self.lock:
                depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale # in meters
                color_image = np.asanyarray(color_frame.get_data())

                # remove the oldest data
                if self.depth_queue.full():
                    self.depth_queue.get()
                if self.color_queue.full():
                    self.color_queue.get()

                # put the newest data
                self.depth_queue.put(depth_image)
                self.color_queue.put(color_image)
            return True
        except RuntimeError as e:
            # Handle device disconnection or timeout gracefully
            if "didn't arrive" in str(e) or "disconnected" in str(e).lower() or "busy" in str(e).lower():
                return False
            raise
        except Exception as e:
            print(f"[WARN] Frame capture error: {e}")
            return False

    # --------------------------------------------------------
    def get_latest_frames(self, *args, timeout: float = 1.0, **kwargs) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return latest (color, depth_colormap) images, or (None, None) if timeout."""
        if not self.running:
            return None, None
        if self.pipeline is None:
            return None, None
        
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            if self._capture_frames(timeout_ms=remaining_ms):
                with self.lock:
                    if not self.color_queue.empty() and not self.depth_queue.empty():
                        depth_image = self.depth_queue.queue[-1] # (480, 640)
                        color_image = self.color_queue.queue[-1] # (480, 640, 3)

                        depth_image = np.transpose(depth_image, (1, 0)).copy()
                        color_image = np.transpose(color_image, (1, 0, 2)).copy()

                        return color_image, depth_image
                    else:
                        # Queue is empty even though _capture_frames returned True
                        # This might indicate a race condition
                        pass
            time.sleep(0.02)
        return None, None

    # --------------------------------------------------------
    def visualize_stream(self, vis_mode: str = "auto"):
        """Open a live window showing both RGB + Depth streams."""
        print("[INFO] Starting visualization. Press ESC to exit.")
        while self.running:
            color_image, depth_image = self.get_latest_frames(timeout=2.0)
            if color_image is None or depth_image is None:
                continue

            # coloremap for visualization
            if vis_mode == "auto":
                depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            elif vis_mode == "fixed":
                min_range = 0.2  # meters
                max_range = 4.0  # meters
                depth_clipped = np.clip(depth_image, min_range, max_range)
                depth_image = ((depth_clipped - min_range) / (max_range - min_range) * 255.0).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(
                    depth_image,
                    cv2.COLORMAP_JET,
                )  # (640, 480, 3)
            else:   
                raise ValueError("vis_mode must be 'auto' or 'fixed'.")
                    
            combined = np.hstack((color_image, depth_colormap))
            cv2.imshow("RealSense D435i (Color + Depth)", combined)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
        self.stop_camera()
        cv2.destroyAllWindows()

    # --------------------------------------------------------
    def display(self, vis_mode: str = "fixed", window_name: str = None, window_created_in_main_thread: bool = False):
        """Non-blocking display method for camera streams.
        
        This method runs in a loop displaying RGB + Depth streams without stopping
        the camera when exiting. Designed to be called from a separate thread.
        
        Args:
            vis_mode: "auto" or "fixed" depth visualization mode
            window_name: Custom window name (defaults to "RealSense {serial_number}")
            window_created_in_main_thread: If True, assumes window was already created in main thread
        """
        serial = self.serial_number or "Unknown"
        if window_name is None:
            window_name = f"RealSense {serial}"
        
        print(f"[INFO] RealSense display started: {window_name}. Press 'q' to exit.")
        
        # Check if camera is properly initialized
        if not self.running:
            print(f"[ERROR] {window_name}: Camera is not running. Cannot start display.")
            return
        
        if self.pipeline is None:
            print(f"[ERROR] {window_name}: Pipeline is None. Camera may not be initialized.")
            return
        
        # Set OpenCV to use single thread for GUI operations to avoid conflicts
        # This is important when multiple cameras are running in separate threads
        cv2.setNumThreads(1)
        
        # Create named window - only if not created in main thread
        if not window_created_in_main_thread:
            lock_acquired = False
            try:
                # Acquire lock before creating window to ensure thread safety
                try:
                    lock_acquired = _opencv_window_lock.acquire(timeout=5.0)
                    if not lock_acquired:
                        print(f"[ERROR] Failed to acquire window creation lock for '{window_name}' within 5 seconds")
                        return
                except AttributeError:
                    # Python < 3.2 doesn't support timeout, use blocking acquire
                    _opencv_window_lock.acquire()
                    lock_acquired = True
                
                try:
                    time.sleep(0.2)  # Small delay to ensure previous window operations complete
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.pollKey()
                finally:
                    if lock_acquired:
                        _opencv_window_lock.release()
                        
            except Exception as e:
                print(f"[ERROR] Failed to create window '{window_name}': {e}")
                if lock_acquired:
                    try:
                        _opencv_window_lock.release()
                    except:
                        pass
                return
        
        # Warm-up: wait for a few frames to ensure camera is ready
        warmup_frames = 5
        warmup_success = 0
        for i in range(warmup_frames):
            try:
                color_image, _ = self.get_latest_frames(timeout=2.0)
                if color_image is not None:
                    warmup_success += 1
            except Exception:
                pass
            time.sleep(0.1)
        
        if warmup_success == 0:
            print(f"[WARN] {window_name}: No frames received during warm-up, but continuing anyway...")
        
        frame_count = 0
        first_frame = True
        consecutive_failures = 0
        max_failures = 50
        last_update_time = time.time()
        
        try:
            while self.running:
                time.sleep(0.01)  # Small delay to prevent tight loop
                
                color_image, _ = self.get_latest_frames(timeout=0.1)
                if color_image is None:
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        print(f"[ERROR] {window_name}: Too many consecutive frame failures. Camera may be disconnected.")
                        break
                    continue
                
                consecutive_failures = 0  # Reset failure counter on success

                # Ensure image has correct shape and type
                if len(color_image.shape) != 3 or color_image.shape[2] != 3:
                    continue
                
                if color_image.size == 0:
                    continue
                
                # Ensure color image is uint8 (0-255 range)
                if color_image.dtype != np.uint8:
                    if color_image.dtype == np.float32 or color_image.dtype == np.float64:
                        if color_image.max() <= 1.0:
                            color_image = (color_image * 255).astype(np.uint8)
                        else:
                            color_image = np.clip(color_image, 0, 255).astype(np.uint8)
                    else:
                        color_image = color_image.astype(np.uint8)
                
                if np.all(color_image == 0):
                    continue
                
                # Set window size on first frame
                if first_frame:
                    img_height, img_width = color_image.shape[:2]
                    cv2.resizeWindow(window_name, img_width, img_height)
                    first_frame = False
                
                # Add camera info overlay with serial number
                serial_display = self.serial_number or "Unknown"
                info_text = f"{window_name} | Serial: {serial_display} | {color_image.shape[1]}x{color_image.shape[0]}"
                cv2.putText(color_image, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display RGB image only (no depth)
                try:
                    with _opencv_window_lock:
                        cv2.imshow(window_name, color_image)
                    
                    # Check for 'q' key press to exit
                    try:
                        key = cv2.pollKey()
                        if key != -1 and (key & 0xFF) == ord('q'):
                            print(f"[INFO] 'q' key pressed, exiting display: {window_name}")
                            break
                    except Exception:
                        pass  # Ignore pollKey errors
                    
                    # Limit update rate to ~30 FPS
                    current_time = time.time()
                    if current_time - last_update_time < 0.033:
                        time.sleep(0.033 - (current_time - last_update_time))
                    last_update_time = time.time()
                    
                    frame_count += 1
                except Exception as e:
                    print(f"[ERROR] {window_name}: Failed to display image: {e}")
                    continue
        
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"[ERROR] Display error: {e}")
        finally:
            try:
                cv2.destroyWindow(window_name)
            except:
                pass

    # --------------------------------------------------------
    def visualize_pointcloud(self, voxel_size: float = 0.001):
        """Real-time color point cloud visualization using Open3D."""
        print("[INFO] Starting real-time point cloud visualization...")

        pc = rs.pointcloud()
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="RealSense D435i PointCloud", width=1280, height=720)
        geom_added = False
        pcd = o3d.geometry.PointCloud()

        try:
            while self.running:
                frames = self.pipeline.wait_for_frames(timeout_ms=10000)
                aligned = self.align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # generate point cloud
                pc.map_to(color_frame)
                points = pc.calculate(depth_frame)
                vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3) # in meters

                # === adjust ===
                if np.mean(np.abs(vtx)) < 0.01:
                    vtx *= 1000.0  # RealSense point clouds use meters; scale by 1000 if needed for visualization.
                tex = np.asanyarray(points.get_texture_coordinates())

                # Ensure texture coordinates form an (N, 2) float32 array.
                if tex.dtype.fields is not None:  # structured array like [('f0','<f4'),('f1','<f4')]
                    tex = np.vstack([tex['f0'], tex['f1']]).T
                elif tex.ndim == 1:
                    tex = tex.reshape(-1, 2)

                color_image = np.asanyarray(color_frame.get_data())
                h, w, _ = color_image.shape

                # Map normalized RealSense texture coordinates to pixel coordinates.
                us = np.clip((tex[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
                vs = np.clip((tex[:, 1] * (h - 1)).astype(np.int32), 0, h - 1)
                colors = color_image[vs, us, ::-1] / 255.0  # BGR → RGB

                pcd.points = o3d.utility.Vector3dVector(vtx)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                if not geom_added:
                    vis.add_geometry(pcd)
                    geom_added = True
                else:
                    vis.update_geometry(pcd)
                    print("[INFO] Updating point cloud...")

                vis.poll_events()
                vis.update_renderer()

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    fname = time.strftime("d435i_%Y%m%d_%H%M%S.ply")
                    o3d.io.write_point_cloud(fname, pcd)
                    print(f"[✅] Saved point cloud: {fname}")
        finally:
            vis.destroy_window()
            print("[INFO] Point cloud visualization closed.")

    # --------------------------------------------------------
    def stop_camera(self):
        """Stop camera pipeline safely."""
        if self.pipeline and self.running:
            print("[INFO] Stopping RealSense D435i...")
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"[WARN] Error stopping pipeline: {e}")
            self.running = False
            time.sleep(1.5)
            print("[INFO] Camera stopped.")

    # --------------------------------------------------------
    def release(self):
        """Release resources and reset the hardware."""
        print("[INFO] Releasing RealSense D435i resources...")
        self.stop_camera()

        # release hardware reset
        ctx = rs.context()
        for dev in ctx.devices:
            try:
                dev.hardware_reset()
            except Exception:
                pass

        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_queue.queue.clear()
        self.color_queue.queue.clear()
        self.intrinsics.clear()
        print("[INFO] Resources released and hardware reset complete.")
