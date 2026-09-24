import cv2
import subprocess
import re
import time
import matplotlib.pyplot as plt  # Plot frame timestamp differences.
import numpy as np
import json
from pathlib import Path
from typing import Optional

def load_usb_camera_metas(json_path="usbcamera_mapping.json", max_retries=3, retry_delay=0.5):
    """Load USB camera bindings and retry device discovery."""
    json_path = Path(json_path)
    if not json_path.is_absolute():
        json_path = Path(__file__).resolve().parent / json_path

    with json_path.open("r", encoding="utf-8") as f:
        mappings = json.load(f)
    print(f"✅ Loaded camera bindings from {json_path}")

    # Allow USB devices time to initialize.
    time.sleep(0.5)

    cam_metas = []
    for alias, usb_port in mappings.items():
        if not usb_port:
            continue

        cam_meta = None
        for attempt in range(max_retries):
            cam_meta = get_video_path_by_usb(usb_port)
            if cam_meta:
                break
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        if cam_meta is None:
            print(f"[WARN] No device found for {alias} ({usb_port})")
            continue

        cam_meta['name'] = alias
        cam_meta['usb'] = usb_port
        cam_metas.append(cam_meta)

    return cam_metas

def get_video_path_by_usb(target_usb, max_open_retries=3, open_retry_delay=0.3):
    """Find a video node by USB port, retrying device access when needed."""
    try:
        # Keep listing output even when some devices cannot be opened.
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            check=False  # Continue when individual devices are unavailable.
        )
        output = result.stdout + result.stderr
    except FileNotFoundError:
        raise RuntimeError("Install first: v4l-utils :  sudo apt install v4l-utils")

    blocks = [b.strip() for b in output.strip().split("\n\n") if b.strip()]
    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 2:
            continue
        header = lines[0]
        devs = [l.strip() for l in lines[1:] if "/dev/video" in l]
        m = re.match(r"(.+?): (.+?)\s*\((usb-[^)]+)\)", header)
        if not m:
            continue
        name, vendor, usb_port = m.groups()
        if target_usb in usb_port:
            camera_meta = None
            if len(devs) > 0:
                path = devs[0]
                try:
                    cam_id = int(path.replace("/dev/video", ""))
                except ValueError:
                    continue

                # Retry opening devices that are still initializing.
                for open_attempt in range(max_open_retries):
                    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            camera_meta = {
                                "id": cam_id,
                                "type": name.strip(),
                                "vendor": vendor.strip(),
                                "path": path,
                                "usb": usb_port.strip(),
                                "res": (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                "fps": int(cap.get(cv2.CAP_PROP_FPS)) or 30
                            }
                            cap.release()
                            return camera_meta
                    cap.release()
                    if open_attempt < max_open_retries - 1:
                        time.sleep(open_retry_delay)
            return camera_meta  # The matching device could not be opened.
    return None

def detect_available_cameras(max_id=10):
    """Discover available camera IDs and metadata, including video nodes and USB addresses."""
    cameras = []
    # List cameras even if V4L2 reports an unavailable device.
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            check=False
        )
        output = result.stdout + result.stderr
    except Exception:
        output = ""

    blocks = [b.strip() for b in output.strip().split("\n\n") if b.strip()]
    for b in blocks:
        lines = b.split("\n")
        if len(lines) < 2:
            continue
        header = lines[0]
        dev_paths = [l.strip() for l in lines[1:] if "/dev/video" in l]
        name_match = re.match(r"(.+?): (.+?) \(?usb-(.*?)\)?", header)
        if not name_match:
            continue
        model, vendor, usb_path = name_match.groups()

        # Include only DICOTA 4K cameras.
        if "DICOTA 4K" not in model:
            continue

        if len(dev_paths) > 0:
            path = dev_paths[0]
            try:
                cam_id = int(path.replace("/dev/video", ""))
            except ValueError:
                continue
            # Verify that the device returns a frame.
            cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cameras.append({
                        "id": cam_id,
                        "name": model.strip(),
                        "path": path,
                        "usb": f"usb-{usb_path.strip()}",  # Retain the USB address prefix.
                        "res": (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                        "fps": int(cap.get(cv2.CAP_PROP_FPS)) or 30
                    })
            cap.release()
    return cameras


def get_usb_address_by_camera_id(cam_id: int) -> Optional[str]:
    """Look up the current USB address for a camera ID."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            check=False
        )
        output = result.stdout + result.stderr
    except FileNotFoundError:
        return None

    blocks = [b.strip() for b in output.strip().split("\n\n") if b.strip()]
    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 2:
            continue
        header = lines[0]
        devs = [l.strip() for l in lines[1:] if "/dev/video" in l]

        # Find the device group containing this video node.
        target_path = f"/dev/video{cam_id}"
        if target_path in devs:
            # Extract the USB address from the device header.
            m = re.search(r"\(usb-([^)]+)\)", header)
            if m:
                return f"usb-{m.group(1)}"
    return None


def align_timestamps(csv_files):
    """
    Compare timestamp CSVs using nearest-frame matches and plot timing differences.
    """
    ts_arrays = []
    for path in csv_files:
        data = np.loadtxt(path, delimiter=",")
        ts_arrays.append(data[:, 1])  # The second column contains timestamps.

    base = ts_arrays[0]
    aligned = []
    for i, other in enumerate(ts_arrays[1:], start=1):
        diffs = []
        for t in base:
            idx = np.argmin(np.abs(other - t))
            diffs.append(abs(other[idx] - t) / 1e6)  # Convert nanoseconds to milliseconds.
        aligned.append(np.array(diffs))
        print(f"Camera 0 vs camera {i}: mean difference {np.mean(diffs):.3f} ms, maximum difference {np.max(diffs):.3f} ms")
        # Plot frame timestamp differences.
        plt.plot(diffs, label=f"Cam0 vs Cam{i}")

    plt.xlabel("Frame index")
    plt.ylabel("Δt (ms)")
    plt.title("Frame Time Difference Across Cameras")
    plt.legend()
    plt.tight_layout()
    plt.savefig("videos/time_diff_plot.png")
    plt.show()
    print("[INFO] Time-difference distribution plot saved to: videos/time_diff_plot.png")
    return aligned
