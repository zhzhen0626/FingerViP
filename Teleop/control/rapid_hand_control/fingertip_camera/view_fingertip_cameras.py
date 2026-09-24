#!/usr/bin/env python3
"""Preview live fingertip camera frames.

Run with the camera environment activated:

    python view_fingertip_cameras.py

Options:
    --save             Record video and timestamps; disabled by default.
    --resolution W H   Set camera resolution, for example 640 480.
    --mapping PATH     Load USB bindings from JSON; otherwise discover cameras.
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2

# Allow direct execution to import sibling modules.
if __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

from dicota4kusbcamera import DexterousHandPerception

try:
    from .utils import get_usb_address_by_camera_id
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from utils import get_usb_address_by_camera_id


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display live views from five fingertip cameras (press 'q' to quit)."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save videos and timestamps to the videos/ directory.",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        type=int,
        default=None,
        help="Set the resolution for all cameras, e.g. --resolution 640 480.",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="Path to the USB camera mapping file; detect available cameras automatically if omitted.",
    )
    return parser.parse_args(argv)


def run(save: bool, resolution: Optional[Tuple[int, int]], mapping_path: Optional[str]) -> None:
    system = DexterousHandPerception(
        save=save,
        json_path=mapping_path,
        resolution=resolution,
    )
    system.init_camera()
    system.start_all()

    # Cache the USB addresses for the active cameras.
    usb_addresses = {}
    last_usb_update = 0
    usb_update_interval = 5.0  # Refresh USB addresses every five seconds.

    def update_usb_addresses():
        """Refresh USB addresses for all active cameras."""
        nonlocal last_usb_update
        current_time = time.time()
        if current_time - last_usb_update < usb_update_interval:
            return  # Wait until the next refresh interval.

        last_usb_update = current_time
        print("\n=== Detecting camera USB addresses ===")
        for alias, cam in system.cameras.items():
            usb_addr = get_usb_address_by_camera_id(cam.cam_id)
            if usb_addr:
                usb_addresses[cam.cam_id] = usb_addr
                print(f"  {alias} (ID: {cam.cam_id}): {usb_addr}")
            else:
                usb_addresses[cam.cam_id] = 'N/A'
                print(f"  {alias} (ID: {cam.cam_id}): N/A (not detected)")
        print("============================\n")

    # Populate the cache before displaying frames.
    update_usb_addresses()

    def save_current_frames():
        """Save the current camera frames in a new snapshot directory."""
        # Resolve the directory containing this script.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Create the snapshot output directory.
        base_dir = os.path.join(script_dir, "saved_images")
        os.makedirs(base_dir, exist_ok=True)

        # Create a timestamped snapshot directory.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(base_dir, f"saved_images_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        saved_count = 0
        for alias, cam in system.cameras.items():
            frame = cam.get_frame()  # Get the original frame without display annotations.
            if frame is not None:
                # Name snapshots using the camera alias and ID.
                filename = f"{alias.replace(' ', '_')}_cam{cam.cam_id}.jpg"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, frame)
                saved_count += 1

        print(f"[SAVED] Saved {saved_count} images to {save_dir}/")
        return save_dir

    try:
        print("Press 'q' to quit or 's' to save images from all cameras.")
        frame_count = 0
        # Arrange camera windows in a grid.
        window_positions = {}
        # Maximum number of windows per row.
        windows_per_row = 3
        while True:
            active = False
            # Refresh the USB address cache periodically.
            update_usb_addresses()

            for idx, cam in enumerate(system.cameras.values()):
                frame = cam.get_frame()
                if frame is None:
                    continue
                active = True

                # Calculate this window position on its first frame.
                if cam.cam_id not in window_positions:
                    h, w = frame.shape[:2]
                    row = idx // windows_per_row
                    col = idx % windows_per_row
                    x = col * w
                    y = row * h
                    window_positions[cam.cam_id] = (x, y)

                # Show the camera name, resolution, and frame rate.
                text_line1 = f"{cam.meta['name']} | {cam.width}x{cam.height} | {cam.fps:.1f} fps"

                # Show the cached USB address on the second line.
                usb_addr = usb_addresses.get(cam.cam_id, 'N/A')
                text_line2 = f"USB: {usb_addr}"

                cv2.putText(frame, text_line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, text_line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                window_name = f"Camera {cam.cam_id}"
                cv2.imshow(window_name, frame)

                # Move each window to its assigned grid position.
                x, y = window_positions[cam.cam_id]
                cv2.moveWindow(window_name, int(x), int(y))

            if not active:
                # Pause briefly when no frames are available.
                time.sleep(0.05)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save_current_frames()

            frame_count += 1
            # Periodically yield to reduce CPU usage.
            if frame_count % 30 == 0:
                time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        system.release()


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    resolution = tuple(args.resolution) if args.resolution else None
    run(args.save, resolution, args.mapping)


if __name__ == "__main__":
    main()

