#!/usr/bin/env python3
"""
Inspect raw Zarr recordings and preview fingertip images.
Usage:
    python inspect_recording.py <recording_dir> [--list] [--frame INDEX] [--camera INDEX] [--save DIR]

Raw recordings use rgb: (T, 5, W, H, 3) and tv_rgb: (T, W, H, 3).
Image previews show fingertip RGB; other arrays are summarized numerically.
Processed ReplayBuffer ZIP archives are not supported by this tool.
"""

import os
import argparse
import numpy as np
import cv2
import zarr

def inspect_zarr_file(zarr_path, frame_idx=None, camera_idx=None, save_dir=None, list_only=False):
    """
    Inspect and visualize a raw Zarr recording directory
    
    Arguments:
        zarr_path: Path to a raw Zarr recording directory
        frame_idx: Frame index to display (None selects a preview)
        camera_idx: Zero-based fingertip index (None selects multiple cameras)
        save_dir: Image output directory (None disables saving)
    """
    print(f"\n{'='*60}")
    print(f"Inspecting Zarr recording: {zarr_path}")
    print(f"{'='*60}\n")
    
    # Open the Zarr recording.
    if os.path.isdir(zarr_path):
        zarr_data = zarr.open_group(zarr_path, mode='r')
        print(f"Opened Zarr directory.")
    else:
        print(f"Path does not exist or is not a directory: {zarr_path}")
        return
    
    # List available arrays.
    print("\nArray information:")
    print("-" * 60)
    datasets = {}
    for key in zarr_data.keys():
        try:
            arr = zarr_data[key]
            shape = arr.shape
            dtype = arr.dtype
            datasets[key] = {
                'shape': shape,
                'dtype': dtype,
                'array': arr
            }
            print(f"  {key:20s} | Shape: {str(shape):30s} | Dtype: {dtype}")
        except Exception as e:
            print(f"  {key:20s} | Error: {str(e)}")
    
    # Show details for raw fingertip RGB data.
    if 'rgb' in datasets and not list_only:
        rgb_shape = datasets['rgb']['shape']
        print(f"\nFingertip RGB information:")
        print("-" * 60)
        print(f"  Frame count: {rgb_shape[0]}")
        print(f"  Camera count: {rgb_shape[1] if len(rgb_shape) > 1 else 1}")
        print(f"  Image dimensions: {rgb_shape[2:] if len(rgb_shape) > 2 else 'N/A'}")
        
        # Show a selected frame or a preview.
        if frame_idx is not None:
            if frame_idx < 0 or frame_idx >= rgb_shape[0]:
                print(f"\nFrame index {frame_idx} is outside [0, {rgb_shape[0]-1}]")
                return
            
            print(f"\nDisplaying frame {frame_idx}:")
            print("-" * 60)
            
            num_cameras = rgb_shape[1] if len(rgb_shape) > 1 else 1
            cameras_to_show = [camera_idx] if camera_idx is not None else range(num_cameras)
            
            for cam_idx in cameras_to_show:
                if cam_idx < 0 or cam_idx >= num_cameras:
                    print(f"Camera index {cam_idx} is outside [0, {num_cameras-1}]")
                    continue
                
                try:
                    # Raw recordings store images as (W, H, C).
                    img = zarr_data['rgb'][frame_idx, cam_idx].transpose(1, 0, 2)
                    print(f"  Camera {cam_idx}: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
                    
                    # Display the image.
                    if img.dtype != np.uint8:
                        # Scale to 0-255 for display.
                        img_display = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                    else:
                        img_display = img.copy()
                    
                    # Convert RGB to BGR for OpenCV.
                    if len(img_display.shape) == 3 and img_display.shape[2] == 3:
                        img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
                    
                    window_name = f"Frame {frame_idx} - Camera {cam_idx}"
                    cv2.imshow(window_name, img_display)
                    
                    # Save the image.
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"frame_{frame_idx:05d}_camera_{cam_idx}.png")
                        cv2.imwrite(save_path, img_display)
                        print(f"    Saved to: {save_path}")
                    
                except Exception as e:
                    print(f"  Could not read camera {cam_idx}: {e}")
            
            print("\nPress any key to close the windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # Preview up to five frames sampled across the recording.
            print(f"\nQuick preview:")
            print("-" * 60)
            preview_frames = list(range(0, rgb_shape[0], max(1, rgb_shape[0] // 10)))
            
            num_cameras = rgb_shape[1] if len(rgb_shape) > 1 else 1
            cameras_to_show = [camera_idx] if camera_idx is not None else range(min(3, num_cameras))  # Preview at most three cameras.
            
            for cam_idx in cameras_to_show:
                print(f"\n  Camera {cam_idx}:")
                for frame_idx_preview in preview_frames[:5]:  # Preview at most five frames.
                    try:
                        img = zarr_data['rgb'][frame_idx_preview, cam_idx].transpose(1, 0, 2)
                        if img.dtype != np.uint8:
                            img_display = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                        else:
                            img_display = img.copy()
                        
                        if len(img_display.shape) == 3 and img_display.shape[2] == 3:
                            img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
                        
                        # Downscale for display.
                        h, w = img_display.shape[:2]
                        scale = min(200 / w, 200 / h)
                        if scale < 1:
                            img_display = cv2.resize(img_display, (int(w*scale), int(h*scale)))
                        
                        window_name = f"Preview - Frame {frame_idx_preview} - Camera {cam_idx}"
                        cv2.imshow(window_name, img_display)
                        
                    except Exception as e:
                        print(f"    Could not read frame {frame_idx_preview}: {e}")
                
                print(f"    Displayed frames: {preview_frames[:5]}")
            
            print("\nPress any key to close the preview windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Show statistics for other arrays.
    print(f"\nOther array statistics:")
    print("-" * 60)
    for key, info in datasets.items():
        if key == 'rgb':
            continue
        try:
            arr = info['array']
            if arr.size > 0:
                print(f"  {key}:")
                print(f"    Shape: {info['shape']}")
                print(f"    Dtype: {info['dtype']}")
                if arr.size < 100:  # Print all values for small arrays.
                    print(f"    Values: {arr[:]}")
                else:
                    print(f"    Range: [{arr[:].min():.4f}, {arr[:].max():.4f}]")
                    print(f"    Mean: {arr[:].mean():.4f}")
        except Exception as e:
            print(f"  {key}: Read failed - {e}")
    
    print(f"\n{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='Inspect and visualize a raw Zarr recording directory')
    parser.add_argument('zarr_path', type=str, help='Path to a raw Zarr recording directory')
    parser.add_argument('--frame', type=int, default=None, help='Frame index (default: preview)')
    parser.add_argument('--camera', type=int, default=None, help='Zero-based fingertip index (default: multiple cameras)')
    parser.add_argument('--save', type=str, default=None, help='Image output directory for --frame (default: disabled)')
    parser.add_argument('--list', action='store_true', help='List information without opening image windows')
    
    args = parser.parse_args()
    
    frame_idx = args.frame
    
    inspect_zarr_file(
        args.zarr_path,
        frame_idx=frame_idx,
        camera_idx=args.camera,
        save_dir=args.save,
        list_only=args.list,
    )

if __name__ == "__main__":
    main()
