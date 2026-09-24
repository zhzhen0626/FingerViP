from Teleop.control.rapid_hand_control.camera_port.realsensemanager import RealSenseCamera

if __name__ == "__main__":
    cam = RealSenseCamera()
    cam.init_camera()
    mode = input("Enter 'd' for RGB+Depth, 'p' for PointCloud: ").strip().lower()
    if mode == "p":
        cam.visualize_pointcloud(voxel_size=0.01)
    else:
        cam.visualize_stream(vis_mode="fixed")
    cam.release()
