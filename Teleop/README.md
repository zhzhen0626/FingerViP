# FingerViP Teleoperation

Vision Pro teleoperation and data collection with a UR5e arm, RAPID Hand, five fingertip cameras and one third-view RealSense camera.

<a id="environment"></a>

## 📦 ROS and system setup

After the [shared environment installation](../README.md#environment), run from the repository root:

```bash
conda activate fingervip
cd Teleop
sudo apt install v4l-utils libgl1 libglib2.0-0
```

**ROS 1 prerequisite:** provide `rospy`, `sensor_msgs`, `std_msgs` and a `cv_bridge` built for Python 3.10. Standard Noetic binaries target Python 3.8.

```bash
source /path/to/ros1_workspace/devel/setup.bash
```

Run the remaining commands from `Teleop/`, with the Python and ROS environments loaded in every terminal.

<a id="configuration"></a>

## ⚙️ Configuration

Fill in your device settings before starting:

| File | Settings |
| --- | --- |
| [args/teleop_args.yaml](args/teleop_args.yaml) | `robot_ip`, `user`, saving and display settings |
| [args/inputstream_args/visionpro.yaml](args/inputstream_args/visionpro.yaml) | `avp_ip`: the Vision Pro address shown by Tracking Streamer |
| [args/robot_args/hand_args/rapid_args.yaml](args/robot_args/hand_args/rapid_args.yaml) | `motor_port`, `use_third_view_camera`, `third_view_serial_number` |
| [args/robot_args/arm_args/ur5e_rapid.yaml](args/robot_args/arm_args/ur5e_rapid.yaml) | Arm initial pose: `init_qpos` |
| [args/eval_args.yaml](args/eval_args.yaml) | `checkpoint_path`, `num_steps` and evaluation settings |

### Motor calibration

Use `ls -l /dev/serial/by-id/` to find `<MOTOR_PORT>`. Set Protocol 2, motor IDs **0–19** and **3,000,000 bps**. Serial/camera access may require `dialout`/`video` group membership.

Close Dynamixel Wizard, mount the calibration fixture and place the fingers in the zero pose:

```bash
python -m utils.motor_init -p <MOTOR_PORT>
```

This updates the Teleop [rapid_full.json](control/rapid_hand_control/servo_driver/rapid_full.json); the Hardware directory has a separate calibration file. Remove the fixture before testing; this commands the zero pose for 100 seconds:

```bash
python -m utils.motor_init -p <MOTOR_PORT> --test
```

### Fingertip cameras

Find all connected cameras:

```bash
v4l2-ctl --list-devices
```

Find the cameras’ USB addresses with visualizations:

```bash
python control/rapid_hand_control/fingertip_camera/view_fingertip_cameras.py
```

Edit [usbcamera_mapping.json](control/rapid_hand_control/fingertip_camera/usbcamera_mapping.json) with USB addresses, rather than `/dev/videoN`, for thumb (`TD_camera`), index (`ID_camera`), middle (`MD_camera`), ring (`RD_camera`) and little (`LD_camera`).

```bash
python control/rapid_hand_control/fingertip_camera/view_fingertip_cameras.py \
  --mapping usbcamera_mapping.json --resolution 640 480
```

Verify each finger view, then press `q` to close the preview.

### Vision Pro calibration

Run [Tracking Streamer](https://github.com/Improbable-AI/VisionProTeleop) 2.5.1 on the same LAN and set `avp_ip` (client: `avp_stream==2.51`).

Use `user: example_user` for the [included calibration](data/user_data/visionpro/example_user.txt).

<details>
<summary>Optional: record personal calibration</summary>

Set `user: my_user` in `args/teleop_args.yaml`, hold the calibration pose and run:

```bash
python - <<'PYTHON'
from utils.load_args import _load_yaml, ARGS_DIR
from teleop.hand_pose_stream.visionpro_input import VisionProInput

settings = _load_yaml(ARGS_DIR / "teleop_args.yaml")
vision = _load_yaml(ARGS_DIR / "inputstream_args" / "visionpro.yaml")
stream = VisionProInput(avp_ip=vision["avp_ip"])
try:
    stream.store_data(user_name=settings["user"], duration=5, frame_rate=10)
    if len(stream.read_data(settings["user"])) < 6:
        raise RuntimeError("At least six valid calibration records are required; repeat the recording.")
finally:
    stream.stop()
PYTHON
```

This overwrites the selected user's calibration. Set `user: example_user` to use the included calibration.

</details>

## 🎮 Teleoperation and collection

1. Finish calibration, remove the fixture and start Tracking Streamer. **The robot node moves the arm and hand on startup.**
2. Move the arm to its initial pose before starting the robot node:

   ```bash
   python control/ur5e_control/scripts/move_to_init_position.py --robot_ip <ROBOT_IP>
   ```

3. Run each command in a separate terminal with the environments loaded:

   ```bash
   roscore
   ```

   ```bash
   python teleop_node.py
   ```

   ```bash
   python robot_node.py
   ```

4. In the robot terminal, enter `t` and wait for `Teleoperation active`, then enter `s` to record. Enter `q` to stop; answer `y` to save or `n` to discard. **Press Enter after each command.**

Recordings are saved under `data/test_data/<session>/<timestamp>.zarr`. Change `save_config` in `args/teleop_args.yaml` to set the output path. Reset returns the arm to its pose at node startup.

Wait for saving to finish before starting another recording or exiting. If saving fails, fix the storage issue and enter `q` to retry or discard.

If camera capture fails, follow the prompt to save or discard earlier samples, then check the cameras before restarting.

## 🚀 Policy evaluation

Follow the [deployment instructions](../README.md#deployment) for checkpoint inference and real-robot evaluation.

## 📄 License

Based on [RAPID Hand](https://github.com/SYSU-RoboticsLab/RAPID-Hand), under [CC BY-NC-SA 4.0](LICENSE). Retain upstream attribution and third-party license notices.
