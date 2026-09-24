# FingerViP Hardware

Vision-enhanced fingertip designs and motor setup utilities for FingerViP, based on [RAPID Hand](https://github.com/SYSU-RoboticsLab/RAPID-Hand).

## 🛠️ Mechanical assembly

Follow the [mechanical guide](mechanical_structure/README.md) for fingertip models, BOM and assembly; use RAPID Hand designs for the remaining structure.

## ⚙️ Motor setup

After the [shared environment installation](../README.md#environment), run from the repository root:

```bash
conda activate fingervip
cd Hardware
```

Close Dynamixel Wizard, mount the calibration fixture and align the fingers to the intended zero pose. Record the motor positions with:

```bash
python motor_init.py -p <MOTOR_PORT>
```

This updates [rapid_full.json](rapid_hand_control/servo_driver/rapid_full.json). After removing the fixture, check the zero pose with:

```bash
python motor_init.py -p <MOTOR_PORT> --test
```

The test commands the zero pose for ten seconds. To run predefined gestures:

```bash
python motion_sequence_controller.py -p <MOTOR_PORT>
```

**Teleop uses a separate calibration file.** For the integrated system, follow [Teleop motor calibration](../Teleop/README.md#motor-calibration).

## 📷 Cameras and teleoperation

Follow the [Teleop guide](../Teleop/README.md) to configure the five fingertip cameras, third-view RealSense D435i and Vision Pro, then collect demonstrations.

## 📖 Citation

See the [FingerViP citation](../README.md#citation) and [RAPID Hand paper](https://arxiv.org/abs/2506.07490).

## 📄 License

See [LICENSE](LICENSE) for CC BY-NC-SA 4.0. Original RAPID Hand attribution and third-party notices are retained.
