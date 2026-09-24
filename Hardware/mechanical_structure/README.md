# FingerViP Mechanical Assembly

Camera-equipped fingertip modules for [RAPID Hand](https://github.com/SYSU-RoboticsLab/RAPID-Hand), using SONY IMX258 cameras and DYNAMIXEL XC330-M288-T motors.

## 📁 Models and BOM

The [BOM and assembly guide](BOM_assembly_guide.pdf) provides quantities, screw sizes and labeled assembly views for **four finger modules and one thumb module**.

| Module | Printable shells | STEP assembly |
| --- | --- | --- |
| Finger ×4 | [F1/F2](3D%20Printed%20Parts/Finger/) | [Finger.STEP](Model/Finger/Finger.STEP) |
| Thumb ×1 | [T1/T2](3D%20Printed%20Parts/Thumb/) | [Thumb.STEP](Model/Thumb/Thumb.STEP) |

### Directory Structure

```text
mechanical_structure/
├── README.md
├── BOM_assembly_guide.pdf
├── 3D Printed Parts/
│   ├── Finger/
│   │   ├── F1.STL
│   │   └── F2.STL
│   └── Thumb/
│       ├── T1.STL
│       └── T2.STL
└── Model/
    ├── Finger/
    │   └── Finger.STEP
    └── Thumb/
        └── Thumb.STEP
```

## 🔗 RAPID Hand Components

Use the upstream [mechanical designs](https://github.com/SYSU-RoboticsLab/RAPID-Hand/tree/main/RapidHandHardware/mechanical_structure) and [BOM/assembly guide](https://github.com/SYSU-RoboticsLab/RAPID-Hand/blob/main/RapidHandHardware/mechanical_structure/BOM_and_Assembly_Guide.pdf) for the remaining joints, palm and hand integration. Each finger module also needs one upstream [P1 idler pin](https://github.com/SYSU-RoboticsLab/RAPID-Hand/blob/main/RapidHandHardware/mechanical_structure/3D%20Printed%20Parts/P1.STL); the thumb does not.

Combine both BOMs, excluding the original fingertips being replaced. Count reused motors and P1 pins only once; the five module motors are not additional hand motors.

## 🔧 Assembly

1. Print four F1/F2 pairs and one T1/T2 pair; prepare four P1 pins.
2. Follow the [PDF assembly views](BOM_assembly_guide.pdf) for camera/motor orientation and fastener placement. Identify replacement parts from the views, not filenames alone.
3. Fit the camera boards and ribbon cables before tightening. Keep cables clear of screws, shell edges and moving joints.
4. Install the modules on the RAPID Hand structure. Check joint clearance, cable slack and all camera views before powered motion.
5. Follow [motor setup](../README.md) and [Teleop configuration](../../Teleop/README.md#configuration).

Finger and thumb modules use different screw lengths. Follow the PDF callouts and avoid overtightening printed parts or camera boards.

## 📄 Citation and License

See the [FingerViP citation](../../README.md#citation), [RAPID Hand paper](https://arxiv.org/abs/2506.07490) and [hardware license](../LICENSE).
