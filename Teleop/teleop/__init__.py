"""Load input backends only when requested, preserving the public exports."""

from importlib import import_module

_EXPORTS = {
    'KinectInputProducer': '.hand_pose_stream.image_loader.kinect_input_producer',
    'VideoInputProducer': '.hand_pose_stream.image_loader.video_input_producer',
    'ImageInputProducer': '.hand_pose_stream.image_loader.image_input_producer',
    'AndroidInputProducer': '.hand_pose_stream.image_loader.android_input_producer',
    'TeleopProcessor': '.teleop_processor',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
