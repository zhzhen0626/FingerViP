"""Load input backends only when requested, preserving the public exports."""

from importlib import import_module

_EXPORTS = {
    'ImageInput': '.image_input',
    'VisionProInput': '.visionpro_input',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
