"""
Provide low-level OpenCL accelerated operations. This backend requires that
PyOpenCL be installed.

"""

from .transform2d import TransformDomainSignal, Transform2d

__all__ = [
    'TransformDomainSignal',
    'Transform2d',
]
