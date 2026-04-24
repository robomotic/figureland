"""
Custom exception types for Figureland.
"""


class FigurelandError(Exception):
    """Base exception class for all Figureland errors."""
    pass


class ShapeOutOfBoundsError(FigurelandError):
    """Raised when a shape is placed outside environment boundaries."""
    pass


class ShapePenetrationError(FigurelandError):
    """Raised when shapes are penetrating each other."""
    pass


class DeviceMismatchError(FigurelandError):
    """Raised when tensors are on different devices."""
    pass


class BatchSizeMismatchError(FigurelandError):
    """Raised when batch sizes don't match between shapes."""
    pass


class EmptyEnvironmentError(FigurelandError):
    """Raised when physics simulation runs with no shapes."""
    pass
