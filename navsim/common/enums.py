from enum import IntEnum


class StateSE2Index(IntEnum):
    """Intenum for SE(2) arrays."""

    _X = 0
    _Y = 1
    _HEADING = 2

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)


class BoundingBoxIndex(IntEnum):
    """Intenum of bounding boxes in logs.

    A box row is [x, y, z, length, width, height, roll, pitch, yaw]. It used to
    be seven wide, ending at a single heading; roll and pitch were added at 6
    and 7 and the heading moved to 8. HEADING is kept as an alias of YAW so the
    feature builders that index through this enum follow the move on their own
    -- but anything still reaching for a bare column 6 now gets roll.
    """

    _X = 0
    _Y = 1
    _Z = 2
    _LENGTH = 3
    _WIDTH = 4
    _HEIGHT = 5
    _ROLL = 6
    _PITCH = 7
    _YAW = 8
    _HEADING = 8  # alias of _YAW: the name the log format used before roll/pitch

    @classmethod
    def size(cls):
        # Counted off the last column rather than by walking dir(cls), which
        # sees _HEADING and _YAW as two entries and would report one too many.
        return cls._YAW + 1

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def Z(cls):
        return cls._Z

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def HEIGHT(cls):
        return cls._HEIGHT

    @classmethod
    @property
    def ROLL(cls):
        return cls._ROLL

    @classmethod
    @property
    def PITCH(cls):
        return cls._PITCH

    @classmethod
    @property
    def YAW(cls):
        return cls._YAW

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def ORIENTATION(cls):
        # assumes ROLL, PITCH, YAW have subsequent indices
        return slice(cls._ROLL, cls._YAW + 1)

    @classmethod
    @property
    def POINT2D(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def POSITION(cls):
        # assumes X, Y, Z have subsequent indices
        return slice(cls._X, cls._Z + 1)

    @classmethod
    @property
    def DIMENSION(cls):
        # assumes LENGTH, WIDTH, HEIGHT have subsequent indices
        return slice(cls._LENGTH, cls._HEIGHT + 1)


class LidarIndex(IntEnum):
    """Intenum for lidar point cloud arrays."""

    _X = 0
    _Y = 1
    _Z = 2
    _INTENSITY = 3
    _RING = 4
    _ID = 5

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def Z(cls):
        return cls._Z

    @classmethod
    @property
    def INTENSITY(cls):
        return cls._INTENSITY

    @classmethod
    @property
    def RING(cls):
        return cls._RING

    @classmethod
    @property
    def ID(cls):
        return cls._ID

    @classmethod
    @property
    def POINT2D(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def POSITION(cls):
        # assumes X, Y, Z have subsequent indices
        return slice(cls._X, cls._Z + 1)
