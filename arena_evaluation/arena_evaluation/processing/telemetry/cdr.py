"""Direct readers of the CDR layout of messages whose generic decoding dominates an extraction."""

from __future__ import annotations

import struct
import typing

import numpy as np

PATH_SCHEMA = "nav_msgs/msg/Path"
_U32 = struct.Struct("<I")
_POSE = struct.Struct("<7d")
_POSE_BYTES = 56  # geometry_msgs/Pose: position x, y, z and orientation x, y, z, w, float64


class PathPoses(typing.NamedTuple):
    """The poses of a nav_msgs/Path as an (n, 7) float64 array: position x, y, z and orientation x, y, z, w."""

    values: np.ndarray


def path_poses(data: bytes | memoryview) -> np.ndarray | None:
    """The poses of a serialized nav_msgs/Path as an (n, 7) float64 array (position x, y, z, orientation x, y, z, w),
    the values mcap_ros2 decodes; None unless the payload is little-endian plain CDR, the encoding ROS 2 records.

    It walks the CDR layout directly: a robot's global plan holds thousands of PoseStamped, and the generic decoder
    spends most of a fleet episode's extraction building them. Poses whose header frame ids have one length (always, in
    practice) are read with one strided gather, any others pose by pose."""
    if len(data) < 8 or data[0] != 0 or data[1] != 1:
        return None
    o = 4
    (k,) = _U32.unpack_from(data, o + 8)  # header: stamp, then frame_id
    o = 4 + ((o + 12 + k - 1) & ~3)  # align 4 counted from the payload start
    (n,) = _U32.unpack_from(data, o)
    o += 4
    out = np.empty((n, 7), dtype=np.float64)
    if n == 0:
        return out
    buf = np.frombuffer(data, dtype=np.uint8)
    i = 0
    while i < n:
        o = 4 + ((o - 1) & ~3)  # a PoseStamped starts 4-aligned: stamp, frame_id, then the pose 8-aligned
        (k,) = _U32.unpack_from(data, o + 8)
        q = 4 + ((o + 12 + k + 3) & ~7)
        if (o - 4) % 8 == 0 and i + 1 < n:
            # from an 8-aligned start every pose with a frame id of this length takes the same stride
            stride = (q - o) + _POSE_BYTES
            m = min(n - i, (len(data) - o) // stride)
            starts = o + stride * np.arange(m)
            lengths = buf[(starts + 8)[:, None] + np.arange(4)].view("<u4").ravel()
            same = int(np.argmin(lengths == k)) if not (lengths == k).all() else m
            if same > 1:
                out[i : i + same] = buf[(starts[:same] + (q - o))[:, None] + np.arange(_POSE_BYTES)].view("<f8")
                i += same
                o = int(starts[same - 1]) + stride
                continue
        out[i] = _POSE.unpack_from(data, q)
        i += 1
        o = q + _POSE_BYTES
    return out


def path_decoder(generic: typing.Callable[[bytes], typing.Any]) -> typing.Callable[[bytes], typing.Any]:
    """A nav_msgs/Path decoder that reads the poses directly (`path_poses`), falling back to `generic`."""

    def decode(data: bytes) -> typing.Any:  # noqa: ANN401 - a decoded Path or its poses
        poses = path_poses(data)
        return generic(data) if poses is None else PathPoses(poses)

    return decode


__all__ = ["PATH_SCHEMA", "PathPoses", "path_decoder", "path_poses"]
