from __future__ import annotations

import hashlib
import math
import re
import struct
from typing import Iterable, List


VECTOR_DIMENSIONS = 512


def _features(text: str) -> Iterable[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    compact = re.sub(r"\s+", "", normalized)
    for size in (2, 3):
        for index in range(max(0, len(compact) - size + 1)):
            yield compact[index:index + size]
    for token in re.findall(r"[a-z0-9._-]{2,}|[\u4e00-\u9fff]{2,}", normalized):
        yield "w:" + token


def encode_vector(text: str) -> bytes:
    values = [0.0] * VECTOR_DIMENSIONS
    for feature in _features(text):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "big")
        index = raw % VECTOR_DIMENSIONS
        values[index] += -1.0 if raw & (1 << 63) else 1.0
    norm = math.sqrt(sum(value * value for value in values))
    if norm:
        values = [value / norm for value in values]
    return struct.pack("<%df" % VECTOR_DIMENSIONS, *values)


def cosine_blob(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    count = len(left) // 4
    a = struct.unpack("<%df" % count, left)
    b = struct.unpack("<%df" % count, right)
    return sum(x * y for x, y in zip(a, b))
