"""The camera frames, as bytes a browser can draw -- with nothing but the standard library.

The live view is two pictures updated twenty times a second. A recorded bundle answers that with
an mp4, which a live episode cannot have: the file is still being written. So the console hands
the page single images instead, one HTTP request per picture, and the page asks for the next one
the moment the previous has painted (`web/PROTOCOL.md`, "Frames").

**PNG, not JPEG, and the reason is the dependency list.** The base install of this package is
numpy and nothing else, and there is no JPEG encoder in the standard library -- Pillow would be a
new dependency for the console alone. `zlib` and `struct` *are* in it, and a PNG is a header, one
deflated block of filtered rows and a checksum, which is the thirty lines below. A 256-square RGB
render is about 100 kB at deflate level 1 and takes a couple of milliseconds to encode; at 20 Hz
and two cameras that is 4 MB/s and a few per cent of one core, over a loopback socket.

Encoding is **lazy and cached by sequence number**: a frame nobody asks for is never compressed,
so a console with no browser attached costs nothing, and two requests for the same frame compress
it once.
"""
from __future__ import annotations

import struct
import threading
import zlib

import numpy as np

#: Deflate level for the frame PNGs. Level 1 on a 256-square render is ~2 ms and ~100 kB; level 6
#: is ~12 ms and ~85 kB. The wire is a loopback socket, so the cheap end is the right end.
PNG_LEVEL: int = 1


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def encode_png(image) -> bytes:
    """One RGB array as a PNG file. 8-bit truecolour, no interlace, filter 0 on every row."""
    a = np.ascontiguousarray(np.asarray(image))
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    a = np.ascontiguousarray(a[:, :, :3])
    height, width = a.shape[:2]
    # Filter byte 0 ("none") in front of every row, which is what the scanline format is.
    raw = np.zeros((height, width * 3 + 1), dtype=np.uint8)
    raw[:, 1:] = a.reshape(height, width * 3)
    return b"".join((
        b"\x89PNG\r\n\x1a\n",
        _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        _chunk(b"IDAT", zlib.compress(raw.tobytes(), PNG_LEVEL)),
        _chunk(b"IEND", b""),
    ))


class FrameBuffer:
    """The latest render of each camera, shared between the episode thread and the HTTP server.

    The episode thread `publish`es whole dicts of arrays; the server `png`s whichever camera a
    request names. One lock guards both, and the encoded bytes are cached against the sequence
    number the arrays arrived with, so the picture a request gets is a picture that really was
    rendered rather than two halves of two frames.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._images: dict[str, np.ndarray] = {}
        self._encoded: dict[str, tuple[int, bytes]] = {}
        self._seq = 0

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    def publish(self, images: dict) -> None:
        """A new frame. An empty dict is ignored: an environment that renders nothing must not
        blank the last picture the page has."""
        if not images:
            return
        with self._lock:
            self._seq += 1
            self._images = {name: np.asarray(img) for name, img in images.items()}

    def clear(self) -> None:
        with self._lock:
            self._seq += 1
            self._images = {}
            self._encoded = {}

    def cameras(self) -> list[str]:
        with self._lock:
            return sorted(self._images)

    def png(self, camera: str) -> tuple[int, bytes] | None:
        """`(sequence, png bytes)` for `camera`, or None while nothing has been rendered."""
        with self._lock:
            image = self._images.get(camera)
            seq = self._seq
            if image is None:
                return None
            hit = self._encoded.get(camera)
            if hit is not None and hit[0] == seq:
                return hit
        # Compressed outside the lock: it is a few milliseconds and the episode thread must not
        # wait on a request. The sequence check above is repeated on store, so a frame that moved
        # on while this was encoding loses the race rather than poisoning the cache.
        data = encode_png(image)
        with self._lock:
            if self._seq == seq:
                self._encoded[camera] = (seq, data)
        return seq, data


__all__ = ["PNG_LEVEL", "FrameBuffer", "encode_png"]
