"""The server half of RFC 6455, in about a hundred lines of standard library.

The console needs one WebSocket on one loopback socket, carrying small JSON text frames a few
times a second. That is the part of the protocol written below: the opening handshake, the frame
codec, fragmentation, and the three control opcodes. What is deliberately absent is everything a
library would add for the open internet -- permessage-deflate, subprotocol negotiation, `Origin`
policy, per-message size accounting across a long connection -- because the alternative to writing
this is a dependency, and a dependency is the one thing the base install of this package does not
have (`pyproject.toml`: numpy, and nothing else).

Two rules the codec keeps, both of them things a server gets wrong quietly rather than loudly:

1. **Every client frame must be masked** and a server frame must not be. An unmasked client frame
   is a protocol error and is refused here rather than read, because reading it would mean this
   server accepts traffic a conforming client never sends.
2. **A payload has a cap.** The page sends commands of a few hundred bytes; a frame claiming
   sixteen megabytes is not a command, and allocating for it before knowing that is how a local
   server becomes a way to exhaust a machine's memory.
"""
from __future__ import annotations

import base64
import hashlib
import struct

#: The magic string RFC 6455 concatenates with the client's key before the digest.
GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONTINUE = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

#: The biggest frame this server will read. The largest thing the page ever sends is a `start`
#: with a checkpoint path in it.
MAX_PAYLOAD: int = 1 << 20

#: Close codes this server sends.
CLOSE_NORMAL = 1000
CLOSE_PROTOCOL_ERROR = 1002
CLOSE_TOO_BIG = 1009


class WebSocketError(Exception):
    """The connection cannot continue: a bad handshake, an unmasked frame, an oversized payload."""

    def __init__(self, message: str, code: int = CLOSE_PROTOCOL_ERROR):
        super().__init__(message)
        self.code = code


def accept_key(key: str) -> str:
    """`Sec-WebSocket-Accept` for a client's `Sec-WebSocket-Key`."""
    return base64.b64encode(hashlib.sha1(key.strip().encode("ascii") + GUID).digest()).decode("ascii")


def is_upgrade(headers: dict) -> bool:
    """Whether these request headers ask for a WebSocket. Lower-cased keys."""
    return ("websocket" in (headers.get("upgrade") or "").lower()
            and "upgrade" in (headers.get("connection") or "").lower())


def handshake(headers: dict) -> bytes:
    """The `101 Switching Protocols` response for an upgrade request, or a raised error.

    The version check is not ceremony: a client speaking anything but 13 speaks a different frame
    format, and answering 101 to it would produce a connection that fails on its first message
    with no explanation.
    """
    key = headers.get("sec-websocket-key")
    if not key:
        raise WebSocketError("the upgrade request carried no Sec-WebSocket-Key")
    version = (headers.get("sec-websocket-version") or "").strip()
    if version and version != "13":
        raise WebSocketError(f"this server speaks WebSocket version 13, the client asked for {version!r}")
    return (
        b"HTTP/1.1 101 Switching Protocols\r\n"
        b"Upgrade: websocket\r\n"
        b"Connection: Upgrade\r\n"
        b"Sec-WebSocket-Accept: " + accept_key(key).encode("ascii") + b"\r\n\r\n"
    )


def encode(payload: bytes, opcode: int = OP_TEXT) -> bytes:
    """One unfragmented server frame. Server frames are never masked."""
    length = len(payload)
    header = bytearray([0x80 | (opcode & 0x0F)])
    if length < 126:
        header.append(length)
    elif length < (1 << 16):
        header.append(126)
        header += struct.pack(">H", length)
    else:
        header.append(127)
        header += struct.pack(">Q", length)
    return bytes(header) + payload


def encode_text(text: str) -> bytes:
    return encode(text.encode("utf-8"), OP_TEXT)


def close_frame(code: int = CLOSE_NORMAL, reason: str = "") -> bytes:
    return encode(struct.pack(">H", code) + reason.encode("utf-8")[:123], OP_CLOSE)


class Reader:
    """Bytes in, whole messages out.

    `feed(data)` returns a list of `(opcode, payload)` for every message that completed, with
    continuation frames already joined; control frames come out as themselves and are never
    fragmented (RFC 6455 forbids it, and a client that does it is refused).
    """

    def __init__(self, max_payload: int = MAX_PAYLOAD, require_mask: bool = True):
        self.max_payload = int(max_payload)
        # True on a server, which must refuse an unmasked client frame; false for the client half
        # a test speaks the other end of, where the server's frames are unmasked by the same rule.
        self.require_mask = bool(require_mask)
        self._buffer = bytearray()
        self._fragments = bytearray()
        self._fragment_opcode: int | None = None

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        self._buffer += data
        out: list[tuple[int, bytes]] = []
        while True:
            frame = self._take()
            if frame is None:
                return out
            fin, opcode, payload = frame
            if opcode >= 0x8:
                if not fin:
                    raise WebSocketError("a control frame was fragmented")
                out.append((opcode, payload))
                continue
            if opcode == OP_CONTINUE:
                if self._fragment_opcode is None:
                    raise WebSocketError("a continuation frame arrived with nothing to continue")
                self._fragments += payload
            else:
                if self._fragment_opcode is not None:
                    raise WebSocketError("a new message began before the previous one finished")
                self._fragment_opcode = opcode
                self._fragments = bytearray(payload)
            if len(self._fragments) > self.max_payload:
                raise WebSocketError(
                    f"a message over {self.max_payload} bytes; this console reads commands, not "
                    f"uploads", CLOSE_TOO_BIG)
            if fin:
                out.append((self._fragment_opcode, bytes(self._fragments)))
                self._fragments = bytearray()
                self._fragment_opcode = None

    def _take(self) -> tuple[bool, int, bytes] | None:
        """One frame off the front of the buffer, or None while it is incomplete."""
        buf = self._buffer
        if len(buf) < 2:
            return None
        first, second = buf[0], buf[1]
        fin = bool(first & 0x80)
        if first & 0x70:
            raise WebSocketError("a reserved frame bit was set; this server negotiates no extensions")
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        at = 2
        if length == 126:
            if len(buf) < at + 2:
                return None
            length = struct.unpack_from(">H", buf, at)[0]
            at += 2
        elif length == 127:
            if len(buf) < at + 8:
                return None
            length = struct.unpack_from(">Q", buf, at)[0]
            at += 8
        if length > self.max_payload:
            raise WebSocketError(
                f"a frame declaring {length} bytes, over this console's {self.max_payload}-byte "
                f"limit", CLOSE_TOO_BIG)
        if not masked and self.require_mask:
            # Refused rather than tolerated: the mask is mandatory for a client, and a server that
            # reads unmasked client frames accepts traffic no conforming client produces.
            raise WebSocketError("a client frame arrived unmasked")
        if len(buf) < at + (4 if masked else 0) + length:
            return None
        mask = b""
        if masked:
            mask = bytes(buf[at:at + 4])
            at += 4
        payload = bytes(buf[at:at + length])
        at += length
        del buf[:at]
        if length and mask:
            data = bytearray(payload)
            for i in range(length):
                data[i] ^= mask[i & 3]
            payload = bytes(data)
        return fin, opcode, payload


__all__ = ["CLOSE_NORMAL", "CLOSE_PROTOCOL_ERROR", "CLOSE_TOO_BIG", "GUID", "MAX_PAYLOAD",
           "OP_BINARY", "OP_CLOSE", "OP_CONTINUE", "OP_PING", "OP_PONG", "OP_TEXT", "Reader",
           "WebSocketError", "accept_key", "close_frame", "encode", "encode_text", "handshake",
           "is_upgrade"]
