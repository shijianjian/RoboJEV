"""The console's wire: the messages, the WebSocket codec, the PNGs and the HTTP helpers.

Everything in `robojev.console` that has no simulator and no socket in it, which is deliberately
most of it -- a protocol that can only be exercised by starting MuJoCo is a protocol nobody checks.
What is pinned here is the shape of every message the page reads, the refusals a bad one gets, and
the two codecs written by hand because the standard library has neither.
"""
from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import pytest

from robojev.console import images as images_mod
from robojev.console import server as server_mod
from robojev.console import wire, ws


# ------------------------------------------------------------------------------ the handshake

def test_the_accept_key_is_the_one_in_the_rfc():
    """RFC 6455 section 1.3's own worked example. A digest that is right for everything except
    this string would produce a handshake every browser refuses and no test would say why."""
    assert ws.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_an_upgrade_is_recognised_by_both_headers_and_not_by_the_path():
    assert ws.is_upgrade({"upgrade": "WebSocket", "connection": "keep-alive, Upgrade"})
    assert not ws.is_upgrade({"upgrade": "websocket"})
    assert not ws.is_upgrade({"connection": "Upgrade"})


def test_the_handshake_answers_101_with_the_accept_header():
    out = ws.handshake({"sec-websocket-key": "dGhlIHNhbXBsZSBub25jZQ==",
                        "sec-websocket-version": "13"})
    assert out.startswith(b"HTTP/1.1 101 Switching Protocols\r\n")
    assert b"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n" in out


def test_a_handshake_without_a_key_or_at_another_version_is_refused():
    """A client speaking a version other than 13 speaks a different frame format, so answering
    101 to it would make a connection that fails on its first message with no explanation."""
    with pytest.raises(ws.WebSocketError):
        ws.handshake({})
    with pytest.raises(ws.WebSocketError):
        ws.handshake({"sec-websocket-key": "abc", "sec-websocket-version": "8"})


# ------------------------------------------------------------------------------- the frame codec

def mask(payload: bytes, opcode: int = ws.OP_TEXT, key: bytes = b"\x01\x02\x03\x04",
         fin: bool = True) -> bytes:
    """One *client* frame: what a browser sends, which is the only thing `Reader` accepts."""
    body = bytes(b ^ key[i & 3] for i, b in enumerate(payload))
    head = bytearray([(0x80 if fin else 0) | opcode])
    n = len(payload)
    if n < 126:
        head.append(0x80 | n)
    elif n < (1 << 16):
        head.append(0x80 | 126)
        head += struct.pack(">H", n)
    else:
        head.append(0x80 | 127)
        head += struct.pack(">Q", n)
    return bytes(head) + key + body


@pytest.mark.parametrize("size", [0, 5, 125, 126, 300, 70000])
def test_a_masked_client_frame_of_any_length_round_trips(size):
    payload = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    assert ws.Reader().feed(mask(payload)) == [(ws.OP_TEXT, payload)]


def test_the_length_field_widens_exactly_where_the_rfc_says_it_does():
    assert ws.encode(b"x" * 125)[1] == 125
    assert ws.encode(b"x" * 126)[1] == 126 and len(ws.encode(b"x" * 126)) == 126 + 4
    assert ws.encode(b"x" * 70000)[1] == 127 and len(ws.encode(b"x" * 70000)) == 70000 + 10


def test_a_server_frame_is_never_masked_and_a_client_reader_reads_it():
    frame = ws.encode_text("hello")
    assert not frame[1] & 0x80
    assert ws.Reader(require_mask=False).feed(frame) == [(ws.OP_TEXT, b"hello")]


def test_a_message_split_across_two_reads_arrives_once_and_whole():
    frame = mask(b'{"type":"step"}')
    reader = ws.Reader()
    assert reader.feed(frame[:3]) == []
    assert reader.feed(frame[3:]) == [(ws.OP_TEXT, b'{"type":"step"}')]


def test_fragments_are_joined_and_a_control_frame_between_them_is_not_swallowed():
    reader = ws.Reader()
    out = reader.feed(mask(b"ab", fin=False) + mask(b"", ws.OP_PING) + mask(b"cd", ws.OP_CONTINUE))
    assert out == [(ws.OP_PING, b""), (ws.OP_TEXT, b"abcd")]


def test_an_unmasked_client_frame_is_refused_rather_than_tolerated():
    """The mask is mandatory for a client. A server that reads unmasked client frames accepts
    traffic no conforming client produces, which is a door nobody meant to open."""
    with pytest.raises(ws.WebSocketError):
        ws.Reader().feed(ws.encode_text("hi"))


def test_a_continuation_with_nothing_to_continue_and_a_fragmented_control_frame_are_both_errors():
    with pytest.raises(ws.WebSocketError):
        ws.Reader().feed(mask(b"x", ws.OP_CONTINUE))
    with pytest.raises(ws.WebSocketError):
        ws.Reader().feed(mask(b"x", ws.OP_PING, fin=False))


def test_an_oversized_frame_is_refused_before_it_is_read():
    """Refused on the declared length, not after reading it: a frame claiming sixteen megabytes
    is not a command, and allocating for it before knowing that is how a local server becomes a
    way to exhaust a machine."""
    reader = ws.Reader(max_payload=64)
    with pytest.raises(ws.WebSocketError) as exc:
        reader.feed(mask(b"x" * 100))
    assert exc.value.code == ws.CLOSE_TOO_BIG


def test_a_reserved_bit_is_an_error_because_this_server_negotiates_no_extensions():
    frame = bytearray(mask(b"x"))
    frame[0] |= 0x40
    with pytest.raises(ws.WebSocketError):
        ws.Reader().feed(bytes(frame))


def test_a_close_frame_carries_the_code_first():
    body = ws.close_frame(ws.CLOSE_TOO_BIG, "too big")
    assert body[0] & 0x0F == ws.OP_CLOSE
    assert struct.unpack(">H", body[2:4])[0] == ws.CLOSE_TOO_BIG


# ------------------------------------------------------------------- client -> server commands

def test_every_op_the_console_reads_parses_to_itself():
    for op in ("step", "run", "pause", "reset", "ping"):
        assert wire.parse_command(json.dumps({"type": op})).op == op


def test_start_defaults_to_the_first_spatial_episode_and_the_scripted_expert():
    spec = wire.parse_command('{"type":"start"}').spec
    assert (spec.suite, spec.task, spec.init, spec.policy, spec.selection) == (
        "libero_spatial", 0, 0, "expert", "argmax")


def test_start_carries_the_scene_the_policy_and_the_selection():
    spec = wire.parse_command(json.dumps({
        "type": "start", "suite": "libero_object", "task": 4, "init": 2, "policy": "expert",
        "selection": "sample@0.7"}), policies=("expert",)).spec
    assert (spec.suite, spec.task, spec.init, spec.selection) == ("libero_object", 4, 2, "sample@0.7")


def test_a_bare_sample_and_a_temperature_are_spelled_into_the_one_canonical_form():
    """`selection` is what the record stores and what `policy.parse_selection` reads. Two
    spellings of one thing on the wire would become two spellings in the bundles."""
    spec = wire.parse_command('{"type":"start","selection":"sample","temperature":0.4}').spec
    assert spec.selection == "sample@0.4"


def test_a_policy_this_console_does_not_serve_is_refused_by_name():
    with pytest.raises(wire.CommandError) as exc:
        wire.parse_command('{"type":"start","policy":"model"}', policies=("expert",))
    assert "model" in str(exc.value) and "expert" in str(exc.value)


@pytest.mark.parametrize("raw, why", [
    ("not json", "not JSON"),
    ("[1,2]", "JSON object"),
    ('{"type":"fly"}', "unknown message type"),
    ('{"type":"override"}', "which question"),
    ('{"type":"start","task":"middle"}', "task"),
    ('{"type":"start","task":-1}', "task"),
])
def test_a_message_this_console_will_not_act_on_is_refused_with_the_reason(raw, why):
    with pytest.raises(wire.CommandError) as exc:
        wire.parse_command(raw)
    assert why in str(exc.value)


def test_an_override_names_a_question_and_a_candidate_and_null_takes_it_back_off():
    """There is no "no override" candidate to click instead, so `null` is how an armed override is
    disarmed -- a client that had to name one would be forcing an answer in order to un-force it."""
    armed = wire.parse_command('{"type":"override","qid":"move_x","candidate":"+"}')
    assert (armed.op, armed.qid, armed.candidate) == ("override", "move_x", "+")
    off = wire.parse_command('{"type":"override","qid":"move_x","candidate":null}')
    assert off.candidate is None


def test_save_may_name_the_bundle_and_tasks_may_name_the_suite():
    assert wire.parse_command('{"type":"save","name":"my-run"}').name == "my-run"
    assert wire.parse_command('{"type":"tasks","suite":"libero_10"}').suite == "libero_10"


# ------------------------------------------------------------------- server -> client messages

def test_hello_is_the_bundle_header_and_the_stream_beside_it():
    header = {"schema_version": 1, "id": "live-1", "suite": "libero_spatial"}
    message = wire.hello(header, {"kind": "poll"})
    assert message["type"] == "hello" and message["schema_version"] == 1
    assert message["episode"] is header and message["video"] == {"kind": "poll"}


def test_a_decision_message_carries_the_bundle_entry_untouched():
    """The whole contract of the live half: the page has one reader for a recorded decision and a
    streamed one, and it only does if the two are the same object."""
    entry = {"index": 0, "state": "x: +1.0 cm", "questions": {}, "action": [0.0] * 7}
    assert wire.decision(entry)["decision"] is entry


def test_status_states_are_a_closed_set_and_the_armed_overrides_ride_with_them():
    """The page draws the armed set from `status` and never from its own memory of what it
    clicked, so a refused override cannot leave a bar lit."""
    message = wire.status("running", step=85, decisions=17, overrides={"grip": "true"})
    assert message["state"] == "running" and message["overrides"] == {"grip": "true"}
    with pytest.raises(ValueError):
        wire.status("thinking")


def test_done_and_error_and_saved_say_what_they_are():
    assert wire.done(success=True, terminated_by="success", steps=144, decisions=29)["success"]
    assert wire.error("nope", fatal=True)["fatal"] is True
    saved = wire.saved(id="live-1", path="/tmp/live-1", url="replays/live-1/", decisions=3,
                       bytes_written=10, success=False)
    assert saved["url"] == "replays/live-1/" and saved["decisions"] == 3


def test_config_names_what_this_console_can_be_asked_for():
    message = wire.config(policies=("expert",), suites=["libero_spatial"],
                          default=wire.StartSpec(), video={"kind": "poll"}, state="idle",
                          replays="replays/")
    assert message["policies"] == ["expert"] and message["default"]["policy"] == "expert"
    assert message["protocol"] == wire.PROTOCOL_VERSION


def test_a_message_with_a_nan_in_it_fails_here_rather_than_in_a_browser():
    """`NaN` is not JSON and `JSON.parse` refuses it. A policy that produced one has to be visible
    at the server rather than as a dead socket three layers away."""
    with pytest.raises(ValueError):
        wire.dumps({"type": "status", "p": float("nan")})
    assert wire.dumps({"type": "pong"}) == '{"type":"pong"}'


# -------------------------------------------------------------------------------------- PNGs

def test_a_png_is_a_png_and_holds_the_pixels_it_was_given():
    rgb = (np.arange(8 * 6 * 3, dtype=np.uint8).reshape(6, 8, 3))
    data = images_mod.encode_png(rgb)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, colour = struct.unpack(">IIBB", data[16:26])
    assert (width, height, depth, colour) == (8, 6, 8, 2)
    raw = zlib.decompress(data[data.index(b"IDAT") + 4:-12])
    rows = np.frombuffer(raw, np.uint8).reshape(6, 8 * 3 + 1)
    assert (rows[:, 0] == 0).all()                                # every row filtered "none"
    assert np.array_equal(rows[:, 1:].reshape(6, 8, 3), rgb)


def test_a_greyscale_or_rgba_render_still_encodes_as_three_channels():
    assert images_mod.encode_png(np.zeros((4, 4), np.uint8))[:8] == b"\x89PNG\r\n\x1a\n"
    assert images_mod.encode_png(np.zeros((4, 4, 4), np.uint8))[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_frame_buffer_encodes_lazily_caches_by_sequence_and_forgets_on_clear():
    buffer = images_mod.FrameBuffer()
    assert buffer.png("agentview") is None                        # nothing rendered yet
    buffer.publish({"agentview": np.zeros((4, 4, 3), np.uint8)})
    first = buffer.png("agentview")
    assert first is not None and buffer.png("agentview")[1] is first[1]   # cached, not re-encoded
    buffer.publish({"agentview": np.full((4, 4, 3), 255, np.uint8)})
    assert buffer.png("agentview")[0] == first[0] + 1
    assert buffer.png("agentview")[1] != first[1]
    buffer.clear()
    assert buffer.png("agentview") is None


def test_an_empty_publish_does_not_blank_the_last_picture():
    """An environment that renders nothing must not take the page's picture away."""
    buffer = images_mod.FrameBuffer()
    buffer.publish({"agentview": np.zeros((4, 4, 3), np.uint8)})
    buffer.publish({})
    assert buffer.png("agentview") is not None


# ---------------------------------------------------------------------------- the HTTP helpers

@pytest.mark.parametrize("header, size, want", [
    (None, 100, None),
    ("", 100, None),
    ("bytes=0-9", 100, (0, 9)),
    ("bytes=10-", 100, (10, 99)),
    ("bytes=-20", 100, (80, 99)),
    ("bytes=0-999", 100, (0, 99)),
    ("bytes=100-", 100, (-1, -1)),
    ("bytes=abc", 100, None),
])
def test_byte_ranges_are_read_the_way_a_seeking_browser_writes_them(header, size, want):
    """A browser seeking an mp4 asks for a range, and a server that answers the whole file to one
    makes seeking silently unreliable -- the site's own bug hiding behind the harness's."""
    assert server_mod.parse_range(header, size) == want


def test_a_request_line_is_parsed_into_a_path_a_query_and_keep_alive():
    import asyncio

    async def parse(raw: bytes):
        reader = asyncio.StreamReader()
        reader.feed_data(raw)
        reader.feed_eof()
        return await server_mod.read_request(reader)

    request = asyncio.run(parse(b"GET /frame/wrist.png?seq=4 HTTP/1.1\r\nHost: x\r\n\r\n"))
    assert (request.method, request.path, request.query["seq"]) == ("GET", "/frame/wrist.png", ["4"])
    assert request.keep_alive


def test_the_console_tells_the_page_it_is_a_console_and_leaves_a_pages_build_alone():
    """The alternative -- letting the page probe its own origin -- costs a request and a console
    error on every static visit, which is the one thing the GitHub Pages deployment must not have."""
    html = server_mod.inject(b"<html><head><title>x</title></head><body></body></html>",
                             "ws://127.0.0.1:8765/ws")
    assert b'window.__ROBOJEV_CONSOLE__={"ws":"ws://127.0.0.1:8765/ws"}' in html
    assert html.index(b"__ROBOJEV_CONSOLE__") < html.index(b"</head>")


def test_content_types_cover_what_the_built_app_is_made_of():
    import pathlib

    assert server_mod.content_type(pathlib.Path("a.mp4")) == "video/mp4"
    assert server_mod.content_type(pathlib.Path("a.js")).startswith("text/javascript")
    assert server_mod.content_type(pathlib.Path("a.unknown")) == "application/octet-stream"
