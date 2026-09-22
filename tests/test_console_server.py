"""The console's server, driven over a real socket.

Everything below opens a port, speaks HTTP and RFC 6455 at it and reads what comes back, because
the three things this file is about are things a unit test of the pieces cannot see: that the
handshake a browser sends is answered, that a second browser is turned away rather than quietly
shown a session it cannot drive, and that the built app and the recorded bundles are served the
way GitHub Pages serves them -- byte ranges included, since a browser seeking an mp4 asks for one.

The simulator is the same thirty-line stub `test_console_session.py` uses.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import pathlib

import pytest

from robojev import policy as policy_mod
from robojev.console import ws
from robojev.console.server import Console
# By module name and not `tests.…`: `tests/` is not a package, and pytest puts the directory
# itself on the path -- which is the only spelling that works whether or not the checkout root
# happens to be importable.
from test_console_session import RenderEnv, StubEnv


class Client:
    """The browser half: a handshake, masked frames out, unmasked frames in."""

    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer
        self.frames = ws.Reader(require_mask=False)
        self.inbox: list[dict] = []
        self.closed = False
        #: How far `wait` has read; each call resumes where the last one stopped.
        self._at = 0

    @classmethod
    async def connect(cls, port: int) -> "Client":
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        writer.write(f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
                     f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                     f"Sec-WebSocket-Version: 13\r\n\r\n".encode())
        await writer.drain()
        head = await reader.readuntil(b"\r\n\r\n")
        assert head.startswith(b"HTTP/1.1 101"), head
        assert f"Sec-WebSocket-Accept: {ws.accept_key(key)}".encode() in head
        return cls(reader, writer)

    async def send(self, **message) -> None:
        payload = json.dumps(message).encode()
        mask = os.urandom(4)
        body = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        header = bytearray([0x81, 0x80 | len(payload)])           # every command is under 126 bytes
        self.writer.write(bytes(header) + mask + body)
        await self.writer.drain()

    async def _pump(self, timeout: float) -> None:
        data = await asyncio.wait_for(self.reader.read(65536), timeout)
        if not data:
            self.closed = True
            return
        for opcode, payload in self.frames.feed(data):
            if opcode == ws.OP_CLOSE:
                self.closed = True
            elif opcode == ws.OP_TEXT:
                self.inbox.append(json.loads(payload.decode()))

    async def wait(self, type_: str, timeout: float = 20.0, **fields) -> dict:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            while self._at < len(self.inbox):
                message = self.inbox[self._at]
                self._at += 1
                if message.get("type") == type_ and all(message.get(k) == v
                                                        for k, v in fields.items()):
                    return message
            if self.closed or loop.time() > deadline:
                raise AssertionError(f"no {type_} {fields or ''} in "
                                     f"{[m.get('type') for m in self.inbox]}")
            try:
                await self._pump(max(0.05, deadline - loop.time()))
            except asyncio.TimeoutError:
                pass

    async def close(self) -> None:
        self.writer.close()


async def http(port: int, path: str, headers: str = "", method: str = "GET"):
    """One request on one connection. Returns `(status, headers, body)`."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{headers}"
                 f"Connection: close\r\n\r\n".encode())
    await writer.drain()
    raw = await reader.read(-1)
    writer.close()
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ")[1])
    fields = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        fields[name.strip().lower()] = value.strip()
    return status, fields, body


def build(tmp_path, *, env=None, **kw) -> Console:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_bytes(b"<html><head><title>RoboJEV</title></head><body>x</body></html>")
    (dist / "assets" / "app.js").write_bytes(b"console.log(1)\n")
    (dist / "replays").mkdir()
    (dist / "replays" / "index.json").write_text("[]")
    replays = tmp_path / "replays"
    replays.mkdir()
    (replays / "clip.mp4").write_bytes(bytes(range(256)) * 4)
    environment = env or StubEnv()
    return Console(host="127.0.0.1", port=0, dist=dist, replays=replays, policies=("expert",),
                   render_size=8, log=lambda *a: None,
                   env_factory=lambda *a, **k: environment,
                   build_policy=lambda spec: policy_mod.build(
                       "expert", spec.suite, seed=spec.seed, selection=spec.selection), **kw)


def run(console: Console, body):
    """Start the server, run one coroutine against it, and stop both."""
    async def main():
        loop = asyncio.get_running_loop()
        console._loop = loop
        server = await asyncio.start_server(console._connection, console.host, 0)
        console.port = int(server.sockets[0].getsockname()[1])
        console.session.base_url = f"http://127.0.0.1:{console.port}"
        async with server:
            try:
                return await asyncio.wait_for(body(console.port), 60)
            finally:
                await loop.run_in_executor(None, console.session.close)

    return asyncio.run(main())


# --------------------------------------------------------------------------------- the socket

def test_a_connecting_page_is_told_what_this_console_can_be_asked_for(tmp_path):
    async def body(port):
        client = await Client.connect(port)
        config = await client.wait("config")
        assert config["policies"] == ["expert"]
        assert "libero_spatial" in config["suites"]
        assert config["state"] == "idle" and config["replays"] == "replays/"
        assert config["video"]["cameras"]["wrist"].endswith("/frame/wrist.png")
        assert (await client.wait("status"))["state"] == "idle"
        await client.close()

    run(build(tmp_path), body)


def test_a_second_browser_is_turned_away_in_the_protocols_own_words(tmp_path):
    """One console, one simulator. Two pages driving one arm is not a race worth having, and a
    second tab that silently showed a session it cannot command would be worse than a refusal."""
    async def body(port):
        first = await Client.connect(port)
        await first.wait("config")
        second = await Client.connect(port)
        refusal = await second.wait("error")
        assert refusal["fatal"] is True and "another window" in refusal["message"]
        await second._pump(2.0)
        assert second.closed is True
        # The first one is untouched: it still has the socket and is still answered.
        await first.send(type="ping")
        await first.wait("pong")
        await first.close()

    run(build(tmp_path), body)


def test_the_socket_leaves_with_the_browser_and_the_next_one_is_let_in(tmp_path):
    async def body(port):
        first = await Client.connect(port)
        await first.wait("config")
        first.writer.close()
        await asyncio.sleep(0.2)
        second = await Client.connect(port)
        assert (await second.wait("config"))["state"] == "idle"
        await second.close()

    run(build(tmp_path), body)


def test_a_command_this_console_will_not_act_on_comes_back_as_an_error_not_a_silence(tmp_path):
    async def body(port):
        client = await Client.connect(port)
        await client.wait("config")
        await client.send(type="step")
        assert "no episode" in (await client.wait("error"))["message"]
        await client.send(type="start", policy="jev")
        assert "jev" in (await client.wait("error", timeout=10.0))["message"]
        await client.close()

    run(build(tmp_path), body)


def test_an_episode_is_driven_saved_and_reset_over_the_socket(tmp_path):
    """The whole round trip a browser makes, at the level the browser makes it."""
    async def body(port):
        client = await Client.connect(port)
        await client.wait("config")
        await client.send(type="start", task=0, init=0, policy="expert", max_steps=20)
        hello = await client.wait("hello")
        assert hello["episode"]["max_steps"] == 20
        await client.wait("status", state="paused")

        await client.send(type="step")
        first = (await client.wait("decision"))["decision"]
        assert first["index"] == 0

        forced = "+" if first["questions"]["move_z"]["choice"] != "+" else "-"
        await client.send(type="override", qid="move_z", candidate=forced)
        await client.wait("status", overrides={"move_z": forced})
        await client.send(type="step")
        second = (await client.wait("decision"))["decision"]
        assert second["index"] == 1
        assert second["questions"]["move_z"]["overridden"] is True

        await client.send(type="run")
        done = await client.wait("done", timeout=40.0)
        assert done["terminated_by"] == "max_steps"

        await client.send(type="save", name="over-the-socket")
        saved = await client.wait("saved", timeout=40.0)
        bundle = json.loads((pathlib.Path(saved["path"]) / "episode.json").read_text())
        assert bundle["decisions"][1]["questions"]["move_z"]["overridden"] is True

        await client.send(type="reset")
        await client.wait("status", state="idle", timeout=40.0)
        await client.close()

    run(build(tmp_path), body)


# ----------------------------------------------------------------------------- the static half

def test_the_page_is_served_from_dist_and_told_it_is_a_console(tmp_path):
    async def body(port):
        status, headers, page = await http(port, "/")
        assert status == 200 and headers["content-type"].startswith("text/html")
        assert f'"ws://127.0.0.1:{port}/ws"'.encode() in page
        assert b"__ROBOJEV_CONSOLE__" in page
        # And the assets beside it, unmodified.
        status, headers, body_ = await http(port, "/assets/app.js")
        assert status == 200 and body_ == b"console.log(1)\n"
        assert headers["content-type"].startswith("text/javascript")

    run(build(tmp_path), body)


def test_a_bundles_video_answers_a_range_request_with_the_range(tmp_path):
    """A browser seeking an mp4 asks for a range; a server that answers the whole file to one
    makes seeking silently unreliable, which is the site's bug hiding behind the harness's."""
    async def body(port):
        status, headers, body_ = await http(port, "/replays/clip.mp4", "Range: bytes=10-19\r\n")
        assert status == 206 and body_ == bytes(range(10, 20))
        assert headers["content-range"] == "bytes 10-19/1024"
        assert headers["content-type"] == "video/mp4"
        status, _, whole = await http(port, "/replays/clip.mp4")
        assert status == 200 and len(whole) == 1024
        status, headers, _ = await http(port, "/replays/clip.mp4", "Range: bytes=9999-\r\n")
        assert status == 416 and headers["content-range"] == "bytes */1024"

    run(build(tmp_path), body)


def test_the_replays_directory_wins_over_the_built_copy_so_a_save_needs_no_rebuild(tmp_path):
    console = build(tmp_path)
    (console.replays / "index.json").write_text('[{"id":"live-1"}]')

    async def body(port):
        status, _, body_ = await http(port, "/replays/index.json")
        assert status == 200 and json.loads(body_) == [{"id": "live-1"}]

    run(console, body)


def test_a_path_that_climbs_out_of_the_served_directories_is_a_404(tmp_path):
    async def body(port):
        for path in ("/../secret.txt", "/assets/../../secret.txt", "/nope.js"):
            status, _, _ = await http(port, path)
            assert status == 404, path

    (tmp_path / "secret.txt").write_text("no")
    run(build(tmp_path), body)


def test_only_get_and_head_are_answered(tmp_path):
    async def body(port):
        status, _, _ = await http(port, "/", method="POST")
        assert status == 405
        status, headers, body_ = await http(port, "/assets/app.js", method="HEAD")
        assert status == 200 and body_ == b"" and headers["content-length"] == "15"

    run(build(tmp_path), body)


# -------------------------------------------------------------------------------- the pictures

def test_the_camera_endpoint_is_empty_before_an_episode_and_a_png_during_one(tmp_path):
    """204 and not 404 while nothing has been rendered: a page polling for the first picture must
    not fill a console with red lines while it waits for one."""
    async def body(port):
        status, _, _ = await http(port, "/frame/agentview.png")
        assert status == 204

        client = await Client.connect(port)
        await client.wait("config")
        await client.send(type="start", max_steps=10)
        await client.wait("hello")
        await client.send(type="step")
        await client.wait("decision")

        status, headers, png = await http(port, "/frame/agentview.png")
        assert status == 200 and png[:8] == b"\x89PNG\r\n\x1a\n"
        assert headers["content-type"] == "image/png" and headers["cache-control"] == "no-store"
        assert int(headers["x-frame-seq"]) > 0
        status, _, wrist = await http(port, "/frame/wrist.png")
        assert status == 200 and wrist != png
        status, _, _ = await http(port, "/frame/nosuchcamera.png")
        assert status == 204
        await client.close()

    run(build(tmp_path, env=RenderEnv()), body)


def test_one_connection_answers_several_requests(tmp_path):
    """Keep-alive, because the page asks for a picture twenty times a second per camera and a
    connection per picture would be forty handshakes a second for nothing."""
    async def body(port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        for _ in range(3):
            writer.write(f"GET /assets/app.js HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
            await writer.drain()
            head = await reader.readuntil(b"\r\n\r\n")
            assert head.startswith(b"HTTP/1.1 200")
            assert await reader.readexactly(15) == b"console.log(1)\n"
        writer.close()

    run(build(tmp_path), body)


@pytest.mark.parametrize("raw", [b"GARBAGE\r\n\r\n", b"GET\r\n\r\n"])
def test_a_request_that_is_not_a_request_is_a_400_and_not_a_traceback(tmp_path, raw):
    async def body(port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(raw)
        await writer.drain()
        assert (await reader.read(-1)).startswith(b"HTTP/1.1 400")
        writer.close()

    run(build(tmp_path), body)
