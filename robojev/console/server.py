"""`robojev console`: the web app and the live socket, on one loopback port, from the stdlib.

Three things are served and they are deliberately the same three a visitor to the GitHub Pages
deployment gets, plus one:

* the built front end out of `web/dist`, so `robojev console` on its own opens a usable page;
* the recorded bundles out of the replays directory, byte ranges and all -- a browser seeking an
  mp4 asks for a range, and a server that answers the whole file to one makes seeking silently
  unreliable;
* `GET /frame/<camera>.png`, the live renders (`images.py`);
* `GET /ws`, the protocol in `web/PROTOCOL.md`.

**No dependency.** `asyncio` has the sockets, `ws.py` has RFC 6455 and `images.py` has the PNG
encoder, so the console adds nothing to an install that is numpy and the standard library. A
WebSocket library would be a better WebSocket library than `ws.py`; it would also be the first
thing standing between a reader and `pip install -e .`.

**One episode at a time, one client at a time.** The console holds a simulator -- and under
`--policy model` a GPU -- so a second browser is told so and disconnected rather than quietly
shown a session it cannot drive. The page is a local tool on `127.0.0.1` with no authentication,
which is a deliberate limit and is why the default host is not `0.0.0.0`.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import time
import urllib.parse

from robojev import episode as episode_mod
from robojev.console import wire, ws
from robojev.console.images import FrameBuffer
from robojev.console.session import IDLE_TIMEOUT_SECONDS, Refused, Session

#: What the static half answers with, by extension. The list is short because the built app is
#: short: a page, a script, a stylesheet, the bundles and their videos.
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".mp4": "video/mp4",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".woff2": "font/woff2",
}

#: How long a request line and its headers may be. A local console reads commands over the socket,
#: not uploads over the request.
MAX_HEADER_BYTES = 32 * 1024

STATUS_TEXT = {200: "OK", 204: "No Content", 206: "Partial Content", 304: "Not Modified",
               400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed",
               416: "Range Not Satisfiable", 500: "Internal Server Error",
               503: "Service Unavailable"}


class Request:
    """One parsed HTTP request line plus headers. No body: nothing here takes one."""

    __slots__ = ("method", "target", "path", "query", "version", "headers")

    def __init__(self, method: str, target: str, version: str, headers: dict):
        self.method = method
        self.target = target
        self.version = version
        self.headers = headers
        split = urllib.parse.urlsplit(target)
        self.path = urllib.parse.unquote(split.path)
        self.query = urllib.parse.parse_qs(split.query)

    @property
    def keep_alive(self) -> bool:
        connection = (self.headers.get("connection") or "").lower()
        if self.version == "HTTP/1.0":
            return "keep-alive" in connection
        return "close" not in connection


async def read_request(reader: asyncio.StreamReader) -> Request | None:
    """The next request off a keep-alive connection, or None when the client has gone."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, ConnectionResetError):
        return None
    except asyncio.LimitOverrunError:
        raise ValueError("the request headers are longer than this console reads") from None
    if len(head) > MAX_HEADER_BYTES:
        raise ValueError("the request headers are longer than this console reads")
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3:
        raise ValueError(f"not an HTTP request line: {lines[0]!r}")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return Request(parts[0].upper(), parts[1], parts[2], headers)


def response_head(status: int, headers: dict) -> bytes:
    lines = [f"HTTP/1.1 {status} {STATUS_TEXT.get(status, 'OK')}"]
    lines += [f"{name}: {value}" for name, value in headers.items()]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def content_type(path: pathlib.Path) -> str:
    return TYPES.get(path.suffix.lower(), "application/octet-stream")


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """`bytes=<a>-<b>` against a file of `size`, or None for no (or an unusable) range."""
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes="):].split(",")[0].strip()
    first, _, last = spec.partition("-")
    try:
        if first == "":
            if last == "":
                return None
            start, end = max(0, size - int(last)), size - 1
        else:
            start = int(first)
            end = size - 1 if last == "" else min(int(last), size - 1)
    except ValueError:
        return None
    if start >= size or start > end:
        return (-1, -1)                                  # unsatisfiable, which is a 416 and not a 200
    return start, end


def inject(html: bytes, ws_url: str) -> bytes:
    """Tell the page it is being served by a console.

    One global, written before the bundle runs. The alternative -- letting the page probe its own
    origin for a console -- costs a request and a console error on every GitHub Pages visit, which
    is the one thing the static deployment must not have. A page that finds no global is a replay
    viewer and makes no request it did not mean to.
    """
    tag = (f'<script>window.__ROBOJEV_CONSOLE__={wire.dumps({"ws": ws_url})};</script>'
           ).encode("utf-8")
    marker = b"</head>"
    at = html.find(marker)
    return html[:at] + tag + html[at:] if at != -1 else tag + html


class Console:
    """The server. One instance owns the port, the frame buffer and the session."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 8765,
                 dist: pathlib.Path | None = None, replays: pathlib.Path,
                 policies: tuple[str, ...] = ("expert",), default: wire.StartSpec | None = None,
                 render_size: int = 256, idle_timeout: float = IDLE_TIMEOUT_SECONDS,
                 env_factory=None, build_policy=None, log=print):
        self.host = host
        self.port = int(port)
        self.dist = None if dist is None else pathlib.Path(dist)
        self.replays = pathlib.Path(replays)
        self.policies = tuple(policies)
        self.default = default or wire.StartSpec()
        self.log = log
        self.frames = FrameBuffer()
        self.session = Session(
            self._emit, frames=self.frames, replays_dir=self.replays,
            base_url=f"http://{host}:{port}", policies=self.policies, render_size=render_size,
            idle_timeout=idle_timeout, env_factory=env_factory, build_policy=build_policy)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._out: asyncio.Queue | None = None
        self._client: object | None = None
        self._tasks_cache: dict[str, list] = {}

    # -- the socket's outbound half -----------------------------------------------------------

    def _emit(self, message: dict) -> None:
        """Called from the episode thread. Hops to the loop and never blocks it."""
        loop, out = self._loop, self._out
        if loop is None or out is None:
            return
        try:
            loop.call_soon_threadsafe(out.put_nowait, message)
        except RuntimeError:                                       # pragma: no cover - shutting down
            pass

    async def _sender(self, writer: asyncio.StreamWriter, out: asyncio.Queue) -> None:
        while True:
            message = await out.get()
            if message is None:
                return
            writer.write(ws.encode_text(wire.dumps(message)))
            await writer.drain()

    # -- serving ------------------------------------------------------------------------------

    async def serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        server = await asyncio.start_server(self._connection, self.host, self.port)
        if self.port == 0:
            # `--port 0` is "any free port", which is what a test wants and what a second console
            # on a busy box wants. The bound one has to be read back, because the session builds
            # absolute picture URLs out of it.
            self.port = int(server.sockets[0].getsockname()[1])
            self.session.base_url = f"http://{self.host}:{self.port}"
        where = f"http://{self.host}:{self.port}/"
        self.log(f"robojev console: {where}")
        self.log(f"robojev console: app {self.dist if self.dist else '(not built: run `cd web && npm ci && npm run build`)'}")
        self.log(f"robojev console: replays {self.replays}")
        self.log(f"robojev console: policies {', '.join(self.policies)}")
        async with server:
            await server.serve_forever()

    async def _connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                try:
                    request = await read_request(reader)
                except ValueError as exc:
                    await self._send(writer, 400, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                    return
                if request is None:
                    return
                if request.path == "/ws" and ws.is_upgrade(request.headers):
                    await self._websocket(request, reader, writer)
                    return
                await self._http(request, writer)
                if not request.keep_alive:
                    return
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        finally:
            try:
                writer.close()
            except Exception:                                      # pragma: no cover
                pass

    # -- the static and picture halves ---------------------------------------------------------

    async def _http(self, request: Request, writer: asyncio.StreamWriter) -> None:
        if request.method not in ("GET", "HEAD"):
            await self._send(writer, 405, b"this console answers GET\n", "text/plain; charset=utf-8")
            return
        path = request.path
        if path.startswith("/frame/") and path.endswith(".png"):
            await self._frame(request, writer, path[len("/frame/"):-len(".png")])
            return
        await self._static(request, writer)

    async def _frame(self, request: Request, writer: asyncio.StreamWriter, camera: str) -> None:
        got = self.frames.png(camera)
        if got is None:
            # 204 rather than 404: there is no picture *yet*, which is an ordinary state before an
            # episode has been started, and a page polling for one must not fill a console with
            # red lines while it waits.
            await self._send(writer, 204, b"", "image/png", request=request)
            return
        seq, data = got
        await self._send(writer, 200, data, "image/png", request=request,
                         extra={"x-frame-seq": str(seq), "cache-control": "no-store"})

    async def _static(self, request: Request, writer: asyncio.StreamWriter) -> None:
        file = self._resolve(request.path)
        if file is None:
            await self._send(writer, 404, b"not found\n", "text/plain; charset=utf-8", request=request)
            return
        if file.name == "index.html":
            host = request.headers.get("host") or f"{self.host}:{self.port}"
            body = inject(file.read_bytes(), f"ws://{host}/ws")
            await self._send(writer, 200, body, "text/html; charset=utf-8", request=request,
                             extra={"cache-control": "no-store"})
            return
        size = file.stat().st_size
        wanted = parse_range(request.headers.get("range"), size)
        if wanted == (-1, -1):
            await self._send(writer, 416, b"", "text/plain; charset=utf-8", request=request,
                             extra={"content-range": f"bytes */{size}"})
            return
        type_ = content_type(file)
        if wanted is None:
            await self._send_file(writer, request, file, 200, 0, size - 1, size, type_)
            return
        start, end = wanted
        await self._send_file(writer, request, file, 206, start, end, size, type_)

    def _resolve(self, path: str) -> pathlib.Path | None:
        """A URL path to a file on disk, or None.

        `/replays/...` is served out of the replays directory first, so an episode saved a second
        ago is on the page after a refresh rather than after a rebuild; anything it does not have
        falls through to the built app's own copy.
        """
        clean = pathlib.PurePosixPath(path).as_posix().lstrip("/")
        if clean == "" or clean.endswith("/"):
            clean += "index.html"
        parts = [p for p in clean.split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            return None
        roots: list[pathlib.Path] = []
        if parts and parts[0] == "replays":
            roots.append(self.replays)
            parts = parts[1:]
            if self.dist is not None:
                roots.append(self.dist / "replays")
        elif self.dist is not None:
            roots.append(self.dist)
        for root in roots:
            candidate = root.joinpath(*parts) if parts else root / "index.html"
            try:
                resolved = candidate.resolve()
                resolved.relative_to(root.resolve())
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                return resolved
        return None

    async def _send(self, writer: asyncio.StreamWriter, status: int, body: bytes, type_: str,
                    *, request: Request | None = None, extra: dict | None = None) -> None:
        headers = {"content-type": type_, "content-length": str(len(body)),
                   "connection": "keep-alive" if (request is None or request.keep_alive) else "close"}
        headers.update(extra or {})
        writer.write(response_head(status, headers))
        if request is None or request.method != "HEAD":
            writer.write(body)
        await writer.drain()

    async def _send_file(self, writer: asyncio.StreamWriter, request: Request,
                         file: pathlib.Path, status: int, start: int, end: int, size: int,
                         type_: str) -> None:
        length = end - start + 1
        headers = {"content-type": type_, "content-length": str(length), "accept-ranges": "bytes",
                   "connection": "keep-alive" if request.keep_alive else "close"}
        if status == 206:
            headers["content-range"] = f"bytes {start}-{end}/{size}"
        writer.write(response_head(status, headers))
        if request.method == "HEAD":
            await writer.drain()
            return
        with file.open("rb") as handle:
            handle.seek(start)
            left = length
            while left > 0:
                block = handle.read(min(1 << 16, left))
                if not block:
                    break
                left -= len(block)
                writer.write(block)
                await writer.drain()

    # -- the socket ----------------------------------------------------------------------------

    async def _websocket(self, request: Request, reader: asyncio.StreamReader,
                         writer: asyncio.StreamWriter) -> None:
        try:
            writer.write(ws.handshake(request.headers))
            await writer.drain()
        except ws.WebSocketError as exc:
            await self._send(writer, 400, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
            return
        if self._client is not None:
            # A second browser. Told so in the protocol's own words and then closed, because the
            # alternative is two pages driving one simulator.
            writer.write(ws.encode_text(wire.dumps(wire.error(
                "this console is already open in another window. One episode at a time: close the "
                "other tab, or reload this one once it has gone.", fatal=True))))
            writer.write(ws.close_frame(ws.CLOSE_NORMAL, "already connected"))
            await writer.drain()
            return
        self._client = writer
        out: asyncio.Queue = asyncio.Queue()
        self._out = out
        sender = asyncio.create_task(self._sender(writer, out))
        try:
            await self._greet(out)
            await self._read_socket(reader, writer, out)
        finally:
            sender.cancel()
            self._out = None
            self._client = None

    async def _greet(self, out: asyncio.Queue) -> None:
        out.put_nowait(wire.config(
            policies=self.policies,
            suites=sorted(episode_mod.UPSTREAM_MAX_STEPS),
            default=self.default,
            video=self.session.video(),
            state=self.session.state,
            replays="replays/",
        ))
        header = self.session.header()
        if header is not None:
            out.put_nowait(wire.hello(header, self.session.video()))
        out.put_nowait(self.session.snapshot())

    async def _read_socket(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                           out: asyncio.Queue) -> None:
        frames = ws.Reader()
        while True:
            try:
                data = await reader.read(65536)
            except (ConnectionResetError, BrokenPipeError):
                return
            if not data:
                return
            try:
                messages = frames.feed(data)
            except ws.WebSocketError as exc:
                writer.write(ws.close_frame(exc.code, str(exc)[:110]))
                await writer.drain()
                return
            for opcode, payload in messages:
                if opcode == ws.OP_CLOSE:
                    writer.write(ws.close_frame())
                    await writer.drain()
                    return
                if opcode == ws.OP_PING:
                    writer.write(ws.encode(payload, ws.OP_PONG))
                    await writer.drain()
                    continue
                if opcode != ws.OP_TEXT:
                    continue
                await self._command(payload.decode("utf-8", "replace"), out)

    async def _command(self, raw: str, out: asyncio.Queue) -> None:
        try:
            command = wire.parse_command(raw, policies=self.policies)
        except wire.CommandError as exc:
            out.put_nowait(wire.error(str(exc)))
            return
        if command.op == "ping":
            out.put_nowait(wire.pong())
            return
        if command.op == "tasks":
            await self._tasks(command.suite or self.default.suite, out)
            return
        try:
            self.session.submit(command)
        except Refused as exc:
            out.put_nowait(wire.error(str(exc)))

    async def _tasks(self, suite: str, out: asyncio.Queue) -> None:
        """One suite's task sentences, read off the simulator's own task definitions.

        Off the main thread and cached: it imports LIBERO, which costs seconds the first time and
        would otherwise be paid on the connect of every reload. A console with no simulator answers
        with the reason and keeps numbering the tasks.
        """
        cached = self._tasks_cache.get(suite)
        if cached is not None:
            out.put_nowait(wire.tasks(suite, cached))
            return
        try:
            rows = await asyncio.get_running_loop().run_in_executor(None, suite_tasks, suite)
        except Exception as exc:                                   # noqa: BLE001 - shown verbatim
            out.put_nowait(wire.error(
                f"the task list for {suite!r} is unavailable ({type(exc).__name__}: {exc}). The "
                f"pickers stay numeric; the console still runs."))
            return
        self._tasks_cache[suite] = rows
        out.put_nowait(wire.tasks(suite, rows))


def suite_tasks(suite: str) -> list[dict]:
    """`[{index, instruction, init_states}]` for one LIBERO suite. Imports the simulator."""
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    if suite not in suites:
        raise KeyError(f"no LIBERO suite {suite!r}; have {sorted(suites)}")
    bench = suites[suite]()
    count = int(getattr(bench, "n_tasks", 0) or len(getattr(bench, "tasks", [])))
    rows = []
    for index in range(count):
        try:
            states = len(bench.get_task_init_states(index))
        except Exception:                                          # pragma: no cover - LIBERO's
            states = 0
        rows.append({"index": index, "instruction": bench.get_task(index).language,
                     "init_states": states})
    return rows


def available_policies(explicit: str | None = None) -> tuple[str, ...]:
    """Which engines this interpreter can actually serve.

    `expert` always: it is numpy. `model` only with torch on the path, because the failure without
    it is a two-minute import error in the middle of an episode. `jev` only with a key, because
    every decision it answers is a paid request and offering it without one would fail at the same
    place.
    """
    import importlib.util
    import os

    found = ["expert"]
    try:
        if importlib.util.find_spec("torch") is not None:
            found.append("model")
    except (ImportError, ValueError):                              # pragma: no cover - broken env
        pass
    if os.environ.get("JEV_API_KEY"):
        found.append("jev")
    if explicit is not None and explicit not in found:
        found.append(explicit)
    return tuple(found)


def default_dist(root: pathlib.Path | None = None) -> pathlib.Path | None:
    """`web/dist`, if this checkout has one built."""
    base = root or pathlib.Path(__file__).resolve().parents[2]
    dist = base / "web" / "dist"
    return dist if (dist / "index.html").is_file() else None


def default_replays(root: pathlib.Path | None = None) -> pathlib.Path:
    """`web/public/replays` in a checkout, or `./replays` anywhere else."""
    base = root or pathlib.Path(__file__).resolve().parents[2]
    public = base / "web" / "public" / "replays"
    return public if public.is_dir() else pathlib.Path.cwd() / "replays"


def serve(console: Console) -> int:
    """Run one console until Ctrl-C, and let go of the simulator on the way out."""
    started = time.time()
    try:
        asyncio.run(console.serve())
    except KeyboardInterrupt:
        print(f"robojev console: stopped after {time.time() - started:.0f}s", file=sys.stderr)
    finally:
        console.session.close()
    return 0


__all__ = ["Console", "MAX_HEADER_BYTES", "Request", "TYPES", "available_policies",
           "content_type", "default_dist", "default_replays", "inject", "parse_range",
           "read_request", "response_head", "serve", "suite_tasks"]
