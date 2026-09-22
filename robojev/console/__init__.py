"""The live console: one local server, one episode, one browser.

`robojev console` serves the same front end GitHub Pages serves and adds the half a static
deployment cannot have -- a simulator and a policy, stepped by hand, with every decision on screen
as it is made. The contract is `web/PROTOCOL.md`, and its one load-bearing sentence is that **a
live `decision` message is byte-for-byte one entry of a bundle's `decisions` array**: the page has
one reader for a recorded decision and a streamed one, and a live episode can therefore be saved
as an ordinary bundle with `recorder.record` and nothing fixed up.

Four modules, each with one job:

* `wire` -- every message, built and parsed. No socket, no simulator: the protocol is testable on
  a laptop.
* `ws` -- RFC 6455's server half, because the standard library has no WebSocket and a dependency
  here would be the first thing between a reader and `pip install -e .`.
* `images` -- the camera renders as PNGs, encoded with `zlib` for the same reason.
* `session` -- one episode, driven by commands instead of by a clock. A deliberate mirror of
  `episode.run_episode`, checked against it rather than refactored out of it.
* `server` -- the asyncio server that ties them to a port.

Nothing heavy is imported here: `import robojev.console` costs the standard library and numpy, and
the simulator is reached only when an episode is started.
"""
from __future__ import annotations

from robojev.console.images import FrameBuffer, encode_png
from robojev.console.server import Console, available_policies, default_dist, default_replays, serve
from robojev.console.session import Session, write_index
from robojev.console.wire import PROTOCOL_VERSION, Command, CommandError, StartSpec, parse_command

__all__ = ["Command", "CommandError", "Console", "FrameBuffer", "PROTOCOL_VERSION", "Session",
           "StartSpec", "available_policies", "default_dist", "default_replays", "encode_png",
           "parse_command", "serve", "write_index"]
