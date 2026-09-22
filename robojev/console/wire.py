"""Every message that crosses the console's socket, built and parsed in one file.

`web/PROTOCOL.md` is the prose version of this module and the two must not drift, so the rule is
the same one the bundle format keeps: **a `decision` message is byte-for-byte one entry of a
bundle's `decisions` array** (`recorder.decision_entry`), and every difference between a live view
and a replay lives in the envelope around it.

Nothing here touches a simulator, a policy or a socket. That is what makes the protocol testable
on a laptop: `parse_command` turns a client's JSON into a validated `Command` or refuses it with
the sentence the client will be shown, and the builders below turn server facts into the exact
objects the page reads.
"""
from __future__ import annotations

import dataclasses
import json

#: The envelope's version. Bumped when a *message* changes shape; the bundle inside a `decision`
#: carries `recorder.SCHEMA_VERSION` separately, because the two can move independently.
PROTOCOL_VERSION: int = 1

#: What `status.state` can say.
#:
#: `idle` -- no episode. `starting` -- building the simulator and the policy, which is the slow
#: one (LIBERO takes seconds). `waiting` -- the settling steps. `paused` -- an episode in hand,
#: stopped, waiting for a command. `running` -- stepping on its own. `done` -- over, and still
#: holding the frames so it can be saved. `error` -- over, and why.
STATES: tuple[str, ...] = ("idle", "starting", "waiting", "paused", "running", "done", "error")

#: The ops a client may send.
OPS: tuple[str, ...] = ("start", "step", "run", "pause", "reset", "override", "save", "ping",
                        "tasks")


class CommandError(ValueError):
    """A client message this server will not act on. The text is shown to the operator."""


@dataclasses.dataclass(frozen=True)
class StartSpec:
    """What one episode is: a scene, a policy, and how the policy picks its answers."""

    suite: str = "libero_spatial"
    task: int = 0
    init: int = 0
    policy: str = "expert"
    #: `argmax` or `sample@<T>`; `robojev.policy.parse_selection` is the authority on the spelling.
    selection: str = "argmax"
    checkpoint: str | None = None
    seed: int = 7
    max_steps: int | None = None

    def as_json(self) -> dict:
        return {"suite": self.suite, "task": self.task, "init": self.init, "policy": self.policy,
                "selection": self.selection, "checkpoint": self.checkpoint, "seed": self.seed,
                "max_steps": self.max_steps}


@dataclasses.dataclass(frozen=True)
class Command:
    """One validated client message."""

    op: str
    spec: StartSpec | None = None
    qid: str | None = None
    #: The candidate to force, or None to take an armed override back off.
    candidate: str | None = None
    name: str | None = None
    suite: str | None = None
    #: Why the session made this command up, when it is not the client's: the idle timeout writes
    #: a `reset` for an operator who has gone, and the record has to say which it was.
    reason: str | None = None


# --------------------------------------------------------------------------- client -> server

def _int(payload: dict, key: str, default: int, *, low: int = 0, high: int | None = None) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise CommandError(f"{key}: expected a number, got {value!r}")
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise CommandError(f"{key}: {value!r} is not a whole number") from None
    if out < low or (high is not None and out > high):
        raise CommandError(f"{key}: {out} is outside {low}..{'' if high is None else high}")
    return out


def _text(payload: dict, key: str, default: str | None) -> str | None:
    value = payload.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CommandError(f"{key}: expected a string, got {value!r}")
    value = value.strip()
    if len(value) > 512:
        raise CommandError(f"{key}: too long")
    return value or None


def parse_start(payload: dict, *, policies: tuple[str, ...]) -> StartSpec:
    """A `start` message's body as a `StartSpec`, or a refusal naming the field.

    The selection string is *not* parsed here: `robojev.policy.parse_selection` is the one place
    that knows what `sample@0.7` means, and a second parser would be a second opinion. A client
    may also send `temperature` beside a bare `sample`, which is spelled out into the canonical
    form so the record and the socket agree on one spelling.
    """
    suite = _text(payload, "suite", "libero_spatial") or "libero_spatial"
    policy = _text(payload, "policy", "expert") or "expert"
    if policy not in policies:
        raise CommandError(f"policy: this console serves {list(policies)}, not {policy!r}")
    selection = _text(payload, "selection", "argmax") or "argmax"
    temperature = payload.get("temperature")
    if selection == "sample" and temperature is not None:
        try:
            selection = f"sample@{float(temperature):g}"
        except (TypeError, ValueError):
            raise CommandError(f"temperature: {temperature!r} is not a number") from None
    return StartSpec(
        suite=suite,
        task=_int(payload, "task", 0, high=999),
        init=_int(payload, "init", 0, high=9999),
        policy=policy,
        selection=selection,
        checkpoint=_text(payload, "checkpoint", None),
        seed=_int(payload, "seed", 7, low=0, high=2 ** 31 - 1),
        max_steps=None if payload.get("max_steps") is None else _int(payload, "max_steps", 0, low=1,
                                                                     high=100000),
    )


def parse_command(raw: str, *, policies: tuple[str, ...] = ("expert",)) -> Command:
    """One client frame as a `Command`. Raises `CommandError` with the sentence to send back."""
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise CommandError(f"not JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise CommandError("expected a JSON object with a `type`")
    op = payload.get("type")
    if not isinstance(op, str) or op not in OPS:
        raise CommandError(f"unknown message type {op!r}; this console reads {list(OPS)}")
    if op == "start":
        return Command(op, spec=parse_start(payload, policies=policies))
    if op == "override":
        qid = _text(payload, "qid", None)
        if not qid:
            raise CommandError("override: which question? `qid` is required")
        # `null` is how an armed override is taken back off; there is no "no override" candidate to
        # name instead, and a client that had to send one would be forcing an answer to un-force it.
        return Command(op, qid=qid, candidate=_text(payload, "candidate", None))
    if op == "save":
        return Command(op, name=_text(payload, "name", None))
    if op == "tasks":
        return Command(op, suite=_text(payload, "suite", "libero_spatial"))
    return Command(op)


# --------------------------------------------------------------------------- server -> client

def config(*, policies, suites, default: StartSpec, video: dict, state: str,
           replays: str | None) -> dict:
    """Sent once, on connect, before any episode: what this console can be asked for.

    It is not in the bundle format and never will be -- a recording has no pickers -- so it is the
    one message with no replay counterpart.
    """
    return {
        "type": "config",
        "protocol": PROTOCOL_VERSION,
        "policies": list(policies),
        "suites": list(suites),
        "default": default.as_json(),
        "video": video,
        "state": state,
        "replays": replays,
    }


def hello(episode: dict, video: dict) -> dict:
    """The episode header: the bundle's top-level fields, minus the ones only an ended episode has."""
    return {"type": "hello", "schema_version": episode.get("schema_version", 1),
            "episode": episode, "video": video}


def decision(entry: dict) -> dict:
    """One decision. `entry` is `recorder.decision_entry`'s output and is passed through whole."""
    return {"type": "decision", "decision": entry}


def status(state: str, *, step: int | None = None, decisions: int | None = None,
           message: str | None = None, overrides: dict | None = None,
           episode: bool = False) -> dict:
    """Where the run is. `overrides` is the authoritative armed set -- the page draws from this and
    never from its own memory of what it clicked."""
    if state not in STATES:                                       # pragma: no cover - programmer error
        raise ValueError(f"unknown state {state!r}")
    return {"type": "status", "state": state, "step": step, "decisions": decisions,
            "message": message, "overrides": dict(overrides or {}), "episode": bool(episode)}


def done(*, success: bool, terminated_by: str, steps: int, decisions: int,
         error: str | None = None) -> dict:
    return {"type": "done", "success": bool(success), "terminated_by": terminated_by,
            "steps": int(steps), "decisions": int(decisions), "error": error}


def error(message: str, *, fatal: bool = False) -> dict:
    return {"type": "error", "message": str(message), "fatal": bool(fatal)}


def saved(*, id: str, path: str, url: str | None, decisions: int, bytes_written: int,
          success: bool) -> dict:
    """A live episode written out as a bundle -- the same directory `robojev record` writes."""
    return {"type": "saved", "id": id, "path": path, "url": url, "decisions": int(decisions),
            "bytes": int(bytes_written), "success": bool(success)}


def tasks(suite: str, rows) -> dict:
    """One suite's tasks, so the picker can name them instead of numbering them."""
    return {"type": "tasks", "suite": suite, "tasks": list(rows)}


def pong() -> dict:
    return {"type": "pong"}


def dumps(message: dict) -> str:
    """One message on the wire. Compact, and never with NaN in it: `NaN` is not JSON and
    `JSON.parse` refuses it, so a policy that produced one has to be visible here rather than
    three layers away in a browser."""
    return json.dumps(message, allow_nan=False, separators=(",", ":"))


__all__ = ["OPS", "PROTOCOL_VERSION", "STATES", "Command", "CommandError", "StartSpec", "config",
           "decision", "done", "dumps", "error", "hello", "parse_command", "parse_start", "pong",
           "saved", "status", "tasks"]
