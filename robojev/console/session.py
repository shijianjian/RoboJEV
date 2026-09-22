"""One live episode, driven by hand: `run_episode` with the clock taken out.

The same environment, the same policy, the same `Frame`s and the same `recorder` -- what differs is
only *when* a decision is taken. Each one waits for a command from the browser, the operator may
force one question's answer on it, and every control step's renders are published as they are
produced so the page can show the arm moving rather than a still.

**The loop below is a mirror of `episode.run_episode`, not a refactor of it.** That loop is a
statement about how a LIBERO evaluation is run -- wait steps, one chunk per decision, stop on the
task's own success predicate, one terminal frame -- and a number measured with it is comparable
with a published one only for as long as it stays readable against upstream's. A session is not
that loop; it is a human driving the same machinery. So the mechanical parts are shared
(`_policy_obs`, `_frame`, `recorder.decision_entry`) and the loop itself is written twice, on
purpose. The property that matters is checked rather than assumed: `tests/test_console_session.py`
runs the same trajectory through both and compares the bundles byte for byte.

**One thread, and why.** The episode runs on a worker thread and the socket on the event loop.
Stepping a simulator takes tens of milliseconds and encoding a decision takes none, so a loop that
did both would stutter the socket; and a `run` that stepped only when a poll came back would run at
the network's pace rather than the machine's. Commands cross on a `queue.Queue` -- blocking on it
while paused, checked without blocking between decisions while running, which is what makes a
`Pause` land within one decision.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import queue
import re
import threading
import time

import numpy as np

from robojev import episode as episode_mod
from robojev import recorder
from robojev.console import wire

#: How long a blocking wait sits in `queue.Queue.get` before it looks up to check the clock.
#: Nothing depends on the value: it is how often the idle timeout is *noticed*, not what it is.
TICK_SECONDS: float = 0.25

#: How long a session with nobody driving it is kept alive. A console holds a simulator and, under
#: `--policy model`, a GPU; a tab left open over lunch must not hold either.
IDLE_TIMEOUT_SECONDS: float = 900.0

#: Characters allowed in a saved bundle's directory name. Everything else becomes `-`: the name
#: comes from a browser and is used as a path.
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

#: The fields of `index.json` a strip item is made of, in the order `tools/install_bundles.py`
#: writes them -- so a console that saves an episode rewrites the file the same way the installer
#: would, and a repository with no new bundle in it sees no diff.
INDEX_FIELDS = ("id", "title", "instruction", "note", "success", "decisions", "max_decisions",
                "task_index", "init_state_index", "suite", "poster")


def safe_name(name: str | None, fallback: str) -> str:
    cleaned = SAFE_NAME.sub("-", (name or "").strip()).strip("-.")
    return cleaned or fallback


def write_index(replays_dir: pathlib.Path) -> list[dict]:
    """Rebuild `index.json` from the bundles that are actually on disk, and return it.

    Derived, never edited: a strip item is a projection of the bundle beside it, and the one way
    for the two to disagree is for something to write the second one by hand. The order already in
    the file is kept -- it is curated, and a save must not reshuffle the demo -- with anything new
    appended in name order.
    """
    path = replays_dir / "index.json"
    order: list[str] = []
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            order = [row["id"] for row in previous if isinstance(row, dict) and "id" in row]
        except (ValueError, TypeError, KeyError):
            order = []
    entries = []
    for directory in sorted(p for p in replays_dir.iterdir() if p.is_dir()):
        manifest = directory / "episode.json"
        if not manifest.is_file():
            continue
        bundle = json.loads(manifest.read_text(encoding="utf-8"))
        row = {key: bundle.get(key) for key in INDEX_FIELDS}
        row["id"] = bundle.get("id") or directory.name
        row["decisions"] = len(bundle.get("decisions") or [])
        entries.append(row)
    entries.sort(key=lambda row: (order.index(row["id"]) if row["id"] in order else len(order),
                                  row["id"]))
    path.write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8")
    return entries


class Refused(Exception):
    """A command the session will not act on in the state it is in."""


@dataclasses.dataclass
class _Run:
    """Everything one episode holds while it is in hand."""

    spec: wire.StartSpec
    policy: object
    env: object
    protocol: episode_mod.Protocol
    episode_id: str
    started: float
    #: The `hello` payload, built once at setup. Kept rather than rebuilt because `describe()` on
    #: the hosted engine costs a request, and a second client connecting mid-episode must not.
    header: dict | None = None
    frames: list = dataclasses.field(default_factory=list)
    decisions: int = 0
    steps: int = 0
    t: int = 0
    success: bool = False
    first_success_step: int | None = None
    terminated_by: str = "max_steps"
    error: str | None = None
    obs: dict | None = None
    result: episode_mod.EpisodeResult | None = None
    #: The operator reset it (or the idle timeout did). An episode that ended *itself* is held --
    #: the frames are still there and `save` still works -- but one that was told to stop is let
    #: go of at once, because the next thing the operator does is start another.
    closed: bool = False


class Session:
    """The console's one episode at a time, and the state machine around it.

    `emit` is called from the worker thread with one protocol message; the server's job is to get
    it onto the socket. Everything else here -- the environment, the policy, the frames, the
    armed overrides -- belongs to the worker and is never touched from the event loop.
    """

    def __init__(self, emit, *, frames, replays_dir: pathlib.Path, base_url: str,
                 policies: tuple[str, ...] = ("expert",), render_size: int = 256,
                 idle_timeout: float = IDLE_TIMEOUT_SECONDS, env_factory=None, build_policy=None,
                 clock=time.monotonic, note: str = ""):
        self._emit = emit
        self._frames = frames
        self.replays_dir = pathlib.Path(replays_dir)
        self.base_url = base_url.rstrip("/")
        self.policies = tuple(policies)
        self.render_size = int(render_size)
        self.idle_timeout = float(idle_timeout)
        self._env_factory = env_factory
        self._build_policy = build_policy or _default_policy
        self._clock = clock
        self.note = note

        self._commands: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._state = "idle"
        self._thread: threading.Thread | None = None
        self._closing = False
        self._overrides: dict[str, str] = {}
        self._run: _Run | None = None
        self._touched = clock()

    # -- what the event loop may ask ---------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def busy(self) -> bool:
        return self.state != "idle"

    def video(self) -> dict:
        """Where the page fetches the two camera pictures from.

        Absolute, because the page may be a static build served from somewhere else entirely and
        pointed here with `?live=`; relative URLs would resolve against that other origin.
        """
        return {
            "kind": "poll",
            "cameras": {"agentview": f"{self.base_url}/frame/agentview.png",
                        "wrist": f"{self.base_url}/frame/wrist.png"},
            "rate": episode_mod.CONTROL_FPS,
        }

    def submit(self, command: wire.Command) -> None:
        """Accept one command, or raise `Refused` with the sentence the operator is shown."""
        state = self.state
        if command.op == "start":
            if state != "idle":
                raise Refused(
                    "an episode is already open; reset it before starting another. This console "
                    "runs one simulator at a time.")
            self._begin(command.spec)
            return
        if state == "idle":
            if command.op == "reset":
                self._announce("idle")
                return
            raise Refused(f"{command.op}: there is no episode. Start one first.")
        self._touch()
        self._commands.put(command)

    def close(self) -> None:
        """Stop whatever is running and let go of the simulator. Called once, at shutdown."""
        self._closing = True
        self._commands.put(wire.Command("reset"))
        thread = self._thread
        if thread is not None:
            thread.join(timeout=30.0)

    # -- the worker --------------------------------------------------------------------------

    def _begin(self, spec: wire.StartSpec) -> None:
        self._commands = queue.Queue()
        self._overrides = {}
        self._touch()
        self._set_state("starting")
        self._announce("starting", message=f"building {spec.suite} task {spec.task}")
        self._thread = threading.Thread(target=self._work, args=(spec,), name="robojev-console",
                                        daemon=True)
        self._thread.start()

    def _work(self, spec: wire.StartSpec) -> None:
        run = None
        try:
            run = self._setup(spec)
        except BaseException as exc:                              # noqa: BLE001 - shown, not hidden
            self._frames.clear()
            self._set_state("idle")
            self._emit(wire.error(f"{type(exc).__name__}: {exc}"))
            self._announce("idle")
            return
        self._run = run
        try:
            self._wait_steps(run)
            self._loop(run)
        except BaseException as exc:                              # noqa: BLE001 - recorded
            run.terminated_by = "error"
            run.error = f"{type(exc).__name__}: {exc}"
        self._finalise(run)
        try:
            if not run.closed:
                self._after(run)
        finally:
            self._teardown(run)

    def _setup(self, spec: wire.StartSpec) -> _Run:
        policy = self._build_policy(spec)
        protocol = policy.protocol(spec.max_steps)
        factory = self._env_factory
        if factory is None:
            from robojev import envs

            factory = envs.make
        env = factory(spec.suite, spec.task, render_size=self.render_size,
                      env_seed=protocol.env_seed)
        try:
            self._seed(spec.seed)
            obs = env.reset(spec.init)
            policy.reset(env.instruction)
        except BaseException:
            env.close()
            policy.close()
            raise
        run = _Run(spec=spec, policy=policy, env=env, protocol=protocol,
                   episode_id=_episode_id(), started=time.time(), obs=obs)
        run.header = self._header(run)
        self._frames.clear()
        self._publish(env, obs)
        self._emit(wire.hello(run.header, self.video()))
        return run

    @staticmethod
    def _seed(seed: int) -> None:
        try:
            from robojev.envs.libero import set_seed_everywhere
        except ImportError:                                       # pragma: no cover - no simulator
            return
        set_seed_everywhere(seed)

    def _header(self, run: _Run) -> dict:
        """The episode header: exactly the bundle's top-level fields, minus the ones that only
        exist once an episode is over."""
        info = run.policy.describe()
        checkpoint = info.get("checkpoint") or {}
        fps = float(getattr(run.env, "control_freq", episode_mod.CONTROL_FPS))
        return {
            "schema_version": recorder.SCHEMA_VERSION,
            "id": run.episode_id,
            "suite": run.spec.suite,
            "task_index": run.spec.task,
            "init_state_index": run.spec.init,
            "instruction": run.env.instruction,
            "title": run.env.instruction,
            "note": self.note,
            "policy": run.spec.policy,
            "checkpoint_revision": checkpoint.get("revision"),
            "checkpoint_repo": checkpoint.get("repo"),
            "questions_version": info.get("questions_version"),
            "selection": run.spec.selection,
            "control_rate": fps,
            "wait_steps": int(run.protocol.wait_steps),
            "execute_steps": int(run.protocol.execute_steps),
            "max_steps": int(run.protocol.max_steps),
            "max_decisions": int(run.protocol.max_steps // run.protocol.execute_steps),
        }

    # -- the loop ----------------------------------------------------------------------------

    def _wait_steps(self, run: _Run) -> None:
        """The settling steps. They are the scene dropping into place, not something an operator
        decides, so they run before the first command -- and they are published, so the page
        watches the settling rather than jumping over it."""
        if run.protocol.wait_steps:
            self._set_state("waiting")
            self._announce("waiting")
        while run.t < run.protocol.wait_steps:
            action = np.asarray(run.env.dummy_action(), dtype=np.float32)
            self._one_step(run, action, policy_query=False, decisions=None, is_wait=True)
        self._set_state("paused")
        self._announce("paused", run=run)

    def _loop(self, run: _Run) -> None:
        horizon = run.protocol.max_steps + run.protocol.wait_steps
        running = False
        stop = False
        while not stop and run.t < horizon:
            command = self._next(block=not running)
            if command is not None:
                op = command.op
                if op == "reset":
                    run.terminated_by = command.reason or "reset"
                    run.closed = True
                    return
                if op == "pause":
                    running = False
                    self._set_state("paused")
                    self._announce("paused", run=run)
                    continue
                if op == "override":
                    self._arm(run, command)
                    continue
                if op == "save":
                    self._save(run, command.name)
                    continue
                if op == "run":
                    running = True
                    self._set_state("running")
                    self._announce("running", run=run)
                elif op == "step":
                    running = False
                    self._set_state("paused")
                else:
                    continue
            stop = self._decide(run, horizon)
            if not stop:
                self._announce(self.state, run=run)

    def _next(self, block: bool):
        """The next command, or None when nothing is waiting and `block` is false.

        Blocking is how a paused session waits, and it is also the only place the idle timeout can
        fire: a console nobody is driving is a simulator nobody is using.
        """
        while True:
            try:
                return self._commands.get(timeout=TICK_SECONDS) if block else self._commands.get_nowait()
            except queue.Empty:
                pass
            if self._closing:
                return wire.Command("reset")
            if not block:
                return None
            if self._clock() - self._touched > self.idle_timeout:
                self._emit(wire.error(
                    f"the episode was closed after {self.idle_timeout:.0f} s with no command: a "
                    f"console holds the simulator, so an idle one lets go of it."))
                return wire.Command("reset", reason="idle")

    def _arm(self, run: _Run, command: wire.Command) -> None:
        """Hold one question's answer for the next decision, or take a held one back off.

        Checked against the question set *now* rather than at the decision: an override naming a
        question this policy does not ask, or a candidate that question does not have, is a typo,
        and a typo that is only discovered a decision later looks like the model disagreeing with
        the operator.
        """
        from robojev.policy import PolicyError
        from robojev.questions import candidates as v2_candidates

        qid = command.qid or ""
        if command.candidate is None:
            self._overrides.pop(qid, None)
            self._announce(self.state, run=run)
            return
        try:
            options = list(v2_candidates(qid))
        except (KeyError, PolicyError, ValueError):
            options = []
        if not options:
            self._emit(wire.error(f"override {qid!r}: this question set has no such question"))
            return
        if command.candidate not in options:
            self._emit(wire.error(
                f"override {qid!r}: {command.candidate!r} is not one of {options}"))
            return
        self._overrides[qid] = command.candidate
        self._announce(self.state, run=run)

    def _decide(self, run: _Run, horizon: int) -> bool:
        """One decision and the chunk it is executed for. True when the episode ended on it."""
        overrides = dict(self._overrides)
        self._overrides = {}
        payload = episode_mod._policy_obs(run.env, run.obs, bool(run.protocol.privileged_state))
        chunk, decisions = run.policy.act(payload, overrides=overrides or None,
                                          selection=run.spec.selection)
        chunk = np.asarray(chunk, dtype=np.float32)
        execute = int(run.protocol.execute_steps)
        if chunk.shape[0] < execute:
            raise ValueError(
                f"policy returned {chunk.shape[0]} actions, fewer than execute_steps={execute}")
        query = None
        stop = False
        for i, row in enumerate(chunk[:execute]):
            if run.t >= horizon:
                break
            frame = self._one_step(run, np.asarray(row, dtype=np.float32),
                                   policy_query=(i == 0), decisions=decisions, is_wait=False)
            if i == 0:
                query = frame
            if run.success:
                stop = True
                break
        if query is not None:
            fps = float(getattr(run.env, "control_freq", episode_mod.CONTROL_FPS))
            self._emit(wire.decision(recorder.decision_entry(query, run.decisions, fps)))
            run.decisions += 1
        return stop

    def _one_step(self, run: _Run, action, *, policy_query: bool, decisions, is_wait: bool):
        """Build one frame, step the environment, and do exactly the bookkeeping `run_episode`
        does with the result."""
        frame = episode_mod._frame(
            run.env, run.obs, run.t, action, images=True, is_wait_step=is_wait,
            next_success=False, is_terminal=False, policy_query=policy_query, decisions=decisions)
        result = run.env.step(action.tolist())
        frame.next_success = bool(result.done)
        run.frames.append(frame)
        self._frames.publish(frame.images)
        run.obs = result.obs
        if result.done and run.first_success_step is None:
            run.first_success_step = run.t
        if not is_wait:
            run.steps += 1
        if result.done and getattr(run.env, "stop_on_success", True):
            run.success = True
            run.terminated_by = "success"
            return frame
        run.t += 1
        return frame

    def _finalise(self, run: _Run) -> None:
        """The terminal frame, the result, and `done` -- `run_episode`'s own ending."""
        if run.terminated_by != "error" and run.frames:
            run.frames.append(episode_mod._frame(
                run.env, run.obs, run.t + 1,
                np.full(len(run.frames[-1].action), np.nan, dtype=np.float32), images=True,
                is_wait_step=False, next_success=run.frames[-1].next_success, is_terminal=True,
                policy_query=False, decisions=None))
        run.result = episode_mod.EpisodeResult(
            frames=run.frames, success=run.success, steps=run.steps,
            terminated_by=run.terminated_by, error=run.error,
            first_success_step=run.first_success_step)
        self._set_state("error" if run.error else "done")
        if run.error:
            self._emit(wire.error(run.error))
        self._emit(wire.done(success=run.success, terminated_by=run.terminated_by,
                             steps=run.steps, decisions=run.decisions, error=run.error))
        self._announce(self.state, run=run)

    def _after(self, run: _Run) -> None:
        """The episode is over and the frames are still here: the operator can save it, or reset.

        The simulator is still open, because `recorder.record` reads the task sentence and the
        control rate off it -- so this is also where the idle timeout does its real work.
        """
        while not self._closing:
            command = self._next(block=True)
            if command is None or command.op == "reset":
                return
            if command.op == "save":
                self._save(run, command.name)
                continue
            if command.op in ("step", "run", "pause", "override"):
                self._emit(wire.error(
                    f"{command.op}: this episode has ended ({run.terminated_by}). Reset and start "
                    f"another."))

    def _teardown(self, run: _Run) -> None:
        try:
            run.env.close()
        except Exception:                                          # pragma: no cover - adapter's
            pass
        try:
            run.policy.close()
        except Exception:                                          # pragma: no cover - adapter's
            pass
        self._run = None
        self._overrides = {}
        self._frames.clear()
        self._set_state("idle")
        self._announce("idle")

    # -- saving ------------------------------------------------------------------------------

    def _save(self, run: _Run, name: str | None) -> None:
        """Write the episode so far into the replays directory, as `robojev record` writes it.

        Mid-episode is allowed and is not a special case: the frames are copied and given the same
        terminal frame the ending would have given them, so what is written is a bundle of the
        episode *up to here* rather than a truncated one.
        """
        out_name = safe_name(name, run.episode_id)
        out = self.replays_dir / out_name
        result = run.result
        if result is None:
            frames = list(run.frames)
            if frames:
                frames.append(episode_mod._frame(
                    run.env, run.obs, run.t + 1,
                    np.full(len(frames[-1].action), np.nan, dtype=np.float32), images=True,
                    is_wait_step=False, next_success=frames[-1].next_success, is_terminal=True,
                    policy_query=False, decisions=None))
            result = episode_mod.EpisodeResult(
                frames=frames, success=run.success, steps=run.steps,
                terminated_by="in_progress", error=run.error,
                first_success_step=run.first_success_step)
        if not result.frames:
            self._emit(wire.error("save: this episode has no frames yet"))
            return
        try:
            self.replays_dir.mkdir(parents=True, exist_ok=True)
            bundle = recorder.record(
                run.env, run.policy, result, out, suite=run.spec.suite, task_index=run.spec.task,
                init_state_index=run.spec.init, engine=run.spec.policy, protocol=run.protocol,
                started=run.started, name=out_name, note=self.note)
            write_index(self.replays_dir)
        except (Exception, SystemExit) as exc:                     # noqa: BLE001 - shown verbatim
            self._emit(wire.error(f"save: {exc}"))
            return
        size = sum(f.stat().st_size for f in out.iterdir())
        self._emit(wire.saved(id=out_name, path=str(out), url=f"replays/{out_name}/",
                              decisions=len(bundle["decisions"]), bytes_written=size,
                              success=bool(bundle["success"])))

    # -- bookkeeping -------------------------------------------------------------------------

    def _publish(self, env, obs) -> None:
        images = env.images(obs) if hasattr(env, "images") else {}
        self._frames.publish(images)

    def _touch(self) -> None:
        self._touched = self._clock()

    def _set_state(self, state: str) -> None:
        with self._lock:
            self._state = state

    def _announce(self, state: str, *, run: _Run | None = None, message: str | None = None) -> None:
        self._emit(wire.status(
            state, step=None if run is None else run.t,
            decisions=None if run is None else run.decisions, message=message,
            overrides=self._overrides, episode=run is not None))

    def snapshot(self) -> dict:
        """The `status` a freshly connected client is sent, so a reload lands on what is going on."""
        run = self._run
        return wire.status(self.state, step=None if run is None else run.t,
                           decisions=None if run is None else run.decisions,
                           overrides=self._overrides, episode=run is not None)

    def header(self) -> dict | None:
        """The `hello` payload of the episode in hand, for a client that connected mid-episode."""
        run = self._run
        return None if run is None else run.header


def _default_policy(spec: wire.StartSpec):
    from robojev import policy as policy_mod

    return policy_mod.build(spec.policy, spec.suite, checkpoint=spec.checkpoint, seed=spec.seed,
                            task_index=spec.task, selection=spec.selection)


def _episode_id() -> str:
    return "live-" + time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())


__all__ = ["IDLE_TIMEOUT_SECONDS", "INDEX_FIELDS", "Refused", "Session", "TICK_SECONDS",
           "safe_name", "write_index"]
