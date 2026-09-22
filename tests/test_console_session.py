"""The live session: the state machine, the override, and the bundle it saves.

A console is `run_episode` with the clock taken out, and the one claim that makes the whole design
work is that taking the clock out changes nothing else: the same frames, the same decisions, the
same bundle. That claim is *measured* here rather than asserted -- one trajectory is driven twice,
once by `episode.run_episode` and once by a `Session` taking commands, and the two `episode.json`s
are compared field by field.

No simulator: a thirty-line `DecisionEnv` and the scripted expert, which is the whole point of the
environment protocol (`robojev.envs`). Every test in this file runs on a laptop with numpy.
"""
from __future__ import annotations

import json
import queue
import threading
import time

import numpy as np
import pytest

from robojev import episode as episode_mod
from robojev import policy as policy_mod
from robojev import recorder
from robojev.console import wire
from robojev.console.images import FrameBuffer
from robojev.console.session import Refused, Session, safe_name, write_index

#: The real LIBERO-Spatial task 0 scene, as `tests/test_policy_expert.py` measured it: five
#: objects on a table at z = 0.97. The expert needs a scene to plan in and this is one.
SCENE = {
    "akita_black_bowl_1": {"pos": [-0.063, 0.202, 0.970], "quat": [0.0, 0.0, 0.0, 1.0]},
    "akita_black_bowl_2": {"pos": [-0.189, 0.320, 0.970], "quat": [0.0, 0.0, 0.0, 1.0]},
    "cookies_1": {"pos": [0.058, 0.026, 0.970], "quat": [0.0, 0.0, 0.0, 1.0]},
    "glazed_rim_porcelain_ramekin_1": {"pos": [-0.197, 0.189, 0.970], "quat": [0.0, 0.0, 0.0, 1.0]},
    "plate_1": {"pos": [0.053, 0.205, 0.970], "quat": [0.0, 0.0, 0.0, 1.0]},
}
INSTRUCTION = "pick up the black bowl between the plate and the ramekin and place it on the plate"


class StubEnv:
    """A `DecisionEnv` with a hand that moves and a success predicate that fires on cue.

    Deterministic to the byte, which is what lets the same trajectory be driven twice and the two
    recordings compared. It renders nothing: `recorder.record` writes videos only for frames that
    carry images, so a bundle from this env needs no ffmpeg and the comparison stays CPU-only.
    """

    instruction = INSTRUCTION
    control_freq = 20.0
    stop_on_success = True
    action_dim = 7

    def __init__(self, suite="libero_spatial", task_index=0, render_size=256, env_seed=0,
                 done_at=None, delay=0.0):
        self.suite, self.task_index = suite, task_index
        self.done_at = done_at
        self.delay = float(delay)
        self.closed = False
        self.steps = 0
        self._eef = np.array([0.0, 0.0, 1.10])
        self._width = 0.08

    @property
    def num_init_states(self) -> int:
        return 10

    def reset(self, init_state_index: int) -> dict:
        self.steps = 0
        self._eef = np.array([0.0, 0.0, 1.10 + 0.001 * init_state_index])
        self._width = 0.08
        return {}

    def step(self, action):
        if self.delay:
            time.sleep(self.delay)
        a = np.asarray(action, dtype=np.float64)
        self._eef = self._eef + a[0:3] * 0.01
        self._width = 0.02 if a[6] > 0 else 0.08
        self.steps += 1
        done = self.done_at is not None and self.steps >= self.done_at
        return episode_mod_step(done)

    def state_vector(self, obs) -> np.ndarray:
        return np.array([*self._eef, np.pi, 0.0, 0.0, self._width / 2, -self._width / 2],
                        dtype=np.float32)

    def privileged(self, obs) -> dict:
        return SCENE

    def dummy_action(self):
        return [0.0] * 6 + [-1.0]

    def close(self) -> None:
        self.closed = True


class RenderEnv(StubEnv):
    """The same, with two tiny camera renders -- for the tests about the picture stream."""

    def images(self, obs) -> dict:
        shade = np.uint8(self.steps % 255)
        return {"agentview": np.full((8, 8, 3), shade, np.uint8),
                "wrist": np.full((8, 8, 3), 255 - shade, np.uint8)}


def episode_mod_step(done: bool):
    from robojev.envs import StepResult

    return StepResult(obs={}, done=done)


class Sink:
    """Everything the session emitted, in order, readable from the test's own thread."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self.seen: list[dict] = []
        #: How far `wait` has read. Each call resumes where the last one stopped, so two waits for
        #: the same type are two messages rather than the same one twice.
        self._at = 0

    def __call__(self, message: dict) -> None:
        self._q.put(message)

    def _pump(self, timeout: float) -> bool:
        try:
            self.seen.append(self._q.get(timeout=timeout))
            return True
        except queue.Empty:
            return False

    def wait(self, type_: str, timeout: float = 20.0, **fields) -> dict:
        """The next message of `type_` (optionally matching `fields`), or a failure naming what
        did arrive -- a hang in a state machine test must say where it stopped."""
        deadline = time.monotonic() + timeout
        while True:
            while self._at < len(self.seen):
                message = self.seen[self._at]
                self._at += 1
                if message.get("type") == type_ and all(message.get(k) == v
                                                        for k, v in fields.items()):
                    return message
            if not self._pump(max(0.05, deadline - time.monotonic())):
                if time.monotonic() > deadline:
                    raise AssertionError(
                        f"no {type_} {fields or ''} in {[m.get('type') for m in self.seen]}"
                        + (f"; errors {[m['message'] for m in self.seen if m.get('type') == 'error']}"
                           if any(m.get("type") == "error" for m in self.seen) else ""))

    def quiet(self, seconds: float = 0.4) -> list[dict]:
        """Whatever arrives in the next `seconds`. Used to prove that nothing does."""
        end = time.monotonic() + seconds
        out: list[dict] = []
        while time.monotonic() < end:
            if self._pump(max(0.01, end - time.monotonic())):
                out.append(self.seen[-1])
        return out

    def of(self, type_: str) -> list[dict]:
        return [m for m in self.seen if m.get("type") == type_]


def make_session(tmp_path, *, env=None, idle_timeout=60.0, note="", **spec_kw):
    """A session over a stub environment and the scripted expert. Returns `(session, sink, env)`."""
    env = env or StubEnv()
    sink = Sink()
    session = Session(
        sink, frames=FrameBuffer(), replays_dir=tmp_path / "replays",
        base_url="http://127.0.0.1:8765", policies=("expert",), render_size=8,
        idle_timeout=idle_timeout, note=note,
        env_factory=lambda suite, task, render_size=256, env_seed=0: env,
        build_policy=lambda spec: policy_mod.build(
            "expert", spec.suite, seed=spec.seed, selection=spec.selection))
    return session, sink, env


def start(session, sink, **kw):
    spec = wire.StartSpec(**{"max_steps": 25, **kw})
    session.submit(wire.Command("start", spec=spec))
    return sink.wait("hello")


def send(session, op, **kw):
    session.submit(wire.Command(op, **kw))


def finish(session):
    session.close()


# ------------------------------------------------------------------------------- the lifecycle

def test_an_idle_console_refuses_every_command_but_start(tmp_path):
    session, sink, _ = make_session(tmp_path)
    assert session.state == "idle" and not session.busy
    for op in ("step", "run", "pause", "override", "save"):
        with pytest.raises(Refused) as exc:
            send(session, op)
        assert "no episode" in str(exc.value)
    # `reset` on nothing is not an error -- it is what a reloaded page sends to find out.
    send(session, "reset")
    assert sink.wait("status")["state"] == "idle"


def test_start_announces_the_episode_header_before_anything_has_been_decided(tmp_path):
    """The page sizes the scrubber and names the task off `hello`, so it has to arrive before the
    first decision rather than with it."""
    session, sink, _ = make_session(tmp_path)
    hello = start(session, sink, task=3, init=2)
    header = hello["episode"]
    assert header["instruction"] == INSTRUCTION
    assert (header["suite"], header["task_index"], header["init_state_index"]) == (
        "libero_spatial", 3, 2)
    assert header["policy"] == "expert" and header["control_rate"] == 20.0
    assert header["execute_steps"] == 5 and header["max_steps"] == 25
    assert header["max_decisions"] == 5 and header["schema_version"] == recorder.SCHEMA_VERSION
    assert hello["video"]["cameras"]["agentview"].endswith("/frame/agentview.png")
    assert sink.wait("status", state="paused")["episode"] is True
    finish(session)


def test_a_second_episode_is_refused_while_one_is_open(tmp_path):
    """One console, one simulator. A second `start` that quietly built another would be two
    MuJoCo instances and two GPUs' worth of weights on a machine that has one of each."""
    session, sink, _ = make_session(tmp_path)
    start(session, sink)
    with pytest.raises(Refused) as exc:
        session.submit(wire.Command("start", spec=wire.StartSpec()))
    assert "already open" in str(exc.value)
    finish(session)


def test_a_policy_that_cannot_be_built_says_so_and_leaves_the_console_idle(tmp_path):
    sink = Sink()
    session = Session(sink, frames=FrameBuffer(), replays_dir=tmp_path, base_url="http://x",
                      env_factory=lambda *a, **k: StubEnv(),
                      build_policy=lambda spec: (_ for _ in ()).throw(
                          policy_mod.PolicyError("no checkpoint at /nowhere")))
    session.submit(wire.Command("start", spec=wire.StartSpec()))
    assert "no checkpoint" in sink.wait("error")["message"]
    assert sink.wait("status", state="idle")["episode"] is False
    assert session.state == "idle"


# --------------------------------------------------------------------------------- one decision

def test_step_takes_exactly_one_decision_and_sends_it_as_a_bundle_entry(tmp_path):
    """The decision on the wire is `recorder.decision_entry`'s object, which is the entry
    `episode.json` holds -- the page has one reader for both and it only does if they match."""
    session, sink, env = make_session(tmp_path)
    start(session, sink)
    send(session, "step")
    first = sink.wait("decision")["decision"]
    assert first["index"] == 0 and first["control_step"] == 0 and first["t"] == 0.0
    assert set(first) >= {"state", "questions", "grip_latch", "grounding", "subgoal", "action",
                          "rim_candidates", "waypoint_cm"}
    assert set(first["questions"]) >= {"move_x", "move_y", "move_z", "rim", "grip", "subgoal"}
    assert len(first["action"]) == 7
    assert env.steps == 5                                          # one chunk, executed whole
    send(session, "step")
    second = sink.wait("decision", timeout=20.0)["decision"]
    assert second["index"] == 1 and second["control_step"] == 5
    assert second["t"] == pytest.approx(0.25)
    assert env.steps == 10
    finish(session)


def test_the_state_text_the_model_read_rides_on_every_decision(tmp_path):
    """Not an optimisation to drop: the panel's whole claim is that it shows what the model
    actually read, and a live view that left the paragraph out would be a different claim."""
    session, sink, _ = make_session(tmp_path)
    start(session, sink)
    send(session, "step")
    state = sink.wait("decision")["decision"]["state"]
    assert "x:" in state and "instruction" in state.lower() or len(state.splitlines()) > 5
    finish(session)


def test_run_steps_on_its_own_and_pause_lands_within_one_decision(tmp_path):
    session, sink, env = make_session(tmp_path, env=StubEnv(delay=0.01), max_steps=500)
    start(session, sink)
    send(session, "run")
    sink.wait("status", state="running")
    sink.wait("decision")
    while len(sink.of("decision")) < 3:
        sink.wait("decision", timeout=10.0)
    send(session, "pause")
    sink.wait("status", state="paused")
    taken = len(sink.of("decision"))
    assert sink.quiet(0.5) == [] or len(sink.of("decision")) <= taken + 1
    assert session.state == "paused"
    finish(session)


def test_an_episode_that_succeeds_ends_itself_and_says_how(tmp_path):
    session, sink, _ = make_session(tmp_path, env=StubEnv(done_at=12))
    start(session, sink)
    send(session, "run")
    done = sink.wait("done")
    assert done["success"] is True and done["terminated_by"] == "success"
    assert done["steps"] == 12 and done["decisions"] == 3
    assert sink.wait("status", state="done")["state"] == "done"
    finish(session)


def test_an_episode_that_runs_out_of_horizon_ends_at_the_cap(tmp_path):
    session, sink, _ = make_session(tmp_path)
    start(session, sink, max_steps=15)
    send(session, "run")
    done = sink.wait("done")
    assert (done["success"], done["terminated_by"], done["decisions"]) == (False, "max_steps", 3)
    finish(session)


def test_reset_ends_the_episode_and_lets_go_of_the_simulator(tmp_path):
    session, sink, env = make_session(tmp_path)
    start(session, sink)
    send(session, "step")
    sink.wait("decision")
    send(session, "reset")
    assert sink.wait("done")["terminated_by"] == "reset"
    sink.wait("status", state="idle")
    assert env.closed is True and session.state == "idle"


def test_a_console_nobody_is_driving_closes_its_episode(tmp_path):
    """A console holds a simulator -- and under `--policy model` a GPU -- so a tab left open over
    lunch must not hold either."""
    session, sink, env = make_session(tmp_path, idle_timeout=0.0)
    start(session, sink)
    assert "no command" in sink.wait("error", timeout=10.0)["message"]
    assert sink.wait("done")["terminated_by"] == "idle"
    sink.wait("status", state="idle")
    assert env.closed is True


def test_a_finished_episode_still_answers_save_and_refuses_another_step(tmp_path):
    session, sink, _ = make_session(tmp_path, env=StubEnv(done_at=6))
    start(session, sink)
    send(session, "run")
    sink.wait("done")
    send(session, "step")
    assert "has ended" in sink.wait("error")["message"]
    finish(session)


# ------------------------------------------------------------------------------- the override

def test_a_candidate_is_armed_then_executed_then_recorded_as_overridden(tmp_path):
    """The assisted mode, end to end: the operator forces one answer, the *forced* one is what the
    arm executes, and the bundle entry says so with the flag the replay types already carry."""
    session, sink, _ = make_session(tmp_path)
    start(session, sink)
    send(session, "step")
    natural = sink.wait("decision")["decision"]
    assert natural["questions"]["move_x"]["overridden"] is False

    forced = "+" if natural["questions"]["move_x"]["choice"] != "+" else "-"
    send(session, "override", qid="move_x", candidate=forced)
    assert sink.wait("status", overrides={"move_x": forced})["overrides"] == {"move_x": forced}

    send(session, "step")
    second = sink.wait("decision", timeout=20.0)["decision"]
    assert second["index"] == 1
    assert second["questions"]["move_x"]["choice"] == forced
    assert second["questions"]["move_x"]["overridden"] is True
    assert second["questions"]["move_y"]["overridden"] is False

    # An override is an instruction about *one* decision, not a setting: it is spent.
    send(session, "step")
    third = sink.wait("decision", timeout=20.0)["decision"]
    assert third["questions"]["move_x"]["overridden"] is False
    assert sink.of("status")[-1]["overrides"] == {}
    finish(session)


def test_an_armed_override_can_be_taken_back_off(tmp_path):
    session, sink, _ = make_session(tmp_path)
    start(session, sink)
    send(session, "override", qid="grip", candidate="true")
    sink.wait("status", overrides={"grip": "true"})
    send(session, "override", qid="grip", candidate=None)
    sink.wait("status", overrides={})
    send(session, "step")
    assert sink.wait("decision")["decision"]["questions"]["grip"]["overridden"] is False
    finish(session)


def test_an_override_naming_a_question_or_a_candidate_that_does_not_exist_is_refused_at_once(tmp_path):
    """Refused when it is armed and not a decision later: an override that silently goes nowhere
    looks exactly like the model disagreeing with the operator."""
    session, sink, _ = make_session(tmp_path)
    start(session, sink)
    send(session, "override", qid="elbow", candidate="up")
    assert "no such question" in sink.wait("error")["message"]
    send(session, "override", qid="move_x", candidate="sideways")
    assert "not one of" in sink.wait("error", timeout=10.0)["message"]
    assert session.snapshot()["overrides"] == {}
    finish(session)


# ------------------------------------------------------------------------------------- saving

def test_save_writes_the_bundle_and_puts_it_in_the_index(tmp_path):
    session, sink, _ = make_session(tmp_path, env=StubEnv(done_at=10), note="a live one")
    hello = start(session, sink)
    send(session, "run")
    sink.wait("done")
    send(session, "save")
    saved = sink.wait("saved", timeout=30.0)
    out = tmp_path / "replays" / saved["id"]
    assert saved["id"] == hello["episode"]["id"] and out.is_dir()

    bundle = json.loads((out / "episode.json").read_text())
    assert bundle["id"] == saved["id"] and bundle["success"] is True
    assert bundle["note"] == "a live one" and bundle["policy"] == "expert"
    assert len(bundle["decisions"]) == saved["decisions"] == 2
    assert bundle["schema_version"] == recorder.SCHEMA_VERSION

    index = json.loads((tmp_path / "replays" / "index.json").read_text())
    assert [row["id"] for row in index] == [saved["id"]]
    assert index[0]["decisions"] == 2 and index[0]["instruction"] == INSTRUCTION
    finish(session)


def test_a_saved_name_from_a_browser_is_used_as_a_name_and_not_as_a_path(tmp_path):
    assert safe_name("../../etc/passwd", "x") == "etc-passwd"
    assert safe_name("", "live-1") == "live-1"
    assert safe_name("My Run #2", "x") == "My-Run-2"
    session, sink, _ = make_session(tmp_path, env=StubEnv(done_at=6))
    start(session, sink)
    send(session, "run")
    sink.wait("done")
    send(session, "save", name="../escape")
    saved = sink.wait("saved", timeout=30.0)
    assert saved["id"] == "escape"
    assert (tmp_path / "replays" / "escape" / "episode.json").is_file()
    finish(session)


def test_the_index_keeps_the_order_it_already_had_and_appends_what_is_new(tmp_path):
    """The strip is curated, so a save must add to it rather than reshuffle it."""
    replays = tmp_path / "replays"
    for name, title in (("drawer", "the drawer"), ("bowl", "the bowl")):
        directory = replays / name
        directory.mkdir(parents=True)
        (directory / "episode.json").write_text(json.dumps({
            "id": name, "title": title, "instruction": title, "note": "", "success": True,
            "decisions": [{}], "max_decisions": 44, "task_index": 0, "init_state_index": 0,
            "suite": "libero_spatial", "poster": "poster.jpg"}))
    (replays / "index.json").write_text(json.dumps([{"id": "drawer"}, {"id": "bowl"}]))
    (replays / "aaa-new").mkdir()
    (replays / "aaa-new" / "episode.json").write_text(json.dumps({
        "id": "aaa-new", "title": "t", "instruction": "i", "note": "", "success": False,
        "decisions": [], "max_decisions": 44, "task_index": 1, "init_state_index": 1,
        "suite": "libero_spatial"}))
    rows = write_index(replays)
    assert [row["id"] for row in rows] == ["drawer", "bowl", "aaa-new"]
    assert rows[0]["decisions"] == 1 and rows[2]["poster"] is None


def test_an_episode_can_be_saved_before_it_has_finished(tmp_path):
    session, sink, _ = make_session(tmp_path)
    start(session, sink)
    send(session, "step")
    sink.wait("decision")
    send(session, "save", name="partial")
    saved = sink.wait("saved", timeout=30.0)
    bundle = json.loads((tmp_path / "replays" / "partial" / "episode.json").read_text())
    assert saved["decisions"] == 1 and bundle["terminated_by"] == "in_progress"
    assert bundle["total_frames"] == 6                             # five executed, one terminal
    finish(session)


# --------------------------------------------------------- the claim the whole design rests on

def test_a_saved_live_episode_is_the_bundle_the_recorder_would_have_written(tmp_path):
    """One trajectory, driven twice.

    `episode.run_episode` is the loop every published number is measured with; the session is a
    human driving the same machinery. If the two ever produced different bundles then a live view
    would be showing something a recording could not, and the one reader the page has for both
    would be reading two formats. So they are compared rather than trusted -- every field but the
    three that are a statement about *when* the recording happened.
    """
    steps, name = 25, "same"

    recorded_env = StubEnv()
    recorded_policy = policy_mod.build("expert", "libero_spatial", seed=7, selection="argmax")
    protocol = recorded_policy.protocol(steps)
    result = episode_mod.run_episode(recorded_env, recorded_policy, protocol,
                                     init_state_index=1, seed=7)
    offline = recorder.record(recorded_env, recorded_policy, result, tmp_path / "offline",
                              suite="libero_spatial", task_index=0, init_state_index=1,
                              engine="expert", protocol=protocol, started=time.time(), name=name)

    session, sink, _ = make_session(tmp_path)
    start(session, sink, task=0, init=1, max_steps=steps)
    send(session, "run")
    sink.wait("done")
    send(session, "save", name=name)
    sink.wait("saved", timeout=30.0)
    live = json.loads((tmp_path / "replays" / name / "episode.json").read_text())
    finish(session)

    # `recorded_at` and `wall_seconds` are when it was written, not what was recorded; `id` is the
    # directory's name and both were told the same one, so it is compared and then dropped with
    # them to keep the diff below readable.
    assert live["id"] == offline["id"] == name
    for field in ("recorded_at", "wall_seconds", "id"):
        live.pop(field), offline.pop(field)

    assert live["decisions"] == offline["decisions"]
    assert live == offline


def test_the_live_decisions_are_the_saved_ones_in_the_same_order(tmp_path):
    """The other half of the same claim, from the socket's side: what the page was shown, decision
    by decision, is what the file it can then download holds."""
    session, sink, _ = make_session(tmp_path, env=StubEnv(done_at=20))
    start(session, sink, max_steps=40)
    send(session, "run")
    sink.wait("done")
    send(session, "save", name="streamed")
    sink.wait("saved", timeout=30.0)
    streamed = [m["decision"] for m in sink.of("decision")]
    bundle = json.loads((tmp_path / "replays" / "streamed" / "episode.json").read_text())
    assert streamed == bundle["decisions"]
    finish(session)


# ------------------------------------------------------------------------------- the pictures

def test_every_control_step_publishes_its_renders(tmp_path):
    """The live view is the two cameras, so they have to arrive as they are produced rather than
    once per decision -- an mp4 that is still being written is not something a browser can play."""
    frames = FrameBuffer()
    sink = Sink()
    env = RenderEnv()
    session = Session(sink, frames=frames, replays_dir=tmp_path / "replays",
                      base_url="http://127.0.0.1:8765", render_size=8,
                      env_factory=lambda *a, **k: env,
                      build_policy=lambda spec: policy_mod.build("expert", spec.suite, seed=7))
    session.submit(wire.Command("start", spec=wire.StartSpec(max_steps=25)))
    sink.wait("hello")
    assert frames.cameras() == ["agentview", "wrist"]              # the reset state, before a step
    before = frames.seq
    session.submit(wire.Command("step"))
    sink.wait("decision")
    assert frames.seq == before + 5                                # one per control step
    assert frames.png("agentview")[1][:8] == b"\x89PNG\r\n\x1a\n"
    session.close()
    assert frames.png("agentview") is None                         # let go of with the episode


def test_the_session_survives_a_render_the_environment_cannot_produce(tmp_path):
    session, sink, _ = make_session(tmp_path)                      # StubEnv has no `images`
    start(session, sink)
    send(session, "step")
    assert sink.wait("decision")["decision"]["index"] == 0
    finish(session)


def test_nothing_is_left_running_after_close(tmp_path):
    session, sink, env = make_session(tmp_path)
    start(session, sink)
    send(session, "run")
    session.close()
    assert session.state == "idle" and env.closed is True
    assert not any(t.name == "robojev-console" and t.is_alive() for t in threading.enumerate())
