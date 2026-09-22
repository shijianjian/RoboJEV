# The live source — what `robojev console` sends, and what the page does with it

Two halves of this repository ship one format. `robojev record` writes an **episode bundle**
(`robojev/recorder.py`) and the page replays it; `robojev console` runs the same loop with the
clock taken out and streams it. The rule that keeps the two from drifting is one sentence:

> **A live `decision` message is byte-for-byte one entry of a bundle's `decisions` array.**

It is true by construction — the console calls `recorder.decision_entry`, the same function
`episode.json` is written with — and it is *measured*, in `tests/test_console_session.py`, which
drives one trajectory through `episode.run_episode` and through a live session and compares the
two bundles field by field. Everything that differs between a live view and a replay is in the
envelope around that object.

Running it:

```bash
robojev console                       # http://127.0.0.1:8765/ — the built app and the socket
robojev console --port 9000 --policy expert --task 4 --init 1
```

Selecting it, when the page is served from somewhere else: `?live=ws://127.0.0.1:8765/ws` on the
URL, or `VITE_LIVE_URL` at build time. A page the console serves itself needs neither — the
console writes its own address into the HTML it hands out (`window.__ROBOJEV_CONSOLE__`), which is
also why **a GitHub Pages visit makes no probe and shows no live controls**: there is nothing to
find and the page does not go looking.

## Transport

One WebSocket at `/ws`. Text frames, one JSON object per frame, each with a `type`. The server is
local (`127.0.0.1`), unauthenticated, **single-client and single-episode**: it holds a simulator
and, under `--policy model`, a GPU. A second browser is sent a fatal `error` and disconnected
rather than shown a session it cannot drive.

Server-side WebSocket and PNG encoding are written out in `robojev/console/ws.py` and
`robojev/console/images.py` rather than imported, so the console adds no dependency to an install
that is numpy and the standard library.

## Server → client

### `config` — once, on connect

What this console can be asked for. The one message with no replay counterpart: a recording has no
pickers.

```jsonc
{
  "type": "config", "protocol": 1,
  "policies": ["expert"],                     // what this interpreter can actually serve
  "suites": ["libero_10", "libero_spatial", …],
  "default": { "suite": "libero_spatial", "task": 0, "init": 0, "policy": "expert",
               "selection": "argmax", "checkpoint": null, "seed": 7, "max_steps": null },
  "video": { … },                             // see `hello`
  "state": "idle",
  "replays": "replays/",
  "weights": [                                // what the page offers: checkpoints, not engines
    { "id": "model:/…/checkpoints/robojev/libero_spatial", "label": "robojev/libero_spatial",
      "revision": "bb7ba09784d7", "policy": "model", "checkpoint": "/…/checkpoints/robojev/libero_spatial" }
  ]
}
```

A client that connects while an episode is already open is then sent that episode's `hello` and a
`status`, so a reload lands back on what is going on.

### `hello` — when an episode begins

The episode header: exactly the bundle's top-level fields, minus the ones that only exist once an
episode is over (`success`, `terminated_by`, `steps`, `media`, `poster`, `recorded_at`, …).

```jsonc
{
  "type": "hello",
  "schema_version": 1,
  "episode": {
    "id": "live-2026-09-22T09-14-03",
    "suite": "libero_spatial", "task_index": 4, "init_state_index": 1,
    "instruction": "pick up the black bowl in the top drawer …",
    "title": "…", "note": "",
    "policy": "expert", "checkpoint_revision": "scripted-v2", "checkpoint_repo": null,
    "questions_version": "v2", "selection": "argmax",
    "control_rate": 20.0, "wait_steps": 0, "execute_steps": 5,
    "max_steps": 220, "max_decisions": 44
  },
  "video": {
    // A live episode has no mp4: it is still happening, and a browser cannot play a file that is
    // still being written. So the header names the two pictures instead, and the page refreshes
    // an <img> one frame at a time — it asks for the next when the previous has painted, so a
    // slow machine shows fewer frames rather than queueing requests behind itself.
    //
    // `poll` and not MJPEG because the encoder is the argument: there is no JPEG encoder in the
    // standard library, `zlib` writes a PNG in thirty lines, and a multipart stream of PNGs is a
    // browser-support question that one request per picture does not have. On loopback at 20 Hz
    // and 256 square it is a few MB/s and a few per cent of a core.
    "kind": "poll",
    "cameras": {
      "agentview": "http://127.0.0.1:8765/frame/agentview.png",
      "wrist": "http://127.0.0.1:8765/frame/wrist.png"
    },
    "rate": 20.0
  }
}
```

`hello` also carries `"scene"`: where the page loads the episode's 3D scene from (robopp's compiled
bundle, by hash), or `null` when the environment exported none.

```jsonc
"scene": { "hash": "df35d37f…", "nq": 48,
           "xml": "http://127.0.0.1:8765/scenes/df35d37f…/scene.xml",
           "assets": "http://127.0.0.1:8765/scenes/df35d37f…/assets/" }
```

**The clock is the one thing that changes.** In a replay, `decision.t` is a position in a finished
video and the video drives the panel. Live there is nothing to seek: the newest decision is the
current one, and `t` is still `control_step / control_rate`, so a live episode saved as a bundle
needs no fixing up.

### `pose` — once per control step

```jsonc
{ "type": "pose", "step": 85, "qpos": [0.01234, -0.18055, …] }   // model.nq numbers, 5 decimals
```

The simulator's joint positions after that step: what the 3D scene is posed with. It is the same
row a saved bundle writes into `qpos.bin` for that frame. ~400 bytes at 20 Hz; the page keeps the
newest and drops one older than what it already has.

### `decision` — one per decision point, as it is made

```jsonc
{ "type": "decision", "decision": { /* exactly one entry of episode.json's `decisions` */ } }
```

Every field of that object is present, `state` included: the panel's whole claim is that it shows
what the model actually read, and a live view that dropped the paragraph to save bandwidth would be
a different claim. At ~4 kB of JSON per decision and four decisions a second this is 16 kB/s.

The message is sent **after** its chunk has been executed, so the picture beside it is the result
of the action rather than the state before it.

### `status` — whenever the run's state changes

```jsonc
{ "type": "status", "state": "running", "step": 85, "decisions": 17,
  "message": null, "overrides": {"move_x": "+"}, "episode": true }
```

`state` is one of `idle` (no episode), `starting` (building the simulator and the policy — the slow
one), `waiting` (the settling steps), `paused` (an episode in hand, stopped), `running`, `done`,
`error`. `episode` says whether there is an episode to drive at all.

**`overrides` is authoritative.** It is the armed set as the *server* holds it, and the page draws
the held candidates from this and never from its own memory of what was clicked — a bar lit for an
override the console refused is the worst thing this page could draw.

### `done` — once, when the episode ends

```jsonc
{ "type": "done", "success": true, "terminated_by": "success", "steps": 144,
  "decisions": 29, "error": null }
```

`terminated_by` is the bundle's own field, plus two a recording cannot have: `reset` (the operator
ended it) and `idle` (nobody did anything for `--idle-timeout` seconds, so the console let go of
the simulator).

After `done` the frames are still in hand and `save` still works; `reset` lets go of them. An
episode the operator reset is let go of at once, because the next thing they do is start another.

### `saved` — a live episode written out as a bundle

```jsonc
{ "type": "saved", "id": "live-2026-09-22T09-14-03", "path": "/…/runs/live-…",
  "url": "replays/live-…/", "decisions": 29, "bytes": 1840112, "success": true }
```

The directory `robojev record` would have written, `index.json` rewritten around it and the scene
copied into `replays/scenes/`. The console serves `/replays/` out of that directory in front of the
built app's copy, and the page reads the index again on `saved`, so the episode is in Recent runs at
once rather than after a rebuild.

### `error`

```jsonc
{ "type": "error", "message": "override 'elbow': this question set has no such question",
  "fatal": false }
```

`fatal` means the console will not take this connection back — today, only "another window has it".
A refused command is *not* fatal and does not end the episode.

### `pong`, `tasks`

`{"type": "pong"}` answers `ping`. `{"type": "tasks", "suite": …, "tasks": [{index, instruction,
init_states}]}` answers `tasks` — read off LIBERO's own task definitions, off the main thread and
cached, because importing the simulator costs seconds and a console that could not be looked at
until it had would take five seconds to say hello. Until it arrives the pickers number the tasks;
a console with no simulator answers with an `error` and they stay numbered.

## Client → server

```jsonc
{ "type": "start", "suite": "libero_spatial", "task": 4, "init": 1, "policy": "expert",
  "selection": "argmax", "temperature": null, "checkpoint": null, "seed": 7, "max_steps": null }
{ "type": "step" }                                   // one decision, then stop
{ "type": "run" }                                    // keep deciding until done or paused
{ "type": "pause" }                                  // lands within one decision
{ "type": "reset" }                                  // end the episode, let go of the simulator
{ "type": "override", "qid": "grip", "candidate": "true" }
{ "type": "override", "qid": "grip", "candidate": null }    // …and take it back off
{ "type": "save", "name": "my-episode" }             // `name` optional: the episode's own id
{ "type": "ping" }
{ "type": "tasks", "suite": "libero_spatial" }
```

`selection` is `argmax` or `sample@<T>`; a bare `sample` with a `temperature` beside it is spelled
into the canonical form, because that string is what the bundle records and two spellings on the
wire would become two spellings in the bundles.

**`override` is the interesting one**, and it is why `QuestionAnswer.overridden` was in the replay
types before any console existed. It holds one question's answer for the **next** decision: the
console executes the forced answer, the bundle entry records it with `overridden: true`, and the
armed set is cleared — an override is an instruction about one decision, not a setting. It is
checked against the question set when it is armed and not a decision later, because an override
that silently goes nowhere looks exactly like the model disagreeing with the operator. A server
that does not implement it must reply with an `error` rather than ignoring it.

Anything the console will not act on comes back as an `error` naming the reason: an unknown
`type`, a policy this interpreter cannot serve, a `step` with no episode, a second `start` while
one is open.

## The HTTP half

The same port serves the rest, so `robojev console` alone opens a working page:

| path | what |
| --- | --- |
| `/` and `/assets/…` | `web/dist`, with `window.__ROBOJEV_CONSOLE__` injected into the page |
| `/replays/…` | the replays directory first, the built app's copy second; byte ranges, because a browser seeking an mp4 asks for one |
| `/frame/<camera>.png` | the latest render, encoded on demand and cached by frame; `204` while nothing has been rendered |
| `/scenes/<hash>/…` | compiled scenes: `showcase/scenes`, `data/scenes`, `$ROBOJEV_HOME/scenes`, then the live episode's export (`--scenes-dir`) |
| `/catalogue/…` | the task catalogue from `showcase/`, `data/`, `$ROBOJEV_HOME`; `/catalogue/index.json` lists it |
| `/ws` | the above |

## What is deliberately not here

No authentication, no TLS, no `0.0.0.0`: this is a local tool, and the page never sends the server
anything the server must trust. No WebRTC — one picture at a time is enough for a 20 Hz 256-square
render on loopback, and it costs the page an `<img>` and the server nothing. No second episode: one
console, one simulator.

## The bundle, version 2

A bundle is `recorder.py`'s directory: `episode.json`, `agentview.mp4`, `wrist.mp4`, `poster.jpg`,
and since `schema_version: 2`

```jsonc
"qpos":  { "path": "qpos.bin", "frames": 155, "nq": 48, "dtype": "<f4" },  // frames x nq, row-major
"scene": { "hash": "73d6392e…", "nq": 48 }
```

`qpos.bin` holds one row per video frame, so the 3D scene and the videos are one clock. The scene is
robopp's compiled bundle `scenes/<hash>/` (`scene.xml` and `assets/`; `robojev/scene_bundle.py` is
robopp's export, ported), the same hash the task's catalogue entry names. The page reads version 1
as well; such a bundle replays on its videos alone.
`robojev scene <bundle>` upgrades one by replaying its recorded actions in the simulator and
refusing unless the replay reproduces the recording (outcome, steps, frames, and every agentview
frame within a few grey levels of the video).

