# The live source — what a `robojev console` server would have to send

**Nothing here is built.** This page ships one data source, `ReplaySource`, which reads recorded
bundles. `LiveSource` is a stub that refuses with a pointer to this file. It is written down now
so the recorded format and the live one cannot drift apart later: **a live `decision` message is
byte-for-byte one entry of a bundle's `decisions` array**, and every difference between live and
replay is confined to the envelope around it.

Selecting it: `?live=ws://127.0.0.1:8765/` on the URL, or `VITE_LIVE_URL` at build time. The
query parameter wins, so one deployed build can be pointed at a console without rebuilding.

## Transport

One WebSocket. Text frames, one JSON object per frame, each with a `type`. The server is expected
to be local (`127.0.0.1`), unauthenticated, and single-client; nothing in the page assumes
otherwise, and nothing in the page ever sends anything the server must trust.

## Server → client

### `hello` — once, immediately on connect

The episode header: exactly the bundle's top-level fields, minus the ones that only exist once an
episode is over.

```jsonc
{
  "type": "hello",
  "schema_version": 1,
  "episode": {
    "id": "live-2026-09-22T09-14-03",
    "suite": "libero_spatial", "task_index": 4, "init_state_index": 1,
    "instruction": "pick up the black bowl in the top drawer …",
    "policy": "robojev", "checkpoint_revision": "bb7ba09784d7",
    "questions_version": "v2",
    "control_rate": 20.0, "wait_steps": 10, "execute_steps": 5,
    "max_steps": 220, "max_decisions": 44
  },
  "video": {
    // A live episode has no mp4: it is still happening. So the header names a stream instead,
    // and the page shows it in place of the <video src>. MJPEG over HTTP is the cheap answer
    // (an <img> tag and nothing else); WebRTC is the good one.
    "kind": "mjpeg",                              // mjpeg | webrtc | none
    "agentview": "http://127.0.0.1:8765/stream/agentview",
    "wrist": "http://127.0.0.1:8765/stream/wrist"
  }
}
```

**The clock is the one thing that changes.** In a replay, `decision.t` is a position in a finished
video and the video drives the panel. Live, there is no seekable timeline: the newest decision is
the current one, and `t` is wall-clock seconds since the episode began — useful for a readout, not
for a seek. A live view therefore follows the stream, and the scrubber becomes a history of what
has already happened rather than a control over what is shown.

### `decision` — one per decision point, as it is made

```jsonc
{ "type": "decision", "decision": { /* exactly one entry of episode.json's `decisions` */ } }
```

Every field of that object is required, `state` included: the panel's whole claim is that it shows
what the model actually read, and a live view that dropped the paragraph to save bandwidth would
be a different claim. At ~4 KB of JSON per decision and four decisions a second this is 16 KB/s.

### `status` — whenever the run's state changes

```jsonc
{ "type": "status", "state": "running", "step": 85, "message": null }
```

`state` is one of `waiting` (the settle steps), `running`, `done`, `error`.

### `done` — once, when the episode ends

```jsonc
{ "type": "done", "success": true, "terminated_by": "success", "steps": 144, "decisions": 29 }
```

After `done` the page holds the episode it has and stops expecting anything. A server that wants
to start another episode sends a fresh `hello`.

### `error`

```jsonc
{ "type": "error", "message": "policy server exited: …", "fatal": true }
```

## Client → server

The page is a viewer, so this half is deliberately thin. It sends nothing at all unless a control
is added later; the two that are anticipated:

```jsonc
{ "type": "ping" }                                   // liveness, answered with {"type":"pong"}
{ "type": "override", "qid": "grip", "candidate": "true" }
```

`override` is the one interesting one, and it is why `QuestionAnswer.overridden` exists in the
replay types already: forcing an answer for the next decision is what a console is *for*, and the
panel can already draw the state where the executed answer is not the model's. A server that does
not implement it must reply with an `error` rather than ignoring it — an override that silently
goes nowhere is the worst failure mode this page has.

## What the server has to do that it does not do today

1. **Own the episode loop.** `robojev.episode.run_episode` with an `on_step` hook, exactly as
   `recorder.py` does, pushing `policy.last_decisions` out of the socket instead of into a list.
2. **Stream frames.** The loop already renders every control step; a live view needs them as they
   are produced, which an mp4 that is still being written cannot provide. MJPEG at 20 Hz of
   256×256 JPEGs is about 1–2 MB/s on localhost.
3. **Say `hello` before the first decision**, so the page can size the scrubber and name the task
   before there is anything to show.
4. **Keep `t` monotone** and consistent with `control_step / control_rate`, so that a live episode
   which is later saved as a bundle needs no fixing up: a recorded bundle should be exactly what
   was streamed, concatenated.
