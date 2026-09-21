# RoboJEV

Code plans and the model judges: a
scripted plan executor proposes a waypoint, a tracker prints the scene and the plan as text, and
one forward pass answers `move_x/y/z`, `size_x/y/z`, `yaw`, `rim`, `grip` and `subgoal`, which
compose into one action. The model is a NanoJev-style parallel decision model — a Qwen3-0.6B
backbone with a set head per question, fine-tuned by NanoJev's own trainer.

Live demo (replays, no backend): <https://shijianjian.github.io/RoboJEV/>

## Results

| what | measured |
| --- | --- |
| scripted expert, LIBERO-Spatial | 90/100 (10 tasks × 10 start states) |
| this model (run 8, `bb7ba09784d7`), closed loop | 45/50 and 46/50 — the same 50 episodes measured twice |
| …the drawer task alone | 3–4 of 5 |
| hosted official Jev, zero-shot, 10 episodes | 3/10, where this model scores 8/10 |
| earlier question sets | 0/40 and 0/30 |

## Quick start

```bash
pip install -e .                       # numpy only; add [libero] for the simulator, [model] for torch
robojev run --task 0 --init 0 --policy expert       # one closed-loop episode, prints success
robojev record --task 0 --init 0 --policy expert --out web/public/replays/demo
cd web && npm ci && npm run dev        # the replay front end
```

`run` and `record` need `robojev[libero]` and `MUJOCO_GL=egl`. The web app needs neither the
simulator nor any weights — it reads the episode bundles in `web/public/replays/`.

## Repository

| path | what |
| --- | --- |
| `robojev/` | the package: plan executor, scripted expert, state text, questions, composer, parser, grounding |
| `robojev/envs/` | the environment protocol and the LIBERO adapter (the only simulator import) |
| `robojev/policy.py` | the in-process policy: scripted expert, local checkpoint, or the hosted Jev |
| `robojev/cli.py` | `run`, `record`, `harvest`, `dagger`, `train` |
| `web/`, `tools/` | the replay front end and the scripts that serve and check it |
| `docs/DESIGN.md` | why it is shaped this way, and every measurement |

## How it works

1. `scene.py` reads the simulator's object poses and the fixtures' collision boxes.
2. `expert.py` + `skill.py` run a stage table (reach, grasp, lift, carry, place, release) and
   propose one waypoint, with ordered grasp candidates and one `held` flag.
3. `state.py` prints that as ~20 lines of text: offsets in centimetres, the plan's own sub-stage,
   the grasp options and what blocks them, a short history.
4. `questions.py` asks ten questions about that text in one forward pass; `grounding.py` names
   the target and the destination once per episode.
5. `compose.py` turns the answers into one action chunk and guards `grip` with a stage table.
6. `parse.py` re-derives every label from the text alone, which is what keeps training and
   serving from drifting.

Details and the failures that shaped each of those steps: [docs/DESIGN.md](docs/DESIGN.md).

## Limits

- One suite (LIBERO-Spatial) and one skill (pick and place). No drawer-opening skill.
- Privileged input: exact object poses from the simulator, no vision.
- Grounding is decided by a text rule at serve time; the model's own answer is recorded, not used.
- The carry can stall against the arm's elbow limit; that is the usual lost episode.
- No weights are published yet. Train one with `robojev harvest` → `robojev train` →
  `robojev dagger`. The web demo needs none.
- The simulator is not deterministic run to run: expect ±1–2 episodes per task.
- `robojev run --policy model` cannot yet share an interpreter with the simulator: LIBERO pins
  Python 3.10 and NanoJev's predictor pins 3.14 with torch 2.14. The scripted expert runs closed
  loop today; the learned one loads a real checkpoint and answers all ten questions in one pass,
  but the closed-loop counts above were measured with the two halves in separate processes.
