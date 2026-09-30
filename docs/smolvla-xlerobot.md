# SmolVLA on the simulated XLeRobot

SmolVLA (`lerobot/smolvla_base`) fine-tuned on scripted demonstrations of the XLeRobot putting a
bowl on a plate, and benchmarked in closed loop on layouts it never saw. The policy sees only what
the real robot has: the head and two wrist cameras (256×256), the 16 joint fields, the instruction.
It outputs the real driver's 16 action fields (LeRobot `xlerobot_2wheels`: arm joints in degrees,
grippers 0–100; head and base fields held at 0 because the base is parked).

## Pieces (`robojev/smolvla/`)

| file | what |
|---|---|
| `scene.py` | the XLeRobot from its URDF at a lowered table (0.74 m) pressed to the cart; LIBERO's bowl, plate and ramekin; three cameras; seeded layouts in the acting arm's 12–16 cm reach band |
| `env.py` | the task at 20 Hz in driver units; success = the bowl resting on the plate, released |
| `expert.py` | the scripted demonstrator (grasps the bowl's +y wall with the arm on the bowl's side) |
| `demos.py` | demonstrations → a LeRobot dataset (successful ones only) |
| `bench.py` | closed-loop benchmark of a SmolVLA checkpoint or the expert on frozen layouts |

Simulation needs MuJoCo and LIBERO's object files (not LIBERO's code, which pins an older Python
than LeRobot): set `ROBOJEV_LIBERO_OBJECTS` to LIBERO's `assets/stable_scanned_objects`.

## Commands

```bash
# environment: Python 3.12, lerobot[smolvla,dataset,training]==0.6.1, mujoco==3.3.7
python -m robojev.smolvla.demos --out runs/demos --episodes 500 --workers 16
lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/xlerobot_bowl --dataset.root=runs/demos/merged \
  --rename_map='{"observation.images.head": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2", "observation.images.right_wrist": "observation.images.camera3"}' \
  --batch_size=64 --steps=20000 --output_dir=runs/smolvla --policy.push_to_hub=false
python -m robojev.smolvla.bench --policy runs/smolvla/checkpoints/last/pretrained_model \
  --out runs/bench --episodes 50 --n-action-steps 10 --workers 16
```

Benchmark layouts are seeds 1 000 000–1 000 049; `demos.py` refuses seeds from 1 000 000 up.

## Results (50 benchmark layouts, 20 s limit)

| run | success | left | right |
|---|---|---|---|
| scripted expert, current layouts | 98% | 100% | 96% |
| SmolVLA, 20k steps, replanning every 5 actions | 62% | 84% | 40% |
| SmolVLA, 20k steps, replanning every 10 actions | 60% | 76% | 44% |

Data: 500 layouts tried, 486 kept (left 280 of 280, right 206 of 220), 132 588 frames at 20 Hz,
batch 64, 20k steps from `lerobot/smolvla_base` (about 2.2 h on one H200).

Changes that mattered, in order: the bowl renders dark (LIBERO's reflective material and its
visible collision boxes made it silver, like the white ramekin); the instruction names the arm;
both arms start folded onto the cart (the URDF zero pose hid a right-side bowl from the head
camera); the plate never spawns overlapping the bowl.

Open: the right arm often reaches over the bowl and moves on without closing on it.
