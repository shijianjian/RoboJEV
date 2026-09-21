# How RoboJEV works, and why it is shaped this way

A 0.6B language model drives a Franka arm in the LIBERO-Spatial simulator by answering ten short
questions about a text description of the scene. This document is the reasoning behind that
sentence: what was tried before, what failed and how it was measured, the principles that came out
of it, and every number the README quotes.

Nothing here is speculation. Where a design choice is defended, it is defended by a count of
completed episodes.

---

## 1. Four designs that did not work

Every one of these was trained and then measured closed-loop — run the policy in the simulator,
count episodes that end in the task's own success predicate. That is the only metric used for
selection anywhere in this project, for reasons that section 8 makes concrete.

| design | closed-loop result |
| --- | --- |
| 0.6B, labels taken from human demonstrations, one 7-way "translate" question | 0/40 |
| 4B backbone, same questions, labels from a goal-conditioned relabelling | 0/30, and it never once closed the gripper |

The diagnosis mattered more than the counts. Eight causes were separated:

**F1 — the grip question asked for a prediction, not a decision.** It was phrased as "closing now
will grasp and lift the object", and the policy acted on `argmax`. A *well-calibrated* answer to a
rarely-true outcome never exceeds 0.5, so the fingers never closed: p(true) peaked at 0.42 and
averaged 0.11 whenever the state said the gripper was open. The model was right and useless.

**F2 — the labels were not a function of the state.** They were what a person did next: one 20 Hz
action, then the executed five-step move. That mixes hesitation, curved paths and multi-axis
motion, none of which the state text can predict. A gradient-boosted tree over eighteen
hand-picked geometric features topped out at 0.59 accuracy on the same labels. This is irreducible
noise, not underfitting, and no backbone size fixes it.

**F3 — grounding and motor control were entangled.** Every decision implicitly required the model
to first resolve "the black bowl between the plate and the ramekin" to one of two identical bowls,
and *then* compare that object's numbers. Ten instructions are nowhere near enough to learn the
first. On a held-out task the arm never got within 25 cm of the bowl.

**F4 — the motion question was a multi-number argmax.** A 7-way single-axis `translate` requires
finding `argmax(|dx|, |dy|, |dz|)`, then its sign, then comparing against a hold threshold. Small
language models are poor at that. The published NanoJev successes are all one question with a few
candidates whose answer is one simple relation.

**F5 — the action vocabulary and the grasp geometry were both wrong.** One axis per 0.25 s makes
paths about three times longer than a diagonal move, so the approach alone ate half the horizon.
And the plan aimed at the *centre* of a ~12 cm bowl with 8 cm fingers: a grasp-feasibility sweep
holds the bowl 16–29 % of the time there, against 73–93 % at the **rim**.

**F6 — the state was memoryless.** The same geometric snapshot needs different answers before and
after a grasp, after a failed attempt, and when the arm is looping. Nothing in the text said which
of those was the case.

**F7 — a bigger model did not help.** The 4B run's dev cross-entropy was flat from step 600. A
better reader of the same noisy labels answering the same badly-shaped question is still stuck.

**F8 — the success metric was wrong.** Every run had been judged by held-out accuracy; the first
closed-loop count arrived after the third training. Accuracy on noisy labels says nothing at all
about completing an episode.

---

## 2. The principles that came out of that

1. **Every label is a deterministic function of the text the model reads.** If a rule cannot
   recover the label from the state string, the model cannot either. This is enforced as a test
   (section 7).
2. **Each question tests one simple relation** — the sign of one number against a tolerance,
   membership in a set, a yes/no about a stated condition — never an argmax across several numbers.
3. **Ask for decisions, not predictions.** A boolean means "do this for the next 0.25 s". Its
   training target is the expert's action, hard 0/1, never an outcome frequency.
4. **Separate grounding from motor control.** Which object the sentence means is decided once,
   written into the state, and then every motor question refers to "the target".
5. **Parallel questions compose into one diagonal move.** The three axes are answered together in
   one forward pass, so an approach costs about a third of the decisions.
6. **Memory is a structured tracker** kept by code from observations; the model reads it.
7. **Code plans, the model judges.** A scripted executor proposes a waypoint and prunes the grasp
   candidates; the model grounds the sentence, judges the stage, decides the fingers and decides
   each axis under its own history.
8. **The scripted expert must finish the task in exactly this vocabulary before anything is
   trained.** If the expert cannot, the vocabulary is wrong and no amount of training will hide it.

---

## 3. The question set

Ten questions, asked **together in one forward pass**, once every five control steps (0.25 s at
20 Hz). Each question's candidates mix only within that question's own set head, which is what
lets "move forward" and "close the gripper" be chosen together without one out-voting the other.

| id | type | candidates | the one relation it tests |
| --- | --- | --- | --- |
| `move_x` | choice | `-`, `hold`, `+` | sign of the waypoint's x offset against the tolerance |
| `move_y` | choice | `-`, `hold`, `+` | same for y |
| `move_z` | choice | `-`, `hold`, `+` | same for z |
| `size_x` | choice | `large`, `medium`, `small` | which band the x offset falls in |
| `size_y` | choice | `large`, `medium`, `small` | same for y |
| `size_z` | choice | `large`, `medium`, `small` | same for z |
| `yaw` | choice | `-`, `hold`, `+` | sign of the wrist yaw error |
| `rim` | choice | `A`–`H` | which of the ordered grasp candidates to stand at |
| `grip` | boolean | close / do not close the fingers | the plan's own finger command for this stage |
| `subgoal` | choice | `reach`, `grasp`, `lift`, `carry`, `place`, `retreat` | which stage the tracker's facts imply |

The step sizes are measured, not chosen: `large` is 5.0 cm, `medium` 1.7 cm, `small` 0.5 cm per
decision. `small` is inside the ±1.5 cm grasp tolerance, which is the property that lets the arm
settle onto a rim point at all.

Two more questions are asked **once per episode**, before any motion, as their own forward pass:

| id | candidates | relation |
| --- | --- | --- |
| `target` | every movable object, each rendered with its position and its spatial relations to the others in words | which object the instruction refers to |
| `destination` | objects and fixtures | where it has to go |

They are re-asked only when the tracker records a failed grasp attempt — on a scene with two
identical bowls, a grasp that never lifted anything is the evidence that the wrong one was named.

Two further questions are *defined* and off by default: a single shared `step` (the ablation the
per-axis sizes replaced), and `yaw` on suites where the expert answers `hold` 100 % of the time.
Which questions a checkpoint was trained on is written into its own manifest and read back at
serve time, by name and never by count.

---

## 4. The state text

One string, about twenty lines. Everything after the task line is computed by the tracker from
observations. Here is a real one, from LIBERO-Spatial task 0 at the first decision:

```
Robot: Franka Panda, gripper-relative frame; x right, y forward, z up; distances in cm.
Task: pick up the black bowl between the plate and the ramekin and place it on the plate
Target: bowl_1 (chosen at t=0). Destination: plate_1.
Subgoal so far: reach (approach). Done: nothing. Decision 1 of 44; 43 left.
Gripper: open 8.0 cm; holding nothing.
Wrist yaw error: +0.0 deg [tolerance 8.0]
Target rests on a flat surface; it is carried 14 cm above the destination.
Grasp candidates (outer finger needs 1.0 cm):
  A: -y, turn +0, room 9990.0 cm -> fits
  B: +y, turn +180, room 9990.0 cm -> fits
  C: -x, turn -90, room 9990.0 cm -> fits
  ... (eight in all, blocked ones marked with the fixture that blocks them)
Waypoint (approach: 8 cm above the rim of bowl_1, -y side): x -6.3, y +15.2, z -7.5   [tolerance 0.3; arrived within 1.3]
  x: -6.3 is outside tolerance -> not aligned
  y: +15.2 is outside tolerance -> not aligned
  z: -7.5 is outside tolerance -> not aligned
Largest remaining offset: 15.2 cm (large range: 3.50 and above).
Step bands: small below 1.17 cm, medium 1.17 to 3.50 cm, large 3.50 and above; one step executes 0.5 / 1.7 / 5.0 cm.
Other objects: bowl_2 x -18.9 y +32.0 z -18.0; cookies_1 x +5.8 y +2.6 z -18.0; ramekin_1 x -19.7 y +18.9 z -18.0; plate_1 x +5.3 y +20.5 z -18.0.
Events: t=0 start, waypoint 18.1 cm away
Attempts: grasps 0, held 0. Closest so far 18.1 cm at t=0. Moved 0.0 cm, net 0.0 cm (not looping).
Last 3: none yet.
```

(The 9990 cm of "room" is the sentinel for a scene with no fixtures in it — this hand-built one
has none. On the drawer task those numbers are real centimetres and most of the candidates read
`blocked by wooden_cabinet_1`.)

Four things about that text are load-bearing.

**Centimetres with a sign, in the gripper's own frame.** The absolute world coordinates and the
quaternions that earlier versions printed carried no decision and cost tokens. Metres to three
decimals also tokenize worse than centimetres to one.

**One line per axis, stating the relation the matching question asks.** Whether those `-> not
aligned` annotations help is an ablation, and it is settled by measurement rather than taste.

**The waypoint is named in the state.** It is produced by code from the committed target — the rim
point is the object's position plus a measured radius toward the hand, at a measured height — so
the label for each axis is a function of text the model can actually see.

**Counters, not prose.** The events line, the attempt counts and the last-three history are what
turn a memoryless snapshot into something that can distinguish "approaching" from "looping".

The whole worst case, state plus the longest question and candidate, measures **682 tokens** under
the Qwen3-0.6B tokenizer, against a 1024 budget. The predictor raises rather than truncating an
oversized path, so that margin is a real constraint and is re-measured whenever the text changes.

---

## 5. The plan executor

The scripted expert used to be a hand-written if/elif phase machine, and every stage had grown its
own arrival test, its own timeout and its own recovery — each added after one observed failure.
Three ideas were implemented several times over, differently: *am I making progress*, *is the
object held*, *what do I try next*.

It is now one small executor over stages as data:

```
approach → descend → close → lift (check: held) → carry (check: held)
        → lower → release → retreat → done
```

Each stage carries a name, the sub-goal it belongs to, the finger command its own plan makes, a
goal point, an arrival test, an optional dwell, an optional outcome `check`, and where to go when
it is done, blocked, or the check fails. Nothing in the executor names a bowl.

**One progress rule.** Distance to the stage's goal must improve by 0.3 cm within three decisions.
No improvement means the stage is *blocked*. There is no other timeout anywhere — the old "out of
time, pretend we arrived" was the direct cause of every failure in one measured run, because in
the descend stage it closed the fingers 2 cm too high.

**One ordered list of alternatives.** The two ends of the wrist's closing axis crossed with the
grasp heights, ranked, with candidates that a fixture blocks marked as blocked. A blocked stage or
a failed check marks the current candidate as tried, backs off to the hover, and takes the next.
When the list is exhausted the executor accepts the best pose it reached — the one surviving
"proceed anyway", and only after every alternative was tried rather than after a clock ran out.

**One definition of "held".** Fingers closed *and* the object has risen with the hand. The three
previous, slightly different definitions used to disagree, and one of them printed "holding bowl_1"
about an empty hand.

The fixtures matter here. The simulator's furniture — a cabinet, a stove — has no observable pose
in the task definition, so earlier versions could not see it and planned straight into it: on the
drawer task the fingers stopped against the drawer's own wall on every control step of every
descent, and nothing in the state said the wall was there. Fixtures are now read live as
**oriented** collision boxes, per geom rather than per fixture. Both of those are necessary: a
drawer's bounding box contains the bowl inside it, and this cabinet is yawed 155°, so the
axis-aligned hull of its thin 21 cm drawer floor is a 21 cm cube that fills the space the fingers
have to enter — with it, all eight grasp candidates read as blocked at 0.0 cm of room.

---

## 6. The grip guard

The `grip` answer is the one the policy cannot afford to get wrong. Motion answers tolerate 5–10 %
error, because a wrong step is undone on the next decision. The finger latch does not: corrupting
it at 10 % takes the scripted expert from 19/20 to 7/20, while exempting the latch from the same
corruption leaves it at 11–13/20.

So the composer guards it, structurally rather than with a tuned probability threshold. A change
of the fingers is applied only in a sub-goal whose own plan makes that change:

| sub-goal | may close | may open |
| --- | --- | --- |
| `reach` | no | yes |
| `grasp` | yes | yes |
| `lift` | no | no |
| `carry` | no | no |
| `place` | no | yes |
| `retreat` | no | yes |

A refused change is recorded (`grip_latch.refused`), so a replay says whether the model closed the
gripper or the guard declined to let it. An operator's explicit override outranks the guard: "force
open" has to open the gripper, not be argued with by a serving rule.

---

## 7. Grounding, and the label invariant

**Grounding.** The `target`/`destination` questions are rendered in the *camera* frame the
instructions are written in — never concatenated with the motor state, which is gripper-relative
and mirrored left for right, so "the bowl on the left" read against the motor state resolves the
wrong bowl. Alongside the model's own answer there is a text rule that reads the relation words
back out of the same request the model was given: no privileged data, no task definition, nothing
the model could not see.

At serve time the **text rule** is what is committed, and the model's answer is recorded beside it.
That is a measurement, not a preference. The rule scores 610/610 on `target` and 515/520 on
`destination` across every LIBERO suite. The first trained checkpoint grounded *confidently* wrong
— it picked the other bowl at p 0.957 on task 0 — and a probability threshold cannot catch a
confident error, so it is not a guard. A later checkpoint improved held-out grounding from 0.78 to
0.95, but returning the decision to the model needs its own closed-loop measurement, which has not
been run. There is no mode that skips grounding: every motor question says "the target".

**The label invariant.** A parser re-derives every label from the state text alone, and
`parse(serialise(state)) == gold` at **100 %** over every harvested row. This is the property that
keeps the training side and the serving side from drifting: a training row and an inference request
are built from the same numbers by the same code, and if a rule cannot recover a label from the
string, the question is not answerable from the string and is rewritten.

---

## 8. How a checkpoint is selected

**By counting completed episodes. Never by accuracy.** Held-out accuracy on the earlier label sets
sat comfortably above the class marginal while the closed-loop count was 0/40. A model that answers
demonstration states well and collapses on its own states will look good by every offline metric
there is.

The pipeline:

1. `robojev harvest` rolls the scripted expert out with ε-noise on the motion answers (never on
   `grip`), so the rows cover states a perfect controller never visits.
2. `robojev train` runs NanoJev's own trainer — full fine-tune, hard cross-entropy, upstream's own
   hyperparameters — and writes the harvest manifest beside the weights. A checkpoint that cannot
   say what step size its answers mean will not load.
3. `robojev dagger` rolls the trained policy out and relabels the states **it** visited, through
   the same episode loop a graded run uses. Otherwise the round's central claim is false.
4. The checkpoint is chosen by the closed-loop count over a fixed set of episodes.

One measurement from step 3 is worth stating: on the states the trained model itself visits, its
motion answers and its stage equal the expert's almost exactly (0 of 1,638 differ in one round,
4 of 1,538 in another, both on size questions). That is precisely the property the failed versions
lacked.

---

## 9. The measured results

LIBERO-Spatial, ten tasks, 220-step horizon, argmax selection.

**The scripted expert** (the label source, answering this exact question set):

| | episodes |
| --- | --- |
| all ten tasks × ten start states | **90/100** |
| median decisions used | 28 of 44 |

**The released model** (run 8, revision `bb7ba09784d7`), closed loop over the same 50 episodes,
measured twice:

| | episodes |
| --- | --- |
| ten tasks × five start states | **45/50** and **46/50** |
| the drawer task alone | 3–4 of 5 |

**Against the hosted official Jev**, zero-shot on the same ten episodes, through the same harness —
the same questions, the same state text, the same plan, the same guard, the same composer, so the
only thing that differs is which model answers:

| | episodes |
| --- | --- |
| hosted Jev, zero-shot | 3/10 |
| this model | 8/10 |

**The designs that came before**, for scale: 0/40 and 0/30.

---

## 10. What this does not do

- **One suite, one skill.** LIBERO-Spatial, and pick-and-place. There is no drawer-*opening* skill:
  that is a new stage table (reach the handle, grasp, pull along the drawer axis, release) plus
  fresh expert rollouts and a retrain, not a recording.
- **Privileged input.** The policy reads the simulator's exact object poses and the fixtures'
  collision geometry. There is no vision anywhere in it. Whatever this says about language-model
  control, it says nothing about perception.
- **Grounding is still the text rule's at serve time.** The model's answer is recorded, not used.
  Section 7 says why, and what it would take to change.
- **The carry stall.** The usual lost episode is the arm's elbow hard against its joint limit
  during a long carry: with the elbow locked the hand slides on a shell, and each single-axis
  command executes as something else. Rising is the one direction that both executes and unlocks
  it, which is the basis of the recovery the executor has — but it does not always fire in time.
- **The drawer task is the hard one**, and its remaining failures are horizon failures: the grasp
  itself works once the fixtures are visible, but the episode runs out of decisions.
- **Simulator noise.** These numbers are not exactly reproducible run to run. Expect ±1–2 episodes
  per task; that is why the 50-episode count is quoted twice rather than averaged into one number.
- **No published weights.** The checkpoint is 2.4 GB and is not in this repository. The pipeline in
  section 8 is how to make one.
