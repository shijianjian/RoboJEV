"""The scripted policy: the expert behind it, the questions it declares, and one real episode.

`robojev.policy.ExpertPolicy` has no model, no upstream and no heavy imports, so unlike the
learned one it can be driven whole on a CPU -- `act()` on hand-built states, and the grounding
and tracker plumbing that every engine shares.
"""
from __future__ import annotations

import numpy as np
import pytest

from robojev import expert as expert_mod
from robojev import policy as policy_mod
from robojev import roles as roles_mod


@pytest.fixture(scope="module")
def srv():
    return policy_mod


# The real LIBERO-Spatial task 0 scene, measured in the checkout under MUJOCO_GL=egl and recorded
# in the plan 9a task-2 report: five objects, all resting at z = 0.970 on a table top at 0.90.
# The two fixtures (`wooden_cabinet_1`, `flat_stove_1`) have no observables and so are absent --
# which is the whole reason the stove and cabinet phrases need a fallback.
SCENE = {
    "akita_black_bowl_1": {"pos": [-0.063, 0.202, 0.970], "quat": [0.0, 0.0, 0.0, 1.0]},
    "akita_black_bowl_2": {"pos": [-0.189, 0.320, 0.970], "quat": [0.0, 0.0, 0.0, 1.0]},
    "cookies_1": {"pos": [0.058, 0.026, 0.970], "quat": [0.0, 0.0, 0.0, 1.0]},
    "glazed_rim_porcelain_ramekin_1": {"pos": [-0.197, 0.189, 0.970], "quat": [0.0, 0.0, 0.0, 1.0]},
    "plate_1": {"pos": [0.053, 0.205, 0.970], "quat": [0.0, 0.0, 0.0, 1.0]},
}
BOWL_1, BOWL_2 = "akita_black_bowl_1", "akita_black_bowl_2"
PLATE = "plate_1"
BETWEEN = "pick up the black bowl between the plate and the ramekin and place it on the plate"
NEXT_TO = "pick up the black bowl next to the cookie box and place it on the plate"
STOVE = "pick up the black bowl on the stove and place it on the plate"
CABINET = "pick up the black bowl on the wooden cabinet and place it on the plate"

#: An 8-d proprio vector at `pos`, top-down orientation, fingers wide open.
def state(pos, width: float = 0.08) -> np.ndarray:
    return np.array([*pos, np.pi, 0.0, 0.0, width / 2, -width / 2], dtype=np.float32)


def _obs(pos, width=0.08):
    return {"state": state(pos, width), "privileged": SCENE}


# --------------------------------------------------------------------- reading the instruction

def test_the_spatial_phrase_stops_before_the_destination_clause(srv):
    """Every instruction ends "and place it on the plate", so a phrase read off the whole
    sentence would find the plate in all four and turn each into a two-anchor phrase."""
    assert roles_mod.spatial_phrase(NEXT_TO).strip() == "next to the cookie box"
    assert roles_mod.spatial_phrase(BETWEEN).strip() == "between the plate and the ramekin"
    assert roles_mod.spatial_phrase(STOVE).strip() == "on the stove"
    assert roles_mod.spatial_phrase(CABINET).strip() == "on the wooden cabinet"


def test_between_the_plate_and_the_ramekin_is_the_midpoint_not_the_nearer_anchor(srv):
    """The measured case the midpoint rule exists for: on task 0 the ramekin is 3 mm *nearer*
    the wrong bowl, and the midpoint of the two anchors is 17x nearer the right one."""
    ramekin = np.array(SCENE["glazed_rim_porcelain_ramekin_1"]["pos"])
    one, two = np.array(SCENE[BOWL_1]["pos"]), np.array(SCENE[BOWL_2]["pos"])
    assert np.linalg.norm(two - ramekin) < np.linalg.norm(one - ramekin)   # the naive rule is wrong

    roles = policy_mod._scene_roles(SCENE, BETWEEN, [0.0, 0.0, 1.1])
    assert roles["target"] == BOWL_1
    assert roles["destination"] == PLATE
    assert roles["anchors"] == ["plate", "ramekin"]
    assert "midpoint" in roles["rule"]


def test_next_to_the_cookie_box_is_the_bowl_nearest_the_cookies(srv):
    roles = policy_mod._scene_roles(SCENE, NEXT_TO, [0.0, 0.0, 1.1])
    assert roles["target"] == BOWL_1 and roles["destination"] == PLATE
    assert roles["anchors"] == ["cookie"]
    assert roles["rule"] == "the bowl nearest the cookie"


@pytest.mark.parametrize("instruction, word", [(STOVE, "stove"), (CABINET, "cabinet")])
def test_a_fixture_anchor_falls_back_to_the_bowl_farther_from_the_plate(srv, instruction, word):
    """The stove and the cabinet get body ids but no `<name>_pos` observable, so there is
    nothing to measure a distance to."""
    roles = policy_mod._scene_roles(SCENE, instruction, [0.0, 0.0, 1.1])
    assert roles["target"] == BOWL_2      # farther from plate_1 than bowl_1 is
    assert roles["destination"] == PLATE
    assert roles["rule"] == f"the {word} has no observable pose: the bowl farther from the plate"


def test_an_unrecognised_sentence_takes_the_bowl_nearest_the_end_effector(srv):
    roles = policy_mod._scene_roles(SCENE, "do something with the bowl", [-0.19, 0.32, 1.1])
    assert roles["target"] == BOWL_2 and roles["destination"] == PLATE
    assert roles["anchors"] == []
    assert "nearest the end effector" in roles["rule"]


def test_a_scene_with_no_plate_or_no_bowl_falls_back_to_nearest_and_farthest(srv):
    scene = {k: v for k, v in SCENE.items() if k not in (PLATE,)}
    roles = policy_mod._scene_roles(scene, BETWEEN, [0.058, 0.026, 1.0])
    assert roles["target"] == "cookies_1"                     # nearest the end effector
    assert roles["destination"] == BOWL_2                     # farthest from it
    assert roles["rule"].startswith("fallback:")


def test_an_empty_scene_is_an_error_not_a_guess(srv):
    with pytest.raises(policy_mod.PolicyError, match="empty"):
        policy_mod._scene_roles({}, BETWEEN, [0.0, 0.0, 1.0])


# ------------------------------------------------------------------------- the server's contract

# ------------------------------------------------------ the question set, and the expert

#: The active set, by name rather than by count, and read from the module that owns it: `yaw`
#: joined it when the drawer task turned the wrist on (note
#: `2026-09-21-stage9-drawer-task.md`) and this server serves whatever that set is.
V2_QIDS = ("move_x", "move_y", "move_z", "size_x", "size_y", "size_z", "yaw", "rim", "grip",
           "subgoal")


def v2_server(srv, **kwargs):
    return policy_mod.ExpertPolicy("libero_spatial", seed=0, **kwargs)


def test_a_bare_server_is_the_expert_and_says_so_in_its_revision(srv):
    """The baseline that completes 85/100 episodes, and a revision string that is not the one the
    retired decider's recorded runs carry: two different policies must never compare against each
    other under one name in `runs.checkpoint_revision`."""
    assert policy_mod.ExpertPolicy("libero_spatial").revision == policy_mod.EXPERT_REVISION


def test_an_explicit_revision_is_a_deliberate_pin_and_is_kept(srv):
    """The default names this file's own behaviour, because there are no weights to name. An
    explicitly different one is somebody saying "record it as this", and is left alone."""
    assert policy_mod.ExpertPolicy("libero_spatial", revision="pinned").revision == "pinned"


def test_v2_describe_advertises_the_v2_question_set(srv):
    """The app's `DecisionPanel` is generic over whatever groups `protocol.questions` declares,
    so advertising `ACTIVE_QIDS` is the whole of what makes the eight v2 groups render in the
    Decide tab. Nothing in the app changes."""
    d = v2_server(srv).describe()
    p = d["protocol"]

    assert p["questions_version"] == "v2"
    assert list(p["questions"]) == list(V2_QIDS)
    assert set(p["questions"]["move_x"]["criteria"]) == {"-", "hold", "+"}
    assert set(p["questions"]["size_z"]["criteria"]) == {"large", "medium", "small"}
    assert p["questions"]["grip"]["type"] == "boolean"
    assert set(p["questions"]["subgoal"]["criteria"]) == {
        "reach", "grasp", "lift", "carry", "place", "retreat"}
    # `yaw` is asked since the drawer task: the wrist's sizes used to be scaled by what the
    # controller is commanded rather than by what it executes, so one `large` answer turned it
    # 3.6 degrees and the expert could only ever answer `hold`. The shared `step` is the ablation
    # that is still defined and not asked.
    assert set(p["questions"]["yaw"]["criteria"]) == {"-", "hold", "+"}
    assert set(p["questions"]["rim"]["criteria"]) == set("ABCDEFGH")
    assert "step" not in p["questions"]
    # Still a decision policy with privileged state and a five-step chunk.
    assert d["family"] == "decision" and p["privileged_state"] is True
    assert p["chunk_size"] == 5 and p["execute_steps"] == 5


def test_v2_is_served_at_delta_t_one(srv):
    """Spec §7 ruling (e), and the single most expensive number in this file to get wrong: one
    decision executes 0.050 m per unit of δ, so composing the expert's answers at the
    demonstrations' 0.3536 executes a fifth of every move they mean and the episode cannot
    finish (note §2 -- 24 of 44 decisions on the approach alone)."""
    p = v2_server(srv).describe()["protocol"]

    assert p["delta_t"] == 1.0
    assert p["cm_per_unit"] == pytest.approx(5.0)
    assert p["step_sizes_cm"]["large"] == pytest.approx(5.0)
    assert p["step_sizes_cm"]["small"] == pytest.approx(0.5)


def test_the_checkpoint_block_says_there_are_no_weights(srv):
    """No repo, because there is nothing published; a revision that names this file's behaviour,
    because that is the only identity a controller has. A record that said otherwise would let a
    scripted run be compared against a trained one under one name."""
    assert v2_server(srv).describe()["checkpoint"] == {"repo": None, "revision": "scripted-v2"}
    assert v2_server(srv, revision="mine").revision == "mine"


def test_v2_describe_reports_the_tracker_settings(srv):
    """The v2 state text is what a model reads, so the knobs that change a rendered byte are in
    the protocol block for the same reason v1's `memory` settings are."""
    memory = v2_server(srv).describe()["protocol"]["memory"]

    assert memory["memory_rule"].startswith("v2-")
    assert memory["every"] == 5
    assert list(memory["qids"]) == list(V2_QIDS)


def test_the_policy_states_how_it_wants_to_be_run(srv):
    """`protocol()` is what the episode loop obeys, and it is derived from the same facts
    `describe()` reports rather than typed out twice."""
    policy = v2_server(srv)
    protocol = policy.protocol()
    assert protocol.family == "decision" and protocol.privileged_state is True
    assert protocol.chunk_size == protocol.execute_steps == 5
    # No settling steps: this policy reads the simulator's own object poses, which are right from
    # the first frame, rather than an image of a scene still dropping into place.
    assert protocol.wait_steps == 0 == policy.describe()["protocol"]["wait_steps"]
    assert protocol.max_steps == 220


def _v2_episode(srv, decisions=10, instruction=BETWEEN):
    """Drive the v2 path over a descending approach, the way a console would step it."""
    server = v2_server(srv)
    server.reset(instruction)
    out = []
    for index in range(decisions):
        z = 1.25 - 0.03 * index
        out.append(server.act(_obs([0.10 - 0.015 * index, 0.202, z])))
    return server, out


def test_a_whole_v2_episode_answers_only_the_v2_questions(srv):
    """Ten decisions through the real `act`: every group is one of the eight, every choice is one
    of that question's candidates, and nothing from v1's vocabulary leaks in."""
    from robojev.questions import candidates as v2_candidates

    _, steps = _v2_episode(srv)

    for actions, decisions in steps:
        assert actions.shape == (5, 7)
        assert np.allclose(actions, actions[0])        # one decision held for the whole chunk
        assert set(decisions) == {*V2_QIDS, "meta"}
        for qid in V2_QIDS:
            answer = decisions[qid]
            assert set(answer) == {"probabilities", "choice", "overridden"}
            assert answer["choice"] in v2_candidates(qid)
            assert answer["overridden"] is False


def test_the_v2_baseline_grounds_once_with_the_rule_and_the_tab_can_see_it(srv):
    """The scripted baseline has no grounding *forward* to run, but it has the same grounding
    *step*: the roles are committed into the tracker once, at the first decision, and reported in
    `meta.grounding` in the shape RoboJEV's v2 server reports its own -- so the tab shows a
    target and a destination for the baseline too, and `source` says which of the two named it.
    """
    server, steps = _v2_episode(srv, decisions=3)
    first = steps[0][1]["meta"]["grounding"]
    assert first["source"] == "rule"
    assert first["sources"] == {"target": "rule", "destination": "rule"}
    assert first["target"] == BOWL_1 and first["destination"] == PLATE
    assert first["regrounded"] is False and first["decision"] == 0
    # The same text rule the learned server commits with -- read off the same request, in the
    # camera frame the sentence is written in -- and no model half, because there is no model.
    assert first["mode"] == "rule" and first["model"] is None and first["model_agrees"] is None
    # Once per episode and not again: every later decision refers to what was committed.
    assert [d["meta"]["grounding"] for _, d in steps[1:]] == [None, None]
    for _, decisions in steps:
        assert decisions["meta"]["target"] == BOWL_1
        assert decisions["meta"]["destination"] == PLATE
        assert decisions["meta"]["target_rule"] == "rule"
    assert server.tracker.committed()["source"] == "rule"
    assert "Target: " in steps[0][1]["meta"]["state"]


def test_the_baseline_commits_exactly_what_the_text_rule_resolves(srv):
    """Not `scene_roles` any more: the same `resolve_request` over the same `grounding_request`
    the learned server commits with, so the baseline and RoboJEV under `--ground rule` name the
    same object from the same sentence."""
    from robojev import grounding as G

    server, steps = _v2_episode(srv, decisions=1)
    request = G.grounding_request(BETWEEN, SCENE)
    names = G.grounding_names(SCENE)
    picked = G.resolve_request(request)
    assert server.tracker.target == names[picked["target"]]
    assert server.tracker.destination == names[picked["destination"]]
    assert steps[0][1]["meta"]["target_rule"] == "rule"


def test_a_failed_attempt_makes_the_baseline_name_its_roles_again(srv):
    """The same rule the learned server re-grounds on (`TrackerV2.needs_regrounding`): a grasp
    that never lifted anything is the evidence that the wrong object was named."""
    server, steps = _v2_episode(srv, decisions=2)
    server.tracker._grasp_attempts = server.tracker._committed_attempts + 1
    server.tracker._rose_at = None
    _, decisions = server.act(_obs([0.07, 0.202, 1.19]))
    assert decisions["meta"]["grounding"]["regrounded"] is True
    assert decisions["meta"]["grounding"]["source"] == "rule"


def test_the_v2_distributions_are_hard_because_the_expert_is_a_controller(srv):
    """It does not have a belief about `move_z`, it has an answer. A fabricated 0.85 is a number
    a console reads as confidence, and `expert.probabilities` (the declared-noise recipe) is for
    the training target's soft arm, not for serve time."""
    _, steps = _v2_episode(srv, decisions=3)

    for _, decisions in steps:
        for qid in V2_QIDS:
            probabilities = decisions[qid]["probabilities"]
            assert sum(probabilities.values()) == pytest.approx(1.0)
            assert set(probabilities.values()) == {0.0, 1.0}
            assert probabilities[decisions[qid]["choice"]] == 1.0


def test_the_v2_meta_carries_the_version_the_subgoal_the_offsets_and_the_sizes(srv):
    """Plan 9c's requirement on the served `decisions.meta`."""
    _, steps = _v2_episode(srv, decisions=4)
    meta = steps[-1][1]["meta"]

    assert meta["questions_version"] == "v2"
    assert meta["subgoal"] in {"reach", "grasp", "lift", "carry", "place", "retreat"}
    assert meta["subgoal"] == steps[-1][1]["subgoal"]["choice"]
    assert len(meta["offset_cm"]) == 3 and len(meta["waypoint_cm"]) == 3
    assert meta["sizes"] == {axis: steps[-1][1][f"size_{axis}"]["choice"] for axis in "xyz"}
    assert meta["target"] == BOWL_1 and meta["destination"] == PLATE
    assert meta["delta_t"] == 1.0


def test_the_v2_state_text_is_the_v2_tracker_s(srv):
    """A console (or a DAgger relabelling) reading `meta.state` off a baseline run and off a
    RoboJEV run has to be reading the same text about the same scene, which is why the baseline
    keeps a tracker it does not itself read."""
    _, steps = _v2_episode(srv, decisions=4)
    text = steps[-1][1]["meta"]["state"]

    assert text.splitlines()[0].startswith("Robot: Franka Panda")
    assert "Task: " + BETWEEN in text
    assert "Waypoint (" in text and "Subgoal so far: " in text
    # 1-based: the tracker counts the decision it has just observed.
    assert "Decision 4 of 44" in text
    # v1's block is metres and prose; v2's is centimetres and counters.
    assert "quat=" not in text


def test_v2_reset_clears_the_tracker_and_the_expert_s_carry(srv):
    """An `attempts` counter carried into the next episode starts its grasp 1.5 cm high, and a
    tracker that remembers the last bowl renders a `Target:` line about a scene that is gone."""
    server, _ = _v2_episode(srv, decisions=5)
    assert server.tracker.decisions == 5
    assert server.tracker.target == BOWL_1

    server.reset(BETWEEN)
    assert server.tracker.decisions == 0
    assert server.tracker.target is None
    assert server.phase == expert_mod.new_phase()
    assert server.released is False

    _, decisions = server.act(_obs([0.30, 0.202, 1.20]))
    assert "Decision 1 of 44" in decisions["meta"]["state"]


def test_a_v2_override_forces_the_answer_and_is_recorded(srv):
    server = v2_server(srv)
    server.reset(BETWEEN)
    actions, decisions = server.act(_obs([0.30, 0.202, 1.20]),
                                    overrides={"move_y": "+", "grip": True})

    assert decisions["move_y"]["choice"] == "+" and decisions["move_y"]["overridden"] is True
    assert decisions["grip"]["choice"] == "true" and decisions["grip"]["overridden"] is True
    assert actions[0][1] > 0 and actions[0][6] == pytest.approx(1.0)
    assert decisions["meta"]["overridden"] == ["move_y", "grip"]


@pytest.mark.parametrize("override", [{"move_y": "north"}, {"translate": "+y"}])
def test_a_v2_override_naming_something_this_set_does_not_have_is_an_error(srv, override):
    """A typo silently ignored looks like the policy disagreeing with the operator -- and
    `translate` is v1's question, which is the typo a console switched between the two would
    actually make."""
    server = v2_server(srv)
    server.reset(BETWEEN)
    with pytest.raises(policy_mod.PolicyError):
        server.act(_obs([0.30, 0.202, 1.20]), overrides=override)


def test_act_returns_the_chunk_and_the_decisions_together(srv):
    """One call, two things, and the loop carries the second onto every frame of the first."""
    server = v2_server(srv)
    server.reset(BETWEEN)
    actions, decisions = server.act(_obs([0.30, 0.202, 1.20]))

    assert actions.shape == (5, 7)
    assert decisions["move_x"]["choice"] == "-"
    assert decisions["meta"]["questions_version"] == "v2"


def test_v2_act_without_privileged_state_says_so(srv):
    server = v2_server(srv)
    server.reset(BETWEEN)
    with pytest.raises(policy_mod.PolicyError, match="privileged"):
        server.act({"state": state([0.0, 0.0, 1.1])})


# ------------------------------------------------------------------------------ the real thing


@pytest.mark.sim
def test_a_whole_libero_episode_through_the_scripted_policy(srv):
    """The end-to-end claim, against the simulator: LIBERO-Spatial task 0, driven entirely
    through `ExpertPolicy.act` -- the same call the episode loop makes -- ends in LIBERO's own
    success predicate inside the suite's horizon.

    The hand-built states above are only worth anything if they are the states `LiberoEnv`
    produces, and this is what says so. ~40 s on a CPU, most of it the env build.
    """
    from robojev.envs.libero import LiberoEnv

    server = v2_server(srv)
    env = LiberoEnv("libero_spatial", 0, render_size=128)
    try:
        obs = env.reset(0)
        server.reset(env.instruction)
        steps, success = 0, False
        while steps < 220 and not success:
            actions, _ = server.act({"state": env.state_vector(obs), "privileged": env.privileged(obs)})
            for action in actions:
                if steps >= 220:
                    break
                result = env.step(action)
                obs, steps = result.obs, steps + 1
                if result.done:
                    success = True
                    break
    finally:
        env.close()
    assert success, f"no success in {steps} steps"
