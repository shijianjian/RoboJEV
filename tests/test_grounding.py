"""RoboJEV v2's grounding data, checked without a simulator wherever it can be.

`robojev.grounding` has three halves that fail in three different ways, so they are
tested separately: the BDDL reader (a parser, tested against a fixture file written here), the
relation words (arithmetic on coordinates, tested against scenes whose answers are obvious by
construction), and the row builder (a training row, tested against *exactly* the rules a
harvested v1 row is held to -- the same vendored `validate_training_row` the harvest and DAgger
tests use, because a row the trainer rejects is worth nothing however well grounded it is).

The `sim`-marked test at the end is the only one that builds a real LIBERO environment: two
tasks, one init state each, end to end from the benchmark API to a validated row.
"""
from __future__ import annotations

import inspect
import types

import pytest

from robojev import grounding as G
from test_dataset import validate_training_row   # tests/ is on sys.path: no __init__.py, by design

# ------------------------------------------------------------------------------- the BDDL reader

FIXTURE_BDDL = """
(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language pick up the black bowl between the plate and the ramekin and place it on the plate)
    (:regions
      (plate_region
          (:target main_table)
          (:ranges (
              (0.05 0.19 0.07 0.21)
            )
          )
      )
      (cook_region
          (:target flat_stove_1)
      )
    )

  (:fixtures
    main_table - table
    flat_stove_1 - flat_stove
  )

  (:objects
    akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl
    plate_1 - plate
  )

  (:obj_of_interest
    akita_black_bowl_1
    plate_1
  )

  (:init
    (On akita_black_bowl_1 main_table_plate_region)
  )

  (:goal
    (And (On akita_black_bowl_1 plate_1))
  )

)
"""

TOGGLE_BDDL = FIXTURE_BDDL.replace("(And (On akita_black_bowl_1 plate_1))",
                                   "(And (Turnon flat_stove_1))")
REGION_GOAL_BDDL = FIXTURE_BDDL.replace("(And (On akita_black_bowl_1 plate_1))",
                                        "(And (On akita_black_bowl_1 flat_stove_1_cook_region))")
TABLE_GOAL_BDDL = FIXTURE_BDDL.replace("(And (On akita_black_bowl_1 plate_1))",
                                       "(And (On akita_black_bowl_1 main_table_plate_region))")


@pytest.fixture
def fixture_file(tmp_path):
    path = tmp_path / "pick_up_the_black_bowl.bddl"
    path.write_text(FIXTURE_BDDL, encoding="utf-8")
    return path


def test_parsing_a_bddl_file_reads_the_sentence_the_objects_and_the_goal(fixture_file):
    task = G.parse_bddl(fixture_file.read_text(encoding="utf-8"), suite="libero_spatial",
                        task_index=0, name="demo")
    assert task.instruction == ("pick up the black bowl between the plate and the ramekin and "
                                "place it on the plate")
    assert task.objects == {"akita_black_bowl_1": "akita_black_bowl",
                            "akita_black_bowl_2": "akita_black_bowl", "plate_1": "plate"}
    assert task.fixtures == {"main_table": "table", "flat_stove_1": "flat_stove"}
    assert task.obj_of_interest == ("akita_black_bowl_1", "plate_1")
    assert task.goal == (G.Predicate("on", ("akita_black_bowl_1", "plate_1")),)
    # `(And …)` is flattened away, so a one-literal goal and a three-literal one read the same.
    assert task.init == (G.Predicate("on", ("akita_black_bowl_1", "main_table_plate_region")),)


def test_a_region_in_the_goal_resolves_to_the_entity_that_owns_it():
    """`(On bowl flat_stove_1_cook_region)` places the bowl on the *stove*: the destination the
    question asks for is an entity, and the region is only how the BDDL spells a place on it."""
    task = G.parse_bddl(REGION_GOAL_BDDL)
    assert task.regions["flat_stove_1_cook_region"] == "flat_stove_1"
    assert G.destination_of(task) == ("flat_stove_1", "destination of the place goal")


def test_the_task_kinds_and_which_of_them_carry_a_destination():
    place = G.parse_bddl(FIXTURE_BDDL)
    assert G.task_kind(place) == "place"
    assert G.target_of(place)[0] == "akita_black_bowl_1"
    assert G.destination_of(place)[0] == "plate_1"

    toggle = G.parse_bddl(TOGGLE_BDDL)
    assert G.task_kind(toggle) == "toggle"
    # Nothing is carried, so there is no destination -- but the fixture the goal changes is still
    # a target, which is why `target`'s candidates include the non-table fixtures.
    assert G.target_of(toggle)[0] == "flat_stove_1"
    assert G.destination_of(toggle) == (None, "no place goal")

    on_table = G.parse_bddl(TABLE_GOAL_BDDL)
    assert G.destination_of(on_table) == (None, "destination is a table region")


# --------------------------------------------------------------------------- the relation words

def test_between_is_reported_only_for_the_object_actually_on_the_line():
    """`a` is on the segment from `left` to `right`; `far` is the same distance along it and 20 cm
    off, which is what the tolerance is for."""
    phrases = G.relations({
        "left_1": (0.0, -20.0, 0.0),
        "a_1": (0.0, 0.0, 0.0),
        "right_1": (0.0, 20.0, 0.0),
        "far_1": (25.0, 0.0, 0.0),
    })
    assert "between left_1 and right_1" in phrases["a_1"]
    assert not [p for p in phrases["far_1"] if p.startswith("between")]


def test_left_right_front_and_behind_are_signed_and_a_tie_says_neither():
    """`+y` is right and `+x` is front (the viewer's frame), and two objects within `LEVEL_CM` of
    each other on an axis are level rather than one being left of the other."""
    phrases = G.relations({
        "a_1": (0.0, 0.0, 0.0),
        "right_1": (0.0, 30.0, 0.0),
        "front_1": (30.0, 0.0, 0.0),
        "tie_1": (0.0, G.LEVEL_CM - 0.5, 0.0),
    })
    assert "left of right_1" in phrases["a_1"]
    assert "right of a_1" in phrases["right_1"]
    assert "behind front_1" in phrases["a_1"]
    assert "in front of a_1" in phrases["front_1"]
    assert not [p for p in phrases["tie_1"] if p.startswith(("left of a_1", "right of a_1"))]


def test_on_top_of_and_inside_come_from_the_heights():
    phrases = G.relations({
        "box_1": (0.0, 0.0, 0.0),
        "bowl_1": (1.0, 1.0, 8.0),      # above the box, within its footprint
        "tray_1": (40.0, 0.0, 0.0),
        "milk_1": (40.5, 0.5, 0.5),     # co-located with the tray: inside it
    })
    assert "on top of box_1" in phrases["bowl_1"]
    assert "inside tray_1" in phrases["milk_1"]
    assert "on top of bowl_1" not in phrases["box_1"]


def test_a_fixtures_own_regions_are_what_on_the_stove_and_in_the_drawer_mean():
    """Measured from LIBERO-Spatial task 4: the stove's body is 15 cm from its burner and the
    cabinet's is 14 cm from its top drawer, so the bodies alone put the bowl "on the stove" next
    to nothing. The `(:regions …)` sites are the reference points the sentences name."""
    positions = {"bowl_1": (-26.2, -13.1, 11.0),   # on the stove's burner
                 "bowl_2": (8.5, -13.0, 25.1),     # in the cabinet's top drawer
                 "bowl_3": (1.8, -27.7, 33.2),     # standing on the cabinet's lid
                 "stove_1": (-40.0, -13.4, 0.5),
                 "cabinet_1": (3.7, -26.0, 0.5)}
    anchors = {"stove_1": [("cook", "region", (-26.1, -13.4, 0.5))],
               "cabinet_1": [("top", "region", (9.5, -12.8, 19.1)),
                             ("middle", "region", (3.0, -26.9, 12.2)),
                             ("top", "side", (2.8, -27.3, 22.7))]}
    phrases = G.relations(positions, anchors)
    assert "on top of stove_1" in phrases["bowl_1"]
    assert "in the top part of cabinet_1" in phrases["bowl_2"]
    assert "on top of cabinet_1" in phrases["bowl_3"]


def test_the_rank_phrases_name_a_position_among_the_identical_objects():
    positions = {"bowl_1": (0.0, -20.0, 0.0), "bowl_2": (0.0, 0.0, 0.0),
                 "bowl_3": (0.0, 20.0, 0.0)}
    ranks = G.extremes(positions, {"bowl": list(positions)})
    assert "the leftmost of the 3 bowls" in ranks["bowl_1"]
    assert "the rightmost of the 3 bowls" in ranks["bowl_3"]
    assert "the middle of the 3 bowls left to right" in ranks["bowl_2"]


# ------------------------------------------------------------------------------- the short ids

def _entities():
    return [
        G.Entity("akita_black_bowl_1", "akita_black_bowl", "object", (-6.0, 20.0, 7.0)),
        G.Entity("akita_black_bowl_2", "akita_black_bowl", "object", (-19.0, 32.0, 7.0)),
        G.Entity("plate_1", "plate", "object", (5.0, 20.0, 7.0)),
        G.Entity("glazed_rim_porcelain_ramekin_1", "glazed_rim_porcelain_ramekin", "object",
                 (-20.0, 19.0, 7.0)),
        G.Entity("main_table", "table", "fixture", (0.0, 0.0, 0.0)),
        G.Entity("flat_stove_1", "flat_stove", "fixture", (-42.0, -13.0, 0.5)),
    ]


def test_the_id_numbering_is_seeded_and_shuffles_inside_a_noun_group():
    """The BDDL names the object of interest `…_1` in almost every LIBERO-Spatial task, so a row
    that always called it `bowl_1` would be solvable by reading the digit."""
    import random

    seen = {tuple(sorted(G.assign_ids(_entities(), random.Random(seed)).items()))
            for seed in range(20)}
    assert len(seen) == 2, "both numberings of the two bowls must occur"
    once = G.assign_ids(_entities(), random.Random(7))
    again = G.assign_ids(_entities(), random.Random(7))
    assert once == again
    assert {once["akita_black_bowl_1"], once["akita_black_bowl_2"]} == {"bowl_1", "bowl_2"}
    assert once["plate_1"] == "plate_1" and once["flat_stove_1"] == "stove_1"


# ---------------------------------------------------------------------------- the rows and split

def _scene(suite, task_index, init_index, instruction, entities=None):
    entities = entities if entities is not None else _entities()
    return {"suite": suite, "task_index": task_index, "task_name": f"task{task_index}",
            "instruction": instruction, "init_index": init_index,
            "entities": [{"name": e.name, "category": e.category, "kind": e.kind,
                          "pos": list(e.pos) if e.pos else None} for e in entities]}


INSTRUCTION = ("pick up the black bowl between the plate and the ramekin and place it on the "
               "plate")


def _scenes():
    return [_scene("libero_spatial", 0, i, INSTRUCTION) for i in range(3)] + [
        _scene("libero_spatial", 8, i, "pick up the black bowl next to the plate and place it on "
                                       "the plate") for i in range(3)] + [
        _scene("libero_90", 5, i, "put the black bowl on the plate") for i in range(3)]


def _rows(**kwargs):
    scenes = _scenes()
    task = G.parse_bddl(FIXTURE_BDDL)
    rows, report = G.build_rows(scenes, task_lookup=lambda record: task, **kwargs)
    G.attach_scene(rows, scenes)
    return rows, report


def test_a_grounding_row_validates_against_the_same_rules_a_harvested_row_does():
    rows, _ = _rows()
    assert rows
    for row in rows:
        validate_training_row(row)
        assert set(row["gold"]) == {"target", "destination"}
        assert row["gold_probs"]["target"][row["gold"]["target"]] == 1.0
        assert sum(row["gold_probs"]["target"].values()) == 1.0


def test_the_state_names_every_candidate_and_the_frame_it_is_written_in():
    rows, _ = _rows()
    state = rows[0]["state"]
    assert "Frame: the work surface" in state
    assert f"Task: {INSTRUCTION}" in state
    for sid in rows[0]["questions"]["target"]["criteria"]:
        assert f"  {sid} (" in state
    # The table is never a candidate: it is the frame, not an object anybody grounds to.
    assert "table_1" not in rows[0]["questions"]["destination"]["criteria"]


def test_the_gold_follows_the_shuffled_id_rather_than_the_bddl_digit():
    """The same task at three init states gets three independent numberings, and the gold points
    at whichever id the row actually gave the BDDL's object of interest."""
    rows, _ = _rows()
    for row in rows:
        ids = row["metadata"]["ids"]
        assert row["gold"]["target"] == ids[row["metadata"]["gold_names"]["target"]]
        assert row["gold"]["destination"] == ids[row["metadata"]["gold_names"]["destination"]]


def test_the_split_is_by_instruction_so_no_sentence_is_in_both_halves():
    rows, _ = _rows()
    by_instruction = {}
    for row in rows:
        by_instruction.setdefault(row["metadata"]["source_group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_instruction.values())
    states = {row["split"]: set() for row in rows}
    for row in rows:
        states[row["split"]].add(row["metadata"]["source_group_id"])
    assert not (states.get("train", set()) & states.get("test", set()))


def test_libero_spatial_tasks_8_and_9_are_always_held_out():
    scenes = _scenes()
    splits = G.split_map(scenes)
    held = G.instruction_key(scenes[3]["instruction"])       # libero_spatial task 8
    assert splits[held] == "test"


def test_the_resolver_reads_the_between_phrase_out_of_the_relation_words():
    """Gate G5(a) in miniature: the two bowls are identical and only the relation words separate
    them, so a resolver that gets this right is reading the scene text."""
    rows, _ = _rows()
    row = next(r for r in rows if r["metadata"]["instruction"] == INSTRUCTION)
    assert G.resolve_row(row, "target") == row["gold"]["target"]
    assert G.resolve_row(row, "destination") == row["gold"]["destination"]


def test_the_written_rows_carry_no_private_probe_keys(tmp_path):
    import json

    rows, _ = _rows()
    path = G.write_rows(rows, tmp_path / "rows.jsonl")
    for line in path.read_text(encoding="utf-8").splitlines():
        assert not [k for k in json.loads(line)["metadata"] if k.startswith("_")]


def test_the_linear_probe_learns_the_training_split():
    """Not a quality claim -- a claim that the probe is wired up: with `target` decided by the
    relation words the scorer has as features, it must at least fit the rows it trained on."""
    rows, _ = _rows()
    for row in rows:
        row["split"] = "train"
    probe = G.linear_probe(rows, "target", steps=200)
    accuracy, _ = probe["accuracy"](rows)
    assert accuracy >= 0.9


# ------------------------------------------------------------------- the serving-time request

#: A live observation's shape: `{name: {"pos": (x, y, z), "quat": …}}`, world metres
#: (`robojev.envs.libero.LiberoEnv.privileged`). These are LIBERO-Spatial task 0's rough poses.
PRIVILEGED = {
    "akita_black_bowl_1": {"pos": [-0.06, 0.20, 0.97]},
    "akita_black_bowl_2": {"pos": [-0.19, 0.32, 0.97]},
    "plate_1": {"pos": [0.05, 0.20, 0.97]},
    "glazed_rim_porcelain_ramekin_1": {"pos": [-0.20, 0.19, 0.97]},
}


def test_the_serving_request_is_nanojevs_three_keys_and_nothing_else():
    """`validate_request` rejects a fourth key with a 400, so the ids mapping a server needs is a
    function beside the request rather than a field inside it."""
    request = G.grounding_request(INSTRUCTION, PRIVILEGED)
    assert list(request) == ["states"] and len(request["states"]) == 1
    assert list(request["states"][0]) == ["id", "state", "questions"]
    assert list(request["states"][0]["questions"]) == ["target", "destination"]


def test_a_served_scene_and_a_training_row_are_the_same_text_for_the_same_scene():
    """The pin. A checkpoint is trained on `build_rows`' text and asked `grounding_request`'s, so
    the two are one renderer: same frame, same candidate lines, same relation words, same
    criteria. The ids are handed over rather than re-drawn only because a training row shuffles
    its numbering per row on purpose (`assign_ids`)."""
    frame = G.serving_frame(PRIVILEGED)
    entities = [G.Entity(name, G._category_of(name), "object",
                         G._to_frame(body["pos"], frame))
                for name, body in sorted(PRIVILEGED.items())]
    scene = _scene("libero_spatial", 0, 0, INSTRUCTION, entities=entities)
    task = G.parse_bddl(FIXTURE_BDDL)
    rows, _ = G.build_rows([scene], task_lookup=lambda record: task)
    row = rows[0]

    request = G.grounding_request(INSTRUCTION, PRIVILEGED, ids=row["metadata"]["ids"])
    served = request["states"][0]
    assert served["state"] == row["state"]
    assert served["questions"] == row["questions"]


def test_the_frame_is_the_objects_own_surface_so_a_scene_reads_in_centimetres():
    """No table body is observable, and none is needed: z is the median resting height, which is
    what the relation words are written against, and x/y are the world origin the table sits at."""
    assert G.serving_frame(PRIVILEGED) == (0.0, 0.0, 0.97)
    for entity in G.serving_entities(PRIVILEGED):
        assert entity.pos[2] == 0.0
        assert entity.kind == "object"
    # The category word is the asset's type, not its name with a digit on the end.
    assert [e.category for e in G.serving_entities(PRIVILEGED)][:1] == ["akita_black_bowl"]
    assert "(black bowl)" in G.grounding_request(INSTRUCTION, PRIVILEGED)["states"][0]["state"]


def test_the_short_ids_map_back_to_the_scenes_own_object_names_and_do_not_move():
    """A chosen candidate has to become the name `TrackerV2.commit` is given, and a re-ground
    after a failed grasp must be written in the same alphabet as the first one."""
    names = G.grounding_names(PRIVILEGED)
    assert set(names.values()) == set(PRIVILEGED)
    assert names == G.grounding_names(PRIVILEGED)
    ids = G.serving_ids(PRIVILEGED)
    assert {sid: names[sid] for sid in names} == {ids[name]: name for name in ids}
    assert ids["plate_1"] == "plate_1"
    assert {ids["akita_black_bowl_1"], ids["akita_black_bowl_2"]} == {"bowl_1", "bowl_2"}


def test_a_live_scene_has_no_fixtures_and_no_region_sites_and_says_so_by_omission():
    """What a served request cannot carry: LIBERO builds observables for movable objects only, so
    a cabinet is not a candidate and its drawer sites cannot produce "in the top part of". The
    rule fallback is what a server does about it (`meta.grounding.source`)."""
    assert all(e.anchors == () for e in G.serving_entities(PRIVILEGED))
    state = G.grounding_request(INSTRUCTION, PRIVILEGED)["states"][0]["state"]
    assert "Fixtures that cannot be picked up" not in state


# ------------------------------------------------------- the rule, at serve time (task 7 follow-up)

def test_the_rule_reads_a_live_request_and_picks_the_bowl_the_sentence_names():
    """Run 6's failure in one test: its grounding forward committed `bowl_2` at p 0.957 on this
    scene, whose `bowl_1` line says "between plate_1 and ramekin_1". A probability threshold does
    not catch that; reading the relation words does."""
    request = G.grounding_request(INSTRUCTION, PRIVILEGED)
    names = G.grounding_names(PRIVILEGED)
    picked = G.resolve_request(request)
    assert names[picked["target"]] == "akita_black_bowl_1"
    assert names[picked["destination"]] == "plate_1"
    # And it is reading the text, not the ids: the winning candidate's own line says so.
    line = next(l for l in request["states"][0]["state"].splitlines()
                if l.strip().startswith(picked["target"] + " "))
    assert "between plate_1 and ramekin_1" in line


def test_the_scene_is_read_back_out_of_the_state_text_and_nothing_else():
    """The rule a server runs must read exactly what the model reads -- no privileged dict, no
    BDDL, no second source of positions that could disagree with the sentence."""
    request = G.grounding_request(INSTRUCTION, PRIVILEGED)
    instruction, entities, phrases = G.scene_from_request(request)
    assert instruction == INSTRUCTION
    assert set(entities) == set(G.grounding_names(PRIVILEGED))
    bowl = next(sid for sid, e in entities.items()
                if e.pos == G._to_frame(PRIVILEGED["akita_black_bowl_1"]["pos"],
                                        G.serving_frame(PRIVILEGED)))
    assert entities[bowl].category == "black bowl" and entities[bowl].kind == "object"
    assert "between plate_1 and ramekin_1" in phrases[bowl]
    # An unresolvable question is `None`, never a guess dressed as an answer.
    assert G.resolve_request(request, qids=["target"]) == {"target": bowl}


#: LIBERO-Spatial task 6 at init 0, in world metres -- the scene whose *live* phrasing broke the
#: rule. The env says "the black bowl next to the cookie box" where the BDDL says "the akita black
#: bowl next to the cookies box", and `cookie box` matches the cookie box's own category word
#: exactly, so the box survives `_matching` and is itself "next to bowl_1".
TASK_6 = {
    "akita_black_bowl_1": {"pos": [0.137, -0.070, 0.97]},
    "akita_black_bowl_2": {"pos": [-0.263, -0.135, 1.01]},
    "cookies_1": {"pos": [0.073, 0.033, 0.97]},
    "plate_1": {"pos": [0.069, 0.194, 0.97]},
    "glazed_rim_porcelain_ramekin_1": {"pos": [-0.213, 0.189, 0.97]},
}


@pytest.mark.parametrize("instruction", [
    "pick up the black bowl next to the cookie box and place it on the plate",
    "Pick the akita black bowl next to the cookies box and place it on the plate",
])
def test_a_relation_names_the_candidate_on_its_left_and_the_neighbour_on_its_right(instruction):
    """"The **bowl** next to the **cookie box**" reads the scene line "next to bowl_1" from one
    side only. Read from the other it answers with the box, which is what the rule did under the
    environment's own phrasing of LIBERO-Spatial task 6 until the relation was cut in two."""
    request = G.grounding_request(instruction, TASK_6)
    names = G.grounding_names(TASK_6)
    picked = G.resolve_request(request)
    assert names[picked["target"]] == "akita_black_bowl_1"
    assert names[picked["destination"]] == "plate_1"


#: LIBERO-Spatial task 3 **after LIBERO's ten settling steps** -- the observation the server
#: actually grounds on. The bowl has settled onto the flat cookie box, so their centres are
#: within `INSIDE_XY_CM`/`INSIDE_DZ_CM` and the scene line reads "inside cookies_1" where the
#: unsettled frame said "on top of cookies_1".
TASK_3_SETTLED = {
    "akita_black_bowl_1": {"pos": [0.068, 0.024, 0.978]},
    "akita_black_bowl_2": {"pos": [0.022, -0.284, 1.187]},
    "cookies_1": {"pos": [0.068, 0.024, 0.970]},
    "plate_1": {"pos": [0.070, 0.190, 0.963]},
    "glazed_rim_porcelain_ramekin_1": {"pos": [-0.214, 0.211, 0.960]},
}


@pytest.mark.parametrize("instruction", [
    "pick up the black bowl on the cookie box and place it on the plate",
    "Pick the akita black bowl on the cookies box and place it on the plate",
])
def test_the_picking_clause_never_feeds_the_destination(instruction):
    """Closed-loop failure, LIBERO-Spatial task 3: the server committed `cookies_1` as the
    *destination* and the arm put the bowl back on the box. Two `on` phrases in one sentence --
    one saying which bowl, one saying where it goes -- and a resolver free to cut the sentence
    anywhere answered the second question with the first phrase. The join says where the clauses
    are, so each question reads only its own."""
    request = G.grounding_request(instruction, TASK_3_SETTLED)
    names = G.grounding_names(TASK_3_SETTLED)
    picked = G.resolve_request(request)
    assert names[picked["destination"]] == "plate_1"
    assert names[picked["target"]] == "akita_black_bowl_1"
    # And the target is *resolved*, not guessed: the sentence's "on" reads the settled scene's
    # "inside cookies_1" as well as an unsettled scene's "on top of cookies_1".
    instruction_body, _ = G.split_clauses(instruction)
    pick, place = G.placing_clauses(instruction_body)
    assert place == "the plate"
    _text, entities, phrases = G.scene_from_request(request)
    assert "inside cookies_1" in phrases[picked["target"]]
    assert G.resolve(pick, sorted(entities), entities,
                     {sid: e.pos for sid, e in entities.items()}, phrases,
                     strict=True) == picked["target"]


def test_the_served_sentence_is_the_task_language_and_not_the_scene_label():
    """LIBERO's task *name* carries a scene label outside LIBERO-Spatial and its `language` does
    not, and the label is not cosmetic: it hides the lead verb `split_clauses` reads, so every
    "turn on the stove and put the moka pot on it" row grounds to the wrong entity. The three
    pairs below are `benchmark.get_benchmark_dict()[suite]().get_task(i)`, read off the real
    benchmark."""
    assert G.served_instruction(
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
    ) == "put both the alphabet soup and the tomato sauce in the basket"
    assert G.served_instruction("KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet") == \
        "close the top drawer of the cabinet"
    # LIBERO-Spatial's names *are* the sentence, which is why the label never showed there.
    spatial = "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
    assert G.served_instruction(spatial) == spatial.replace("_", " ")
    # A sentence that merely starts with a capital is not a label.
    assert G.served_instruction("Pick_the_black_bowl") == "Pick the black bowl"


def test_a_sentence_with_no_placing_join_still_tries_every_cut():
    """`placing_clauses` is `None` for the sentences the old search exists for -- "put the bowl on
    the plate" has no join, and "stack the bowl in the middle on the bowl at the front" is two
    noun phrases of the same noun."""
    assert G.placing_clauses("the black bowl on the plate") is None
    assert G.placing_clauses("the bowl in the middle on the bowl at the front") is None
    assert G.placing_clauses("the black bowl and place it on the plate") == (
        "the black bowl", "the plate")
    assert G.placing_clauses("the moka pot and put it in the microwave") == (
        "the moka pot", "the microwave")


def test_the_serving_path_and_the_row_path_are_the_same_rule():
    """One implementation, two entry points (`resolve_scene`): a request built from the row's own
    text resolves to exactly what `resolve_row` resolves."""
    rows, _ = _rows()
    for row in rows:
        request = {"states": [{"id": row["id"], "state": row["state"],
                               "questions": row["questions"]}]}
        served = G.resolve_request(request)
        for qid in row["questions"]:
            assert served[qid] == G.resolve_row(row, qid), (row["id"], qid)


@pytest.mark.sim
def test_the_serving_resolver_reproduces_gate_g5a_over_every_cached_scene():
    """The measurement, over the 630 rows of `SCENE_CACHE`: the serving entry agrees with
    `resolve_row` on **every** question, and scores G5(a)'s 610/610 `target` and 515/520
    `destination` against the BDDL's own answer. `sim`-marked because it needs the cached scenes
    and LIBERO's bddl files, not because it builds a simulator."""
    scenes = G.load_scenes()
    if not scenes:
        pytest.skip("no scene cache: build it with `python -m robojev.grounding scenes`")
    rows, _ = G.build_rows(scenes)
    G.attach_scene(rows, scenes)
    agree = {"target": [0, 0], "destination": [0, 0]}
    gold = {"target": [0, 0], "destination": [0, 0]}
    for row in rows:
        served = G.resolve_request({"states": [{"id": row["id"], "state": row["state"],
                                                "questions": row["questions"]}]})
        for qid in row["questions"]:
            agree[qid][1] += 1
            agree[qid][0] += int(served[qid] == G.resolve_row(row, qid))
            gold[qid][1] += 1
            gold[qid][0] += int(served[qid] == row["gold"][qid])
    assert agree["target"][0] == agree["target"][1] == 610
    assert agree["destination"][0] == agree["destination"][1] == 520
    assert gold["target"] == [610, 610] and gold["destination"] == [515, 520]


# ----------------------------------------------------------------------------------- the real thing

@pytest.mark.sim
def test_two_real_tasks_build_rows_end_to_end():
    """One LIBERO-Spatial task and one LIBERO-Object task, from the benchmark API to a validated
    row -- the only test here that constructs a simulator."""
    scenes = (G.scene_records("libero_spatial", 0, n_init=1)
              + G.scene_records("libero_object", 0, n_init=1))
    rows, report = G.build_rows(scenes)
    G.attach_scene(rows, scenes)
    assert len(rows) == 2
    for row in rows:
        validate_training_row(row)
        assert row["gold"]["target"] in row["questions"]["target"]["criteria"]
        assert row["metadata"]["ids"][row["metadata"]["gold_names"]["target"]] == \
            row["gold"]["target"]
    assert report["rows_per_suite"] == {"libero_spatial": 1, "libero_object": 1}
    # The frame's z is the scene's own work surface, so the objects resting on it read as 0.
    for scene in scenes:
        heights = sorted(e["pos"][2] for e in scene["entities"]
                         if e["kind"] == "object" and e["pos"])
        assert heights[len(heights) // 2] == 0.0


# ------------------------------------------------------- the served phrasing, and the repeats
#
# The gap this closes, measured on run 6: the rows were written with the BDDL's own
# `(:language …)` -- "Pick the akita black bowl next to the cookies box" -- and the server asks
# with the simulator's sentence -- "pick up the black bowl next to the cookie box". Run 6's
# grounding head names the same wrong bowl at p ~ 0.96 under the served wording on both task-0
# init states, and its held-out `target` accuracy is 0.78.

BDDL_SENTENCE = "Pick the akita black bowl next to the cookies box and place it on the plate"
SERVED_TASK_NAME = "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
SERVED_SENTENCE = "pick up the black bowl next to the cookie box and place it on the plate"


def test_the_served_sentence_is_the_task_name_with_its_underscores_back():
    """LIBERO's `task.language` -- what `libero_utils.get_libero_env` returns and what becomes
    `LiberoEnv.instruction` -- is the BDDL file's stem de-underscored. Derived rather than
    imported so the row builder needs no simulator; verified against the benchmark for
    LIBERO-Spatial 0, 6, 8 and 9."""
    assert G.served_instruction(SERVED_TASK_NAME) == SERVED_SENTENCE
    assert G.served_instruction(
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
    ) == ("pick up the black bowl between the plate and the ramekin and place it on the plate")
    # And it is *not* the BDDL's own sentence, which is the whole reason for the function.
    assert G.served_instruction(SERVED_TASK_NAME) != BDDL_SENTENCE


def _phrasing_scene(instruction=BDDL_SENTENCE, task_name=SERVED_TASK_NAME, entities=None):
    scene = _scene("libero_spatial", 6, 0, instruction, entities=entities)
    scene["task_name"] = task_name
    return scene


def test_a_row_can_be_written_under_either_phrasing_and_says_which():
    """Same scene, same candidates, same gold -- one sentence apart."""
    scene = _phrasing_scene()
    task = G.parse_bddl(FIXTURE_BDDL)
    bddl, _ = G.build_rows([scene], task_lookup=lambda r: task, phrasing="bddl")
    served, _ = G.build_rows([scene], task_lookup=lambda r: task, phrasing="served")
    assert f"Task: {BDDL_SENTENCE}" in bddl[0]["state"]
    assert f"Task: {SERVED_SENTENCE}" in served[0]["state"]
    assert bddl[0]["gold"] == served[0]["gold"]
    assert bddl[0]["questions"] == served[0]["questions"]
    assert bddl[0]["split"] == served[0]["split"]
    assert bddl[0]["metadata"]["phrasing"] == "bddl"
    assert served[0]["metadata"]["phrasing"] == "served"
    assert served[0]["metadata"]["bddl_instruction"] == BDDL_SENTENCE
    assert served[0]["metadata"]["instruction"] == SERVED_SENTENCE
    with pytest.raises(ValueError, match="phrasing must be one of"):
        G.build_rows([scene], task_lookup=lambda r: task, phrasing="invented")


def test_a_served_scene_and_a_training_row_are_the_same_text_under_the_served_phrasing():
    """The pin, extended to the wording a server actually asks with. A checkpoint trained on
    `build_rows`' text and asked `grounding_request`'s must be reading one renderer under *both*
    sentences, or the augmentation has introduced the very gap it is closing."""
    frame = G.serving_frame(PRIVILEGED)
    entities = [G.Entity(name, G._category_of(name), "object", G._to_frame(body["pos"], frame))
                for name, body in sorted(PRIVILEGED.items())]
    scene = _phrasing_scene(entities=entities)
    task = G.parse_bddl(FIXTURE_BDDL)
    for phrasing, sentence in (("bddl", BDDL_SENTENCE), ("served", SERVED_SENTENCE)):
        rows, _ = G.build_rows([scene], task_lookup=lambda r: task, phrasing=phrasing)
        row = rows[0]
        served = G.grounding_request(sentence, PRIVILEGED,
                                     ids=row["metadata"]["ids"])["states"][0]
        assert served["state"] == row["state"], phrasing
        assert served["questions"] == row["questions"], phrasing


def test_the_two_phrasings_of_a_task_are_one_held_out_thing():
    """A sentence in train under one wording and in test under another is leakage wearing a
    paraphrase, so the split is drawn over groups of co-occurring keys rather than sentences."""
    scenes = [_phrasing_scene(), _scene("libero_spatial", 8, 0, "pick up the black bowl next to "
                                        "the plate and place it on the plate")]
    splits = G.split_map(scenes, key_of=G.scene_keys)
    for scene in scenes:
        keys = G.scene_keys(scene)
        assert len({splits[k] for k in keys}) == 1, (scene["task_index"], keys)
    # Spatial 8 is held out by construction, under both of its wordings.
    for key in G.scene_keys(scenes[1]):
        assert splits[key] == "test"


def test_the_default_split_map_is_the_one_already_on_disk():
    """`key_of=None` must be the single-key rule byte for byte: the harvested `grounding.jsonl`
    was written with it, and a re-run that quietly re-drew the split would move rows between
    train and test under a manifest that says it did not."""
    scenes = _scenes()
    assert G.split_map(scenes) == G.split_map(scenes, key_of=None)
    forced = G.instruction_key("pick up the black bowl next to the plate and place it on the "
                               "plate")
    assert G.split_map(scenes)[forced] == "test"


def test_repeated_copies_of_a_scene_differ_in_their_candidate_numbering():
    """The point of repeating: a model that has seen one scene under several numberings cannot
    answer it by memorising a digit (`assign_ids` shuffles inside a noun group, off the seed)."""
    scene = _phrasing_scene()
    task = G.parse_bddl(FIXTURE_BDDL)
    alphabets = []
    for index in range(6):
        rows, _ = G.build_rows([scene], seed=G.SPLIT_SEED + index, task_lookup=lambda r: task,
                               id_suffix=f":g{index}")
        assert rows[0]["id"].endswith(f":grounding:g{index}")
        assert rows[0]["state_id"] == rows[0]["id"]
        alphabets.append(tuple(sorted(rows[0]["metadata"]["ids"].items())))
    assert len(set(alphabets)) > 1, "every repeat drew the same numbering"


# -------------------------------------------------- the settled scene (the second train/serve gap)


class _SettleEnv:
    """The three calls `scene_records` makes on an env, and a counter on each."""

    num_init_states = 5

    def __init__(self):
        self.steps = 0
        self.resets = []
        self.read_at = None

    def dummy_action(self):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

    def reset(self, index):
        self.resets.append(index)
        self.steps = 0
        return {"t": 0}

    def step(self, action):
        assert list(action) == self.dummy_action(), action
        self.steps += 1
        return types.SimpleNamespace(obs={"t": self.steps})


def test_settling_runs_the_episodes_own_no_op_action_the_episodes_own_number_of_times():
    """`robojev.episode.run_episode` runs `protocol.wait_steps` dummy actions before the policy's
    first `act`, so a server grounds against a scene that has dropped and settled. A cache
    captured straight after `set_init_state` describes a different one -- on LIBERO-Spatial task 3
    the bowl's relation flips from "on top of cookies_1" to "inside cookies_1"."""
    from robojev import episode as protocol_mod

    assert G.SETTLED_STEPS == protocol_mod.for_suite("libero_spatial").wait_steps == 10
    env = _SettleEnv()
    obs = G.settle(env, env.reset(0))
    assert env.steps == G.SETTLED_STEPS
    assert obs == {"t": 10}
    # And it is the *observation after* the settling that a caller reads a pose from.
    assert G.settle(env, env.reset(0), 0) == {"t": 0}
    assert G.settle(env, env.reset(0), 3) == {"t": 3}


def test_a_scene_record_says_what_it_was_captured_after():
    """Two caches exist and they describe different scenes, so a record carries the number rather
    than leaving it to a file's modification time."""
    assert "settled_steps" in inspect.signature(G.scene_records).parameters
    assert inspect.signature(G.scene_records).parameters["settled_steps"].default == G.SETTLED_STEPS
    assert inspect.signature(G.build_scene_cache).parameters["path"].default == \
        G.SETTLED_SCENE_CACHE
