"""`robojev.dataset` -- the shape of a harvest directory.

The layout, the manifest, the task splits, the per-row random stream and the row check every
harvester's rows have to pass. Both harvesters (`v2.rollout`, `dagger`) write this directory, so
what is pinned here is what they agree on rather than what either of them happens to do.

`validate_training_row` lives here too, and `test_expert_rollout.py` and `test_grounding_v2.py`
import it from here: it is a vendored transcription of upstream's own validator, and one copy of
it is what keeps the two row writers honest against the same rules.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from robojev import dataset as D


# ------------------------------------------------------------------------------------- splits

def test_the_last_two_tasks_are_the_held_out_pair():
    splits = D.resolve_splits(D.Splits(tasks=tuple(range(10))))
    assert (splits.dev_task, splits.test_task) == (8, 9)
    assert D.split_for(9, splits) == "test"
    assert D.split_for(8, splits) == "dev"
    assert D.split_for(0, splits) == "train"


def test_fewer_than_three_tasks_get_no_held_out_task_at_all():
    """With two tasks the rule would leave `train` empty, and the trainer rejects an empty
    split outright. A one-task proof harvest is a file to inspect, not one to train on."""
    for tasks in ((0,), (0, 4)):
        splits = D.resolve_splits(D.Splits(tasks=tasks))
        assert (splits.dev_task, splits.test_task) == (None, None)
        assert all(D.split_for(t, splits) == "train" for t in tasks)


def test_a_partial_harvest_resolves_against_every_task_on_disk():
    """Re-running task 3 of a finished ten-task harvest must label its rows `train`, because 8
    and 9 are the held-out pair already on disk."""
    splits = D.resolve_splits(D.Splits(tasks=(3,)), known_tasks=list(range(10)))
    assert (splits.dev_task, splits.test_task) == (8, 9)
    assert D.split_for(3, splits) == "train"


def test_an_explicit_pair_is_never_re_derived():
    splits = D.resolve_splits(D.Splits(tasks=tuple(range(10)), dev_task=1, test_task=2))
    assert (splits.dev_task, splits.test_task) == (1, 2)


# -------------------------------------------------------------------------------- the row rng

def test_the_row_stream_depends_on_the_row_and_not_on_the_run():
    """`--tasks 3` alone and `--tasks 0,1,2,3` must produce byte-identical rows for task 3, which
    is what lets a harvest be resumed, extended or re-run one task at a time -- and what makes
    `--jobs 8` identical to `--jobs 1`."""
    a = D.row_rng(17, 3, 0, 40).integers(0, 2 ** 62)
    b = D.row_rng(17, 3, 0, 40).integers(0, 2 ** 62)
    assert a == b
    assert a != D.row_rng(17, 3, 0, 45).integers(0, 2 ** 62)
    assert a != D.row_rng(17, 4, 0, 40).integers(0, 2 ** 62)
    assert a != D.row_rng(18, 3, 0, 40).integers(0, 2 ** 62)


def test_the_row_stream_is_a_numpy_generator():
    assert isinstance(D.row_rng(1, 2, 3, 4), np.random.Generator)


# ------------------------------------------------------------------------------ the row check

def _row(**over) -> dict:
    row = {
        "id": "expert:libero_spatial:0:0:0:0",
        "questions": {"move_x": {"type": "choice", "criteria": {"-": "", "hold": "", "+": ""}},
                      "grip": {"type": "boolean"}},
        "gold_probs": {"move_x": {"-": 0.0, "hold": 0.0, "+": 1.0},
                       "grip": {"false": 1.0, "true": 0.0}},
    }
    row.update(over)
    return row


def test_a_distribution_that_misses_a_candidate_is_refused():
    row = _row()
    del row["gold_probs"]["move_x"]["-"]
    with pytest.raises(AssertionError, match="gold_probs covers"):
        D.check_probs(row)


def test_a_distribution_that_does_not_sum_to_one_is_refused():
    row = _row()
    row["gold_probs"]["move_x"]["+"] = 0.5
    with pytest.raises(AssertionError, match="sums to"):
        D.check_probs(row)


def test_a_boolean_question_is_checked_against_its_two_strings():
    row = _row()
    row["gold_probs"]["grip"] = {"0": 0.5, "1": 0.5}
    with pytest.raises(AssertionError, match="gold_probs covers"):
        D.check_probs(row)


def test_a_valid_row_passes():
    D.check_probs(_row())


# -------------------------------------------------------------------------------- the manifest

def test_the_manifest_of_an_empty_directory_is_none(tmp_path):
    assert D.read_manifest(tmp_path) is None


def test_a_corrupt_manifest_reads_as_absent_rather_than_raising(tmp_path):
    """It is a file this code wrote and can write again; refusing to harvest over it helps
    nobody."""
    (tmp_path / D.MANIFEST).write_text("{not json")
    assert D.read_manifest(tmp_path) is None


def test_a_manifest_round_trips(tmp_path):
    (tmp_path / D.MANIFEST).write_text(json.dumps({"suite": "libero_spatial"}))
    assert D.read_manifest(tmp_path) == {"suite": "libero_spatial"}


def test_the_default_out_root_is_under_robojev_home(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOJEV_HOME", str(tmp_path / "home"))
    assert D.out_root("robojev-v2", "libero_spatial") == (
        tmp_path / "home" / "data" / "robojev-v2" / "libero_spatial")


def test_the_timestamp_is_utc_and_second_resolution():
    assert D.now().endswith("+00:00")


def test_bump_and_totals_count_a_directory_rather_than_one_run():
    records = {0: {"labels": {"+": 2, "-": 1}}, 1: {"labels": {"+": 3}}}
    assert D.totals(records, lambda r: r["labels"]) == {"+": 5, "-": 1}


# ------------------------------------------------------------- the trainer's own row rules

REQUIRED = ("id", "state_id", "family_id", "split", "state", "questions", "gold_probs",
            "gold_probs_kind", "gold_label_kind", "metadata")
PROBS_KINDS = {"deterministic_truth", "programmatic_conditional_distribution", "optimal_action_policy"}
LABEL_KINDS = {"observed_outcome", "deterministic_truth", "reference_argmax_compatibility",
               "unspecified_compatibility_label", "hard_gold_unspecified", "unobserved"}


def validate_training_row(row: dict) -> None:
    """A vendored transcription of `validate_training_row`'s constraints (note §3's list).

    Vendored rather than imported: NanoJev is not a package and its scripts import torch at
    module scope (note §5), so the real function cannot run in the CPU suite. The `gpu` proof
    runs the genuine one over a real harvested file; this is the same rules, kept here so a row
    that would fail there fails in seconds instead.
    """
    for key in REQUIRED:
        assert key in row, f"missing {key}"
    assert row["split"] in D.SPLITS
    assert set(row["gold_probs"]) == set(row["questions"])
    for qid, probs in row["gold_probs"].items():
        q = row["questions"][qid]
        expected = {"false", "true"} if q["type"] == "boolean" else set(q["criteria"])
        assert set(probs) == expected
        assert abs(math.fsum(probs.values()) - 1.0) <= 1e-8
        assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert set(row["gold_probs_kind"]) == set(row["questions"])
    assert set(row["gold_probs_kind"].values()) <= PROBS_KINDS
    assert set(row["gold_label_kind"]) == set(row["questions"])
    assert set(row["gold_label_kind"].values()) <= LABEL_KINDS
    assert set(row["gold"]) <= set(row["questions"])
    assert row["metadata"].get("source_group_id")
    assert isinstance(row["state"], str) and row["state"]
