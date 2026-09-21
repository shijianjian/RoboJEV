"""`robojev train`: the merge, the command it builds, the numbers upstream does not report.

Everything here runs on the CPU with no NanoJev clone, no weights and no GPU. That is possible
because `robojev.train` does not train: it merges rows, builds one argv, runs it as a
subprocess and reads the files it left. The argv can be asserted without running it, the merge and
the eval table are stdlib arithmetic, and `train` itself is exercised with the subprocess replaced
by a function that writes the files a real run would have written -- which is what makes the
rename-on-success rule and the `robojev.json` contents testable rather than merely reasoned about.

`ROBOJEV_HOME` is a `tmp_path` throughout, and `test_train_writes_nothing_outside_robojev_home`
checks that claim rather than trusting it.

The one test that needs the real thing is `gpu`-marked at the bottom. It needs no GPU despite the
marker -- upstream's `--validate-only` is stdlib and refuses to import torch -- but it does need
the built `robojev` environment and the clone, which only the box that trains has.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess

import pytest

from robojev import runtime
from robojev import registry
from robojev import train as trainer

RECIPE = "robojev"
PIN = json.loads((runtime.trainer().dir / "nanojev.json").read_text())
CLONE = f"nanojev@{PIN['commit'][:12]}"

WEIGHTS = b"pretend this is 2.4 GB of safetensors"
DIGEST = hashlib.sha256(WEIGHTS).hexdigest()[:12]


# --------------------------------------------------------------------------------------- fixtures


#: The candidate sets the fake rows answer, small enough to read.
CANDIDATES = {"move_z": ("-", "hold", "+"), "grip": ("false", "true")}


def _row(task: int, demo: int, step: int, split: str, questions=("move_z", "grip")) -> dict:
    """A NanoJev training row of the shape `robojev harvest` writes, small enough to read."""
    row = {
        "id": f"libero_spatial:{task}:{demo}:{step}",
        "state_id": f"libero_spatial:{task}:{demo}:{step}",
        "family_id": "libero_spatial",
        "split": split,
        "state": "Robot state.",
        "questions": {},
        "gold": {},
        "gold_probs": {},
        "gold_probs_kind": "programmatic_conditional_distribution",
        "gold_label_kind": "deterministic_truth",
        "metadata": {"source_group_id": f"libero_spatial:{task}:{demo}", "task_index": task},
    }
    for qid in questions:
        cands = list(CANDIDATES[qid])
        row["questions"][qid] = {"type": "boolean" if qid == "grip" else "choice",
                                 "instructions": "why", "criteria": {}}
        row["gold"][qid] = cands[0]
        row["gold_probs"][qid] = {c: (1.0 if c == cands[0] else 0.0) for c in cands}
    return row


def _write_rows(data_dir: pathlib.Path, per_task: dict[int, tuple[str, int]]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for task, (split, n) in per_task.items():
        lines = [json.dumps(_row(task, demo, 0, split), sort_keys=True) for demo in range(n)]
        (data_dir / f"task_{task}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A `$ROBOJEV_HOME` with a fabricated NanoJev clone under it, and nothing else."""
    monkeypatch.setenv("ROBOJEV_HOME", str(tmp_path / "home"))
    scripts = tmp_path / "home" / "src" / CLONE / PIN["subdir"]
    scripts.mkdir(parents=True)
    (scripts / trainer.TRAINER).write_text("")
    return tmp_path / "home"


@pytest.fixture
def recipe(home):
    """The real `robojev` recipe -- its sidecar is what names the clone the fixture built."""
    return runtime.trainer()


def _finished_checkpoint(out_dir: pathlib.Path, predictions: list[dict]) -> None:
    """Write what a successful `train_pipeline_decisions.py` leaves behind."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best.safetensors").write_bytes(WEIGHTS)
    (out_dir / "config.json").write_text('{"set_head": "attention"}')
    (out_dir / "backbone_config").mkdir()
    (out_dir / "tokenizer").mkdir()
    (out_dir / "summary.json").write_text(json.dumps({
        "best_step": 250, "best_dev_target_ce": 0.5, "training_seconds": 123.0,
        "max_gpu_allocated_gb": 11.9,
        "metrics_by_split": {"test": {"questions": len(predictions), "target_ce": 0.75,
                                      "target_kl": 0.2, "target_tv": 0.1}},
    }))
    (out_dir / "predictions_test.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in predictions))


#: Three `translate` predictions, one of them wrong, with arithmetic that can be checked by hand.
PREDICTIONS = [
    # argmax 0 == gold 0. Brier: (0.8-0.7)^2 + (0.2-0.3)^2 = 0.01 + 0.01 = 0.02
    {"qid": "move_z", "candidate_ids": ["+", "-"], "gold_index": 0,
     "student_probs": [0.8, 0.2], "gold_distribution_probs": [0.7, 0.3]},
    # argmax 1 != gold 0. Brier: (0.4-1)^2 + (0.6-0)^2 = 0.36 + 0.36 = 0.72
    {"qid": "move_z", "candidate_ids": ["+", "-"], "gold_index": 0,
     "student_probs": [0.4, 0.6], "gold_distribution_probs": [1.0, 0.0]},
    # argmax 0 == gold 0. Brier: (0.5-0.5)^2 * 2 = 0.0
    {"qid": "move_z", "candidate_ids": ["+", "-"], "gold_index": 0,
     "student_probs": [0.5, 0.5], "gold_distribution_probs": [0.5, 0.5]},
    # one grip row, so the table has two questions
    {"qid": "grip", "candidate_ids": ["false", "true"], "gold_index": 1,
     "student_probs": [0.25, 0.75], "gold_distribution_probs": [0.25, 0.75]},
]


# ------------------------------------------------------------------------------------- merge_rows


def test_merge_rows_concatenates_in_task_order_and_counts_by_split(tmp_path):
    data = tmp_path / "data"
    _write_rows(data, {0: ("train", 2), 1: ("dev", 1), 2: ("test", 3)})
    counts = trainer.merge_rows(data, tmp_path / "out" / "rows.jsonl")

    rows = [json.loads(line) for line in
            (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    assert [r["metadata"]["task_index"] for r in rows] == [0, 0, 1, 2, 2, 2]
    assert counts["total"] == 6
    assert counts["by_split"] == {"train": 2, "dev": 1, "test": 3}
    assert counts["by_task"] == {"0": 2, "1": 1, "2": 3}
    assert counts["by_question"] == {"move_z": 6, "grip": 6}
    assert counts["files"] == ["task_0.jsonl", "task_1.jsonl", "task_2.jsonl"]


def test_merge_rows_includes_the_v2_grounding_rows_after_the_task_files(tmp_path):
    data = tmp_path / "data"
    _write_rows(data, {0: ("train", 2), 1: ("dev", 1), 2: ("test", 1)})
    row = {"state_id": "grounding:libero_goal:3:0", "split": "train", "state": "Task: ...",
           "questions": {"target": {}}, "metadata": {"source": "grounding"}}
    (data / trainer.GROUNDING_ROWS).write_text(json.dumps(row) + "\n")
    counts = trainer.merge_rows(data, tmp_path / "out" / "rows.jsonl")

    assert counts["files"][-1] == "grounding.jsonl"
    assert counts["total"] == 5
    assert counts["by_task"]["grounding"] == 1
    assert counts["by_question"]["target"] == 1


def test_merge_rows_puts_task_10_after_task_9_not_after_task_1(tmp_path):
    """`sorted()` over file names would interleave them; the concatenation order has to be the
    task order the manifest describes."""
    data = tmp_path / "data"
    _write_rows(data, {1: ("train", 1), 9: ("train", 1), 10: ("train", 1), 2: ("train", 1)})
    trainer.merge_rows(data, tmp_path / "rows.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert [r["metadata"]["task_index"] for r in rows] == [1, 2, 9, 10]


def test_merge_rows_rejects_a_source_group_in_two_splits(tmp_path):
    """The trainer's own hard error (`train_pipeline_decisions.py:168-173`), caught in a second of
    stdlib instead of after a 2.4 GB model load, and naming the group."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "task_0.jsonl").write_text(json.dumps(_row(0, 0, 0, "train")) + "\n")
    # Same task, same demonstration -- so the same `source_group_id` -- relabelled `test`.
    (data / "task_1.jsonl").write_text(json.dumps(
        {**_row(0, 0, 5, "test"), "id": "other"}) + "\n")
    with pytest.raises(ValueError) as caught:
        trainer.merge_rows(data, tmp_path / "rows.jsonl")
    assert "libero_spatial:0:0" in str(caught.value)
    assert "task_0.jsonl" in str(caught.value) and "task_1.jsonl" in str(caught.value)


def test_merge_rows_rejects_a_state_id_in_two_splits(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "task_0.jsonl").write_text(json.dumps(_row(0, 0, 0, "train")) + "\n")
    (data / "task_1.jsonl").write_text(json.dumps(
        {**_row(0, 0, 0, "dev"), "id": "other",
         "metadata": {"source_group_id": "elsewhere", "task_index": 1}}) + "\n")
    with pytest.raises(ValueError, match="state_id"):
        trainer.merge_rows(data, tmp_path / "rows.jsonl")


def test_merge_rows_with_nothing_to_merge_names_harvest(tmp_path):
    (tmp_path / "libero_spatial").mkdir()
    with pytest.raises(FileNotFoundError, match="robojev harvest robojev --suite libero_spatial"):
        trainer.merge_rows(tmp_path / "libero_spatial", tmp_path / "rows.jsonl")


# --------------------------------------------------------------------------------- the invocation


def test_the_trainer_command_is_nanojevs_own_with_the_pinned_backbone_revision(recipe, tmp_path):
    argv, cwd = trainer.trainer_command(recipe, tmp_path / "rows.jsonl", tmp_path / "out",
                                        trainer.TrainConfig(steps=42),
                                        init_checkpoint=tmp_path / "base")
    assert argv[-1] != ""
    assert any(a.endswith("train_pipeline_decisions.py") for a in argv)
    pairs = dict(zip(argv, argv[1:]))
    assert pairs["--revision"] == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert pairs["--model"] == "Qwen/Qwen3-0.6B"
    assert pairs["--objective"] == "gold_distribution"
    assert pairs["--loss"] == "ce"
    assert pairs["--set-head"] == "attention"
    assert pairs["--precision"] == "bf16"
    assert pairs["--steps"] == "42"
    assert pairs["--init-checkpoint"] == str(tmp_path / "base")
    assert pairs["--batch-questions"] == "12" and pairs["--microbatch-questions"] == "4"
    # The runbook's cap, not the flag's default of 16384: four 7-candidate questions at 534
    # tokens is 15k padded tokens of activations, which a 24 GB card does not have.
    assert pairs["--max-microbatch-tokens"] == "6000"
    assert "--gradient-checkpointing" in argv
    # cwd is the clone's `scripts/`: upstream's modules import each other by bare name.
    assert cwd.endswith(os.path.join(CLONE, PIN["subdir"]))


def test_the_trainer_is_asked_for_the_path_budget_robojev_owns(recipe, tmp_path):
    """One number, two readers. A checkpoint trained at 512 and served at 1536 -- or the other way
    round -- is asked a differently truncated question about the same state, and a five-object
    LIBERO-Spatial path is 1245 tokens since ruling 14's memory block, well over NanoJev's
    default."""
    argv, _ = trainer.trainer_command(recipe, tmp_path / "rows.jsonl", tmp_path / "out",
                                      trainer.TrainConfig())
    budget = dict(zip(argv, argv[1:]))["--max-length"]
    assert int(budget) == trainer.MAX_PATH_TOKENS == 1024
    assert int(budget) == registry.max_path_tokens(registry.DEFAULT_VERSION)
    # Well over NanoJev's own default of 512, which is the reason it is passed explicitly.
    assert int(budget) > 512


def test_gradient_checkpointing_can_be_turned_off_for_a_card_that_does_not_need_it(recipe, tmp_path):
    argv, _ = trainer.trainer_command(recipe, tmp_path / "rows.jsonl", tmp_path / "out",
                                      trainer.TrainConfig(gradient_checkpointing=False))
    assert "--gradient-checkpointing" not in argv


def test_a_from_scratch_run_passes_no_init_checkpoint(recipe, tmp_path):
    argv, _ = trainer.trainer_command(recipe, tmp_path / "rows.jsonl", tmp_path / "out",
                                      trainer.TrainConfig(from_scratch=True))
    assert "--init-checkpoint" not in argv


def test_the_trainer_runs_in_the_robojev_environment(recipe, tmp_path, monkeypatch):
    """Not `sys.executable`: NanoJev wants Python 3.14 and torch 2.14, and the interpreter that
    runs the CLI is the pixi one."""
    venv = tmp_path / "envs" / recipe.env_name / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("")
    monkeypatch.setattr(type(recipe), "built_venv", lambda self: venv)
    argv, _ = trainer.trainer_command(recipe, tmp_path / "rows.jsonl", tmp_path / "out",
                                      trainer.TrainConfig())
    assert argv[0] == str(venv / "bin" / "python")


def test_the_validate_only_command_is_upstreams_own_audit(recipe, tmp_path):
    argv, cwd = trainer.validate_command(recipe, tmp_path / "rows.jsonl")
    assert "--validate-only" in argv and "--output-dir" not in argv
    assert cwd.endswith(PIN["subdir"])


def test_a_missing_clone_names_trainer_env(recipe, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "nanojev_clone", lambda r: None)
    with pytest.raises(FileNotFoundError, match="robojev train --build-env"):
        trainer.upstream_scripts(recipe)


# ---------------------------------------------------------------------------------- the eval table


def test_eval_table_computes_accuracy_and_brier_per_question(tmp_path):
    path = tmp_path / "predictions_test.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in PREDICTIONS))
    table = trainer.eval_table(path)

    translate = table["by_question"]["move_z"]
    assert translate["n"] == 3
    assert translate["accuracy"] == pytest.approx(2 / 3)
    assert translate["brier"] == pytest.approx((0.02 + 0.72 + 0.0) / 3)
    grip = table["by_question"]["grip"]
    assert grip["n"] == 1 and grip["accuracy"] == 1.0 and grip["brier"] == pytest.approx(0.0)
    assert table["total"]["n"] == 4
    assert table["total"]["accuracy"] == pytest.approx(3 / 4)
    assert table["total"]["brier"] == pytest.approx(0.74 / 4)


def test_eval_table_scores_brier_against_the_soft_target_not_a_one_hot(tmp_path):
    """The rows carry ρ-softened distributions on purpose (spec §5). Scoring against a one-hot
    would reward exactly the overconfidence the soft target exists to prevent: a model that says
    0.7 where the target is 0.7 scores 0 here and 0.18 against a one-hot."""
    path = tmp_path / "predictions_test.jsonl"
    path.write_text(json.dumps({"qid": "move_z", "candidate_ids": ["+", "-"],
                                "gold_index": 0, "student_probs": [0.7, 0.3],
                                "gold_distribution_probs": [0.7, 0.3]}) + "\n")
    assert trainer.eval_table(path)["by_question"]["move_z"]["brier"] == pytest.approx(0.0)


def test_eval_table_falls_back_to_the_rows_own_gold_probs_in_candidate_order(tmp_path):
    """`prediction_record` carries `gold_distribution_probs` (a list) when the pipeline row had
    one and `gold_probs` (a dict keyed by candidate id) regardless; the dict has to be put in the
    model's candidate order before it can be differenced with `student_probs`."""
    path = tmp_path / "predictions_test.jsonl"
    path.write_text(json.dumps({"qid": "grip", "candidate_ids": ["false", "true"],
                                "gold_index": 1, "student_probs": [0.25, 0.75],
                                "gold_probs": {"true": 0.75, "false": 0.25}}) + "\n")
    assert trainer.eval_table(path)["by_question"]["grip"]["brier"] == pytest.approx(0.0)


def test_eval_table_copies_upstreams_target_ce_rather_than_recomputing_it(tmp_path):
    path = tmp_path / "predictions_test.jsonl"
    path.write_text(json.dumps(PREDICTIONS[0]) + "\n")
    # Upstream spells them `target_ce`/`target_kl` at the split level; the bare `ce`/`kl` keys
    # only exist inside `by_target_kind`, and reading those would silently report nothing.
    table = trainer.eval_table(
        path, {"metrics_by_split": {"test": {"target_ce": 0.4242, "target_kl": 0.1,
                                             "by_target_kind": {"x": {"ce": 9.9}}}}})
    assert table["total"]["target_ce"] == 0.4242
    assert table["total"]["target_kl"] == 0.1


def test_a_question_with_no_hard_gold_has_no_accuracy_rather_than_a_zero(tmp_path):
    path = tmp_path / "predictions_test.jsonl"
    path.write_text(json.dumps({"qid": "grip", "candidate_ids": ["false", "true"],
                                "gold_index": None, "student_probs": [0.5, 0.5],
                                "gold_distribution_probs": [0.5, 0.5]}) + "\n")
    table = trainer.eval_table(path)
    assert table["by_question"]["grip"]["accuracy"] is None
    assert "    -" in trainer.format_table(table)


def test_format_table_prints_the_question_set_in_its_own_order(tmp_path):
    path = tmp_path / "predictions_test.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in PREDICTIONS))
    text = trainer.format_table(
        trainer.eval_table(path, {"metrics_by_split": {"test": {"target_ce": 0.5}}}))
    lines = [line.split()[0] for line in text.splitlines()]
    assert lines[:1] == ["question"]
    assert lines.index("move_z") < lines.index("grip")   # the question set's order, not seen
    assert "all" in lines
    assert "0.75" in text and "test target CE 0.5000" in text


# ------------------------------------------------------------------------------------------ train


@pytest.fixture
def trained(home, recipe, monkeypatch):
    """`train` with the subprocess replaced by a function that writes what a real run leaves."""
    _write_rows(home / "data" / "robojev" / "libero_spatial",
                {0: ("train", 2), 1: ("dev", 1), 2: ("test", 1)})
    (home / "data" / "robojev" / "libero_spatial" / "manifest.json").write_text(json.dumps({
        "policy": "robojev", "suite": "libero_spatial", "rho": 0.85, "seed": 17,
        "questions_version": "v2",
        "delta_t": 0.3536, "delta_r": 0.0579, "every": 5, "grip_repeats": 8,
        "splits": {"train": [0], "dev": [1], "test": [2]},
    }))
    monkeypatch.setattr(trainer, "base_checkpoint",
                        lambda log=None, revision=None: (home / "base", revision or "cafef00d"))

    calls = {}

    def fake_run(argv, cwd, log=trainer.log_to_stderr):
        calls["argv"], calls["cwd"] = argv, cwd
        out = pathlib.Path(dict(zip(argv, argv[1:]))["--output-dir"])
        _finished_checkpoint(out, PREDICTIONS)
        return []

    monkeypatch.setattr(trainer, "run_trainer", fake_run)
    return calls


def test_train_writes_the_checkpoint_where_pull_looks_for_it(home, trained):
    summary = trainer.train("robojev", "libero_spatial", trainer.TrainConfig(steps=3))
    directory = home / "checkpoints" / "robojev" / "libero_spatial"
    # The merged rows are the trainer's input and do not ship inside the weights.
    assert not (directory / trainer.ROWS).exists()
    assert not list(directory.parent.glob("*.rows"))
    assert pathlib.Path(summary["checkpoint"]) == directory
    assert directory == runtime.local_checkpoint_dir("robojev", "libero_spatial")
    for name in runtime.LOCAL_CHECKPOINT_FILES:
        assert (directory / name).exists()
    assert not list(directory.parent.glob("*.tmp-*"))


def test_the_printed_revision_is_the_weights_digest(home, trained):
    summary = trainer.train("robojev", "libero_spatial", trainer.TrainConfig())
    directory = home / "checkpoints" / "robojev" / "libero_spatial"
    assert summary["revision"] == DIGEST == runtime.local_checkpoint_revision(directory)
    assert json.loads((directory / trainer.CHECKPOINT_META).read_text())["revision"] == DIGEST


def test_robojev_json_carries_the_manifests_deltas_rho_and_seed(home, trained):
    """Spec §5: "the harvest manifest (row counts, ρ, δ, seed)" goes into the checkpoint, and
    `robojev/policy.py` reads δ_t/δ_r back out of it -- a served checkpoint that did not
    know its own step size would compose every answer at the wrong scale."""
    trainer.train("robojev", "libero_spatial", trainer.TrainConfig(steps=3))
    directory = home / "checkpoints" / "robojev" / "libero_spatial"
    meta = json.loads((directory / trainer.CHECKPOINT_META).read_text())
    assert (meta["delta_t"], meta["delta_r"], meta["rho"], meta["seed"]) == (0.3536, 0.0579, 0.85, 17)
    assert meta["rows"]["total"] == 4 and meta["rows"]["by_split"]["test"] == 1
    assert meta["base_checkpoint"] == "C-Tianyu/NanoJev@cafef00d"
    assert meta["max_path_tokens"] == trainer.MAX_PATH_TOKENS
    assert meta["eval"]["by_question"]["move_z"]["accuracy"] == pytest.approx(2 / 3)
    assert meta["trainer"][-1] and "--objective" in meta["trainer"]
    assert meta["trained_at"].endswith("Z")
    assert meta["eval"]["total"]["target_ce"] == 0.75   # upstream's own selection metric, copied
    # and the manifest itself is beside it, verbatim, so nothing has to trust the copy
    assert json.loads((directory / "manifest.json").read_text())["delta_t"] == 0.3536


def test_train_renames_into_place_only_on_success(home, recipe, trained, monkeypatch):
    def explode(argv, cwd, log=trainer.log_to_stderr):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(trainer, "run_trainer", explode)
    with pytest.raises(subprocess.CalledProcessError):
        trainer.train("robojev", "libero_spatial", trainer.TrainConfig())
    checkpoints = home / "checkpoints" / "robojev"
    assert not (checkpoints / "libero_spatial").exists()
    assert not list(checkpoints.glob("*.tmp-*")), "a half-written checkpoint was left behind"


# ------------------------------------------------- the weights survive a post-training failure


def test_a_failure_after_the_weights_exist_keeps_them_and_says_how_to_resume(home, recipe, trained,
                                                                             monkeypatch):
    """The whole point of splitting the cleanup: `work` holds hours of a shared GPU by the time
    the eval table is computed, and nothing that can go wrong afterwards is worth deleting it."""
    def boom(predictions, summary=None):
        raise KeyError("a key upstream renamed")

    monkeypatch.setattr(trainer, "eval_table", boom)
    with pytest.raises(trainer.TrainError) as caught:
        trainer.train("robojev", "libero_spatial", trainer.TrainConfig())

    checkpoints = home / "checkpoints" / "robojev"
    [work] = list(checkpoints.glob("libero_spatial.tmp-*"))
    assert (work / "best.safetensors").read_bytes() == WEIGHTS   # untouched, not deleted
    assert not (checkpoints / "libero_spatial").exists()         # and not renamed into place
    message = str(caught.value)
    assert str(work) in message and "NOT been touched" in message
    assert f"--finish {work}" in message
    # The scratch rows directory is the one thing that does go: it is the trainer's input.
    assert not list(checkpoints.glob("*.rows"))


def test_finish_picks_up_where_the_failure_left_off(home, recipe, trained, monkeypatch):
    """And re-running it costs seconds and no GPU: everything it reads is already on disk."""
    real = trainer.eval_table
    calls = []

    def once(predictions, summary=None):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("no space left on device")
        return real(predictions, summary)

    monkeypatch.setattr(trainer, "eval_table", once)
    with pytest.raises(trainer.TrainError):
        trainer.train("robojev", "libero_spatial", trainer.TrainConfig())
    [work] = list((home / "checkpoints" / "robojev").glob("libero_spatial.tmp-*"))

    summary = trainer.finish(work)
    directory = home / "checkpoints" / "robojev" / "libero_spatial"
    assert pathlib.Path(summary["checkpoint"]) == directory
    assert summary["revision"] == DIGEST
    assert not work.exists()
    meta = json.loads((directory / trainer.CHECKPOINT_META).read_text())
    # The context survived the crash, so the argv, the row counts and the warm start are all still
    # the ones the run really used -- `finish` was told none of them.
    assert meta["rows"]["total"] == 4
    assert meta["base_checkpoint"] == "C-Tianyu/NanoJev@cafef00d"
    assert "--objective" in meta["trainer"]
    assert meta["delta_t"] == 0.3536
    # And the context file is not left inside the published checkpoint.
    assert not (directory / trainer.TRAIN_CONTEXT).exists()


def test_finish_refuses_a_directory_that_is_not_a_training_run(home, tmp_path):
    """No context file *and* no `<suite>.tmp-<stamp>` name: nothing says what this is."""
    (tmp_path / "somewhere").mkdir()
    with pytest.raises(trainer.TrainError) as caught:
        trainer.finish(tmp_path / "somewhere")
    assert caught.value.exit_code == 2
    assert trainer.TRAIN_CONTEXT in str(caught.value)


def test_finish_reads_a_directory_written_before_the_context_existed(home, recipe, trained,
                                                                    monkeypatch):
    """A run started by an older `robojev train` has no context file, and it is still a directory
    full of finished weights: refusing to describe it would be the same mistake as deleting it.
    The name carries the convention -- `<checkpoints>/<policy>/<suite>.tmp-<stamp>` -- so the
    policy, the suite and the destination all read off the path."""
    real = trainer.eval_table
    calls = []

    def once(predictions, summary=None):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("no space left on device")
        return real(predictions, summary)

    monkeypatch.setattr(trainer, "eval_table", once)
    with pytest.raises(trainer.TrainError):
        trainer.train("robojev", "libero_spatial", trainer.TrainConfig())
    [work] = list((home / "checkpoints" / "robojev").glob("libero_spatial.tmp-*"))
    (work / trainer.TRAIN_CONTEXT).unlink()          # as an older run would have left it

    summary = trainer.finish(work)
    directory = home / "checkpoints" / "robojev" / "libero_spatial"
    assert pathlib.Path(summary["checkpoint"]) == directory
    meta = json.loads((directory / trainer.CHECKPOINT_META).read_text())
    assert meta["inferred_from_the_directory_name"] is True
    # The δ pair still comes from the harvest, because that is not something a name can carry.
    assert meta["delta_t"] == 0.3536
    # And what the context would have held is recorded as unknown rather than guessed.
    assert meta["rows"] is None and meta["base_checkpoint"] is None


def test_finish_refuses_a_run_whose_weights_are_missing_without_deleting_it(home, tmp_path):
    work = tmp_path / "libero_spatial.tmp-x"
    work.mkdir()
    (work / trainer.TRAIN_CONTEXT).write_text(json.dumps({
        "policy": "robojev", "suite": "libero_spatial", "data": str(tmp_path),
        "out": str(tmp_path / "libero_spatial"), "stamp": "x", "trainer": [], "trainer_cwd": "",
        "rows": {}, "max_path_tokens": 1024, "base_checkpoint": None,
    }))
    with pytest.raises(trainer.TrainError, match="best.safetensors"):
        trainer.finish(work)
    assert work.exists(), "a directory that might yet be finished must not be deleted"


# ------------------------------------------------------------------ the manifest is not optional


def test_a_harvest_without_its_manifest_is_refused_before_the_gpu(home, recipe, monkeypatch):
    """δ_t/δ_r come from the manifest and from nowhere else. A run given rows without it would
    train fine, verify under `pull`, and then compose every action at the wrong scale."""
    _write_rows(home / "data" / "robojev" / "libero_spatial",
                {0: ("train", 2), 1: ("dev", 1), 2: ("test", 1)})
    monkeypatch.setattr(trainer, "run_trainer",
                        lambda *a, **k: pytest.fail("the trainer must not be launched"))
    with pytest.raises(trainer.TrainError) as caught:
        trainer.train("robojev", "libero_spatial", trainer.TrainConfig())
    assert caught.value.exit_code == 2
    assert "manifest.json" in str(caught.value) and "robojev harvest" in str(caught.value)
    # And nothing was created for it to have failed into.
    assert not (home / "checkpoints").exists()


def test_a_manifest_without_the_deltas_is_refused_too(home, recipe, tmp_path):
    data = home / "data" / "robojev" / "libero_spatial"
    _write_rows(data, {0: ("train", 1)})
    (data / "manifest.json").write_text(json.dumps({"policy": "robojev", "rho": 0.85}))
    with pytest.raises(trainer.TrainError, match="delta_t"):
        trainer.read_manifest(data)


# ------------------------------------------------- the question set comes from the rows, not a flag

#: A v2 harvest manifest, in `robojev.rollout.write_manifest`'s shape: the knobs that
#: change a rendered byte or a composed number, which are exactly what the server reads back.
V2_MANIFEST = {
    "policy": "robojev", "suite": "libero_spatial", "source": "expert_rollout",
    "questions_version": "v2",
    "qids": ["move_x", "move_y", "move_z", "size_x", "size_y", "size_z", "yaw", "rim",
             "grip", "subgoal"],
    "annotate": True, "cm_per_unit": 5.0, "delta_t": 1.0, "delta_r": 0.05785714285714285,
    "seed": 17, "memory_rule": "v2",
    "tracker": {"annotate": True, "tolerance_cm": 0.8, "history": 3},
    "steps": {"cm": {"large": 5.0, "medium": 1.7, "small": 0.5}, "cm_per_unit": 5.0},
    "bands": {"tolerance_cm": 0.8, "arrival_cm": 1.5, "yaw_tolerance_deg": 10.0},
    "splits": {"train": [0], "dev": [1], "test": [2]},
}


def _manifest(home: pathlib.Path, manifest: dict) -> pathlib.Path:
    path = home / "data" / "robojev" / "libero_spatial" / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    return path


def test_a_harvest_that_names_no_question_set_never_reaches_the_gpu(home, trained, monkeypatch):
    """A manifest that says nothing predates the key, which means the retired set. Upgrading it
    silently would train on rows nothing can serve; the refusal names the set instead."""
    _manifest(home, {"delta_t": 0.3536, "delta_r": 0.0579,
                     "splits": {"train": [0], "dev": [1], "test": [2]}})
    monkeypatch.setattr(trainer, "run_trainer",
                        lambda *a, **k: pytest.fail("the trainer must not be launched"))
    with pytest.raises(trainer.TrainError) as caught:
        trainer.train("robojev", "libero_spatial", trainer.TrainConfig(steps=3))
    assert "'v1'" in str(caught.value) and "retired" in str(caught.value)
    assert caught.value.exit_code == 2


def test_a_v2_harvest_picks_v2s_path_budget_with_no_flag_anywhere(home, trained):
    """`--max-length` is chosen from the rows' own `questions_version`. A checkpoint trained at
    one budget and served at another is asked a differently truncated question about the same
    state, and upstream raises rather than truncating."""
    _manifest(home, V2_MANIFEST)
    trainer.train("robojev", "libero_spatial", trainer.TrainConfig(steps=3))
    budget = dict(zip(trained["argv"], trained["argv"][1:]))["--max-length"]
    assert int(budget) == registry.max_path_tokens("v2") == 1024


def test_the_checkpoint_carries_the_vocabulary_the_server_has_to_agree_with(home, trained):
    """`robojev.json` is where `robojev/policy.py` learns which questions these weights
    answer and at what scale -- the step size above all, because a checkpoint trained on 5 cm
    steps and served at 25 cm executes five times every move it meant."""
    _manifest(home, V2_MANIFEST)
    trainer.train("robojev", "libero_spatial", trainer.TrainConfig(steps=3))
    meta = json.loads((home / "checkpoints" / "robojev" / "libero_spatial"
                       / trainer.CHECKPOINT_META).read_text())
    assert meta["questions_version"] == "v2"
    # The active qids by name, never a count. A manifest's set is what its rows were harvested
    # with, and the server reads it from here rather than from today's default -- which is what
    # keeps a checkpoint harvested before `yaw` was asked serving (note
    # `2026-09-21-stage9-drawer-task.md` §3.6).
    assert meta["qids"] == V2_MANIFEST["qids"] == list(registry.qids("v2"))
    assert meta["cm_per_unit"] == 5.0
    assert meta["steps"] == V2_MANIFEST["steps"]
    assert meta["tracker"] == V2_MANIFEST["tracker"]
    assert meta["bands"] == V2_MANIFEST["bands"]
    assert meta["memory_rule"] == "v2"
    assert meta["max_path_tokens"] == 1024


def test_a_manifest_that_lists_its_own_qids_is_believed_over_the_registry(home):
    """A harvest may have asked a subset -- the `yaw` ablation, say. The rows are the authority on
    what was asked; the registry is only the fallback for a manifest that does not say."""
    block = trainer.vocabulary({**V2_MANIFEST, "qids": ["move_x", "grip"]})
    assert block["qids"] == ["move_x", "grip"]
    assert trainer.vocabulary({"questions_version": "v2"})["qids"] == list(registry.qids("v2"))


def test_a_manifest_naming_an_unknown_question_set_never_reaches_the_gpu(home, recipe,
                                                                        trained, monkeypatch):
    _manifest(home, {**V2_MANIFEST, "questions_version": "v3"})
    monkeypatch.setattr(trainer, "run_trainer",
                        lambda *a, **k: pytest.fail("the trainer must not be launched"))
    with pytest.raises(trainer.TrainError) as caught:
        trainer.train("robojev", "libero_spatial", trainer.TrainConfig())
    assert caught.value.exit_code == 2
    assert "v3" in str(caught.value) and "v2" in str(caught.value)
    assert not list((home / "checkpoints" / "robojev").glob("*.tmp-*"))


def test_for_rows_leaves_everything_else_about_the_run_alone(home):
    cfg = trainer.TrainConfig(steps=3350, seed=4, gradient_checkpointing=False)
    v2 = trainer.for_rows(cfg, V2_MANIFEST)
    assert v2.max_length == 1024
    assert dataclasses.replace(v2, max_length=cfg.max_length) == cfg
    assert trainer.for_rows(cfg, {"questions_version": "v2"}).max_length == 1024


def test_a_trainer_that_exits_zero_without_weights_is_still_a_failure(home, recipe, trained,
                                                                     monkeypatch):
    """And *this* one is cleaned up: an exit-0 run with no weights produced nothing to keep, which
    is the last moment at which that is true -- everything after it is `finish`."""
    monkeypatch.setattr(trainer, "run_trainer",
                        lambda argv, cwd, log=None: pathlib.Path(
                            dict(zip(argv, argv[1:]))["--output-dir"]).mkdir(exist_ok=True) or "")
    with pytest.raises(trainer.TrainError, match="best.safetensors"):
        trainer.train("robojev", "libero_spatial", trainer.TrainConfig())
    checkpoints = home / "checkpoints" / "robojev"
    assert not (checkpoints / "libero_spatial").exists()
    assert not list(checkpoints.glob("*.tmp-*")) and not list(checkpoints.glob("*.rows"))


def test_a_retrain_keeps_the_checkpoint_it_replaces(home, trained):
    """Nothing published these weights (plan ruling 4), so the old ones are the only copy: they
    are moved aside, not deleted, and the operator decides when to remove them."""
    first = trainer.train("robojev", "libero_spatial", trainer.TrainConfig())
    second = trainer.train("robojev", "libero_spatial", trainer.TrainConfig())
    assert first["checkpoint"] == second["checkpoint"]
    kept = list((home / "checkpoints" / "robojev").glob("libero_spatial.was-*"))
    assert len(kept) == 1 and (kept[0] / "best.safetensors").exists()


def test_train_writes_nothing_outside_robojev_home(home, trained, tmp_path):
    before = {p for p in tmp_path.rglob("*")} | {p for p in (tmp_path.parent).glob("*")}
    trainer.train("robojev", "libero_spatial", trainer.TrainConfig())
    new = {p for p in tmp_path.rglob("*")} - before
    assert new, "the test would pass vacuously if nothing was written"
    assert all(str(p).startswith(str(home)) for p in new), sorted(str(p) for p in new)[:5]


def test_from_scratch_records_that_there_was_no_warm_start(home, trained):
    trainer.train("robojev", "libero_spatial", trainer.TrainConfig(from_scratch=True))
    meta = json.loads((home / "checkpoints" / "robojev" / "libero_spatial"
                       / trainer.CHECKPOINT_META).read_text())
    assert meta["base_checkpoint"] is None
    assert "--init-checkpoint" not in meta["trainer"]


# ------------------------------------------------------------------------------- the backbone

def test_the_default_run_is_nanojevs_own_warm_start_and_nothing_else_is():
    """One predicate, three callers (`trainer_command`, `train`, `slurm`)."""
    assert trainer.warm_starts_from_nanojev(trainer.TrainConfig())
    assert not trainer.warm_starts_from_nanojev(trainer.TrainConfig(from_scratch=True))
    assert not trainer.warm_starts_from_nanojev(trainer.TrainConfig(base_model="Qwen/Qwen3-4B"))


def test_a_bare_backbone_is_the_model_flag_and_a_warm_start_is_not(recipe):
    """Spec ruling 13: `Qwen/Qwen3-4B` goes to upstream's *other* branch -- the body from the hub
    and fresh heads sized from its own hidden size -- so `--model` carries it and there is no
    `--init-checkpoint` for a `DecisionModel` that does not exist at that width."""
    r = runtime.trainer()
    four_b = trainer.TrainConfig(base_model="Qwen/Qwen3-4B")
    argv, _cwd = trainer.trainer_command(r, "rows.jsonl", "out", four_b)
    assert argv[argv.index("--model") + 1] == "Qwen/Qwen3-4B"
    assert argv[argv.index("--revision") + 1] == "main"
    assert "--init-checkpoint" not in argv
    # And the default is unchanged: the runbook's pinned 0.6B, inert beside the warm start.
    argv, _cwd = trainer.trainer_command(r, "rows.jsonl", "out", trainer.TrainConfig(),
                                         init_checkpoint="/warm")
    assert argv[argv.index("--model") + 1] == trainer.BACKBONE_MODEL
    assert argv[argv.index("--revision") + 1] == trainer.BACKBONE_REVISION
    assert argv[argv.index("--init-checkpoint") + 1] == "/warm"


def test_a_pinned_base_revision_reaches_both_branches(recipe):
    r = runtime.trainer()
    argv, _ = trainer.trainer_command(r, "rows.jsonl", "out",
                                      trainer.TrainConfig(base_model="Qwen/Qwen3-4B",
                                                          base_revision="deadbeef"))
    assert argv[argv.index("--revision") + 1] == "deadbeef"
    # Under a NanoJev warm start `base_revision` is a commit of *that repository*, which the
    # node downloads; the backbone pin beside it stays the runbook's and stays inert.
    assert trainer.backbone_for(trainer.TrainConfig(base_revision="cafe")) == (
        trainer.BACKBONE_MODEL, trainer.BACKBONE_REVISION)


def test_a_4b_run_never_downloads_the_0_6b_warm_start(home, trained, monkeypatch):
    """2.4 GB of `DecisionModel` weights whose heads are the wrong width: not fetched at all."""
    def refuse(*a, **kw):
        raise AssertionError("base_checkpoint must not be called for a bare-backbone run")

    monkeypatch.setattr(trainer, "base_checkpoint", refuse)
    trainer.train("robojev", "libero_spatial",
                  trainer.TrainConfig(base_model="Qwen/Qwen3-4B", base_revision="abc123"))
    meta = json.loads((home / "checkpoints" / "robojev" / "libero_spatial"
                       / trainer.CHECKPOINT_META).read_text())
    assert meta["base_checkpoint"] is None
    assert meta["base_model"] == "Qwen/Qwen3-4B@abc123"
    assert "--init-checkpoint" not in meta["trainer"]


def test_the_checkpoint_says_which_backbone_it_is_a_fine_tune_of(home, trained):
    """So a served checkpoint can be identified without opening 16 GB of weights."""
    trainer.train("robojev", "libero_spatial", trainer.TrainConfig())
    meta = json.loads((home / "checkpoints" / "robojev" / "libero_spatial"
                       / trainer.CHECKPOINT_META).read_text())
    assert meta["base_model"] == f"{trainer.BACKBONE_MODEL}@{trainer.BACKBONE_REVISION}"
    assert meta["base_checkpoint"] == "C-Tianyu/NanoJev@cafef00d"


def test_a_finish_with_no_context_reads_the_backbone_out_of_upstreams_own_config():
    assert trainer._backbone_from_trainer_config(
        {"model": "Qwen/Qwen3-4B", "resolved_model_revision": "abc"}) == "Qwen/Qwen3-4B@abc"
    assert trainer._backbone_from_trainer_config({"model": "Qwen/Qwen3-4B"}) == "Qwen/Qwen3-4B"
    assert trainer._backbone_from_trainer_config({}) is None


# -------------------------------------------------------------------------------------------- gpu


@pytest.mark.gpu
def test_nanojevs_own_validator_accepts_the_harvested_rows(tmp_path, monkeypatch):
    """Spec §9's first proof: "the row format ... its trainer accepts the file".

    `gpu`-marked because it needs the built `robojev` environment and the NanoJev clone, not
    because it needs the card: `--validate-only` is upstream's stdlib audit and imports neither
    torch nor a tokenizer (`train_pipeline_decisions.py:409,421-423`).

    Point `ROBOPP_ROBOJEV_DATA` at a harvested directory. Its home is read back off that path
    rather than asked for separately -- a harvest lands in `$ROBOJEV_HOME/data/<policy>/<suite>`,
    so the three levels above it *are* the home that holds the clone and the built environment,
    and `conftest._isolated_robojev_home` has pointed `$ROBOJEV_HOME` at a `tmp_path` by now.
    """
    data = os.environ.get("ROBOPP_ROBOJEV_DATA")
    if not data:
        pytest.skip("set ROBOPP_ROBOJEV_DATA to a harvested directory")
    monkeypatch.setenv("ROBOJEV_HOME", str(pathlib.Path(data).resolve().parents[2]))
    rows = tmp_path / "rows.jsonl"
    counts = trainer.merge_rows(pathlib.Path(data), rows)
    argv, cwd = trainer.validate_command(runtime.trainer(), rows)
    done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    audit = json.loads(done.stdout.splitlines()[-1])
    assert audit["records"] == counts["total"]
    # `questions_by_split` counts flattened questions, not rows: four groups per row, minus the
    # `grip` a row with no measurement to carry omits entirely.
    assert {"train", "dev", "test"} <= set(audit["questions_by_split"])
    assert audit["questions_by_split"]["train"] >= counts["by_split"]["train"]
    # Every question of every split is usable for the objective `robojev train` asks for.
    assert all(audit["eligible_by_split_objective"][f"{split}/gold_distribution"] == n
               for split, n in audit["questions_by_split"].items())
