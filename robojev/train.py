"""`robojev train --suite libero_spatial` -- NanoJev's own trainer, driven from here.

Nothing here trains anything. Every gradient this command produces is produced by
`scripts/train_pipeline_decisions.py` at the commit `robojev/trainer/nanojev.json` pins,
run in the trainer's own Python 3.14 / torch 2.14 environment. What this module contributes is the
four things that script does not do and cannot be asked to do:

1. **One input file.** The harvest writes `task_0.jsonl … task_9.jsonl`, one per LIBERO task,
   because a task is the unit a harvest re-runs. The trainer takes a single `--input` (or a
   directory of `<split>.jsonl`, which is a different layout). `merge_rows` concatenates them in
   task order and re-checks, in a second of stdlib, the one cross-file rule the trainer makes a
   hard error hours later on a loaded GPU (`train_pipeline_decisions.py:168-173`).
2. **Where the checkpoint goes, and that a half-written one never lands.** The trainer refuses to
   write into a directory that already holds `config.json` or `best.safetensors` (`:433`), which
   is the right rule and the wrong place for it to be enforced from this side: an interrupted
   run must not leave a directory that a later verification would then happily accept. So
   training runs into `<checkpoints>/<policy>/<suite>.tmp-<stamp>/` and is renamed into place only
   after the four files upstream's own loader requires are all there.
3. **Accuracy and Brier** (a design ruling). Upstream's evaluator reports CE, KL and TV against the
   objective's target and nothing else -- reasonable for a calibration paper, useless as the
   headline of a robot policy. `prediction_record` (`train_toy_decisions.py:156-172`) carries
   exactly what the two missing numbers need: `qid`, `candidate_ids`, `gold_index`,
   `student_probs` and `gold_distribution_probs`. `eval_table` computes them per question.
4. **The provenance the server reads back.** `robojev.json` beside the weights is the harvest
   manifest verbatim -- δ_t, δ_r, ρ, the row counts, the seed (docs/DESIGN.md) -- plus the trainer's own
   argv, the base checkpoint, the eval table and a timestamp. `robojev/policy.py` reads
   δ_t/δ_r out of it at load; without it a served checkpoint does not know how big a step its own
   `translate` answer means.

**This is an operator command, never a queue job** (docs/DESIGN.md). It wants the whole GPU for minutes
and the worker is using the same card, so `cli.train_one` prints that in words before it starts.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

from robojev import home, registry, runtime

#: The provenance file this writes beside the weights, and `robojev/policy.py` reads.
#: Spelt the same in both places and nowhere else.
CHECKPOINT_META = "robojev.json"

#: Upstream's released weights: the warm start the design notes asks for ("full fine-tune from
#: `C-Tianyu/NanoJev`"). Not redistributed here -- only fetched into the Hugging Face cache on
#: the box that trains.
BASE_CHECKPOINT_REPO = "C-Tianyu/NanoJev"

#: The backbone and the exact revision the runbook pins (`research/pipeline_runbook.md:19-29`).
#: Only read when `--from-scratch` skips the warm start; with `--init-checkpoint` the tokenizer
#: and the body config come out of the checkpoint itself (`train_pipeline_decisions.py:448-456`).
BACKBONE_MODEL = "Qwen/Qwen3-0.6B"
BACKBONE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

#: The upstream script this runs, inside the clone's `subdir`.
TRAINER = "train_pipeline_decisions.py"

#: What `merge_rows` writes, in a scratch directory beside the training output -- the trainer's
#: input, not part of its output, so it is not carried into the published checkpoint.
ROWS = "rows.jsonl"

#: The harvest's manifest, `robojev.dataset.MANIFEST`. Named here rather than imported for
#: the same reason `data_root` is spelt out: importing the harvester drags numpy in for one string.
HARVEST_MANIFEST = "manifest.json"

#: Written into the `.tmp-` directory before the trainer starts and removed when it is renamed into
#: place: what `--finish` needs to describe a checkpoint whose run is over.
TRAIN_CONTEXT = "robojev-train-context.json"


class TrainError(Exception):
    """A `robojev train` failure with an exit code, so `cli.train_one` can report it as itself.

    `exit_code` is 2 for the things the spec calls configuration -- a missing harvest manifest, a
    `--finish` pointed at a directory that is not a training run -- and 1 for a failure of the run.
    """

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


#: The tokenizer budget for one candidate path, `--max-length`. **`robojev` owns the
#: number**, so the trainer and `robojev/policy.py` cannot disagree about it: a checkpoint
#: trained at one budget and served at another is asked a differently-shaped question about the
#: same state, and upstream raises rather than truncates when a path does not fit
#: (`predict_toy_decisions.py:110-112`).
#:
#: The default, for a `TrainConfig` built without a version in hand; a run resolves the number
#: from the harvest's own `questions_version` through `registry.max_path_tokens`.
MAX_PATH_TOKENS: int = registry.max_path_tokens(registry.DEFAULT_VERSION)


def log_to_stderr(message: str) -> None:
    """Progress goes to stderr; stdout is the eval table and the digest line. Same contract
    `robojev harvest` has, for the same reason -- `$(robojev train …)` should be usable."""
    print(f"robojev train: {message}", file=sys.stderr, flush=True)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Everything that changes what the trainer is asked, in one object `robojev.json` can echo.

    The defaults are NanoJev's own (`train_pipeline_decisions.py:386-410`, runbook `:131-141`) and
    are not tuned here: the released checkpoint this warm-starts from was trained with them, and
    the design notes asks for a fine-tune "through its own code", which includes its own hyperparameters.
    Every learning rate, the objective, the loss, the head, the precision, the seed and both batch
    numbers are upstream's untouched.

    Three fields are not the flag's default, and each is a fact about *this* data or *this* card
    rather than a tuning choice. `max_length` is `MAX_PATH_TOKENS`, because the rows do not fit
    512. `max_microbatch_tokens` is the runbook's 6000 rather than the flag's 16384, and
    `gradient_checkpointing` is on rather than off, because a 24 GB card runs out of memory
    otherwise -- both are argued in `trainer_command`.
    """

    steps: int = 300
    seed: int = 17
    from_scratch: bool = False
    #: What the run starts from. The default is NanoJev's released `DecisionModel`, which is spec
    #: §5's warm start: its Qwen3-0.6B body *and* its trained heads, `--init-checkpoint`.
    #:
    #: Any other value is a plain Hugging Face backbone id -- `Qwen/Qwen3-4B` is a design ruling's
    #: -- and then the trainer takes upstream's *other* branch (`train_pipeline_decisions.py:462-466`):
    #: `AutoModel.from_pretrained(<id>)` for the body and **fresh heads** on top of it, sized from
    #: `backbone.config.hidden_size` (`train_toy_decisions.py:88-100`), because a 0.6B model's
    #: 1024-wide heads are not a 4B model's 2560-wide ones and there is nothing to carry over.
    #: `--head-steps` then defaults to 12 rather than 0, which is upstream's own head-only warm-up
    #: for exactly this case.
    base_model: str = BASE_CHECKPOINT_REPO
    #: The commit of `base_model` to use, or `None` for whatever its `main` is now. Those weights
    #: are unversioned and they move -- `C-Tianyu/NanoJev`'s `best.safetensors` changed under one
    #: branch during the original plan -- so a run meant to be compared with an earlier one passes the
    #: commit that run recorded in its `robojev.json`.
    base_revision: str | None = None
    data: pathlib.Path | None = None
    out: pathlib.Path | None = None
    max_length: int = MAX_PATH_TOKENS
    batch_questions: int = 12
    microbatch_questions: int = 4
    # The runbook's number, not the flag's default of 16384: see `trainer_command`.
    max_microbatch_tokens: int = 6000
    eval_every: int = 50
    # Upstream's own flag, off by default there and on by default here: see `trainer_command`.
    gradient_checkpointing: bool = True
    backbone_lr: float = 2e-5
    head_lr: float = 2e-4
    head_warmup_lr: float = 1e-3
    precision: str = "bf16"
    objective: str = "gold_distribution"
    loss: str = "ce"
    set_head: str = "attention"


# --------------------------------------------------------------------------------------------
# which question set the rows speak

#: What a manifest that says nothing speaks: the set that predates the key, which is retired.
#: **Not** `registry.DEFAULT_VERSION` -- a silent upgrade of an old directory is exactly the drift
#: `robojev.registry` exists to prevent, and an old directory is the one case where there
#: is nobody left to ask. `registry.check` refuses it by name, which is the right answer.
DEFAULT_QUESTIONS_VERSION: str = registry.UNVERSIONED

#: The manifest keys that describe how a *state was rendered* rather than how it was trained on:
#: the step scale, the tracker's settings, the bands, the memory rule and whether the per-axis
#: annotations are in the text. Copied verbatim into `robojev.json` beside the weights, because
#: `robojev/policy.py` reads them back at load and refuses to serve a checkpoint whose
#: numbers are not the ones this box would render. A key the manifest does not have is simply not
#: written: an absent key is honest where a default is not.
VOCABULARY_KEYS: tuple[str, ...] = (
    "cm_per_unit", "tracker", "steps", "bands", "annotate", "memory_rule",
)


def questions_version(manifest: dict) -> str:
    """Which question set the rows under `manifest` speak.

    Checked against `registry.VERSIONS` here, at the one moment a `TrainError` costs nothing: a
    directory naming a version this build does not know -- or one it has retired -- must not
    reach a GPU, and the manifest is already read before anything is created (`read_manifest`).
    A manifest that says nothing predates the key and is refused by name rather than upgraded.
    """
    version = manifest.get("questions_version") or DEFAULT_QUESTIONS_VERSION
    try:
        return registry.check(version)
    except ValueError as err:
        raise TrainError(
            f"{HARVEST_MANIFEST} names questions_version {version!r}, which this build does not "
            f"know how to train ({err}). Re-harvest the rows with a version in "
            f"{', '.join(registry.VERSIONS)}.",
            exit_code=2,
        ) from err


def vocabulary(manifest: dict) -> dict:
    """The block of `robojev.json` a server has to agree with before it serves these weights.

    `questions_version` and the active `qids` always, because a checkpoint served under a
    vocabulary it was not trained on answers a different question about the same state; then
    whichever of `VOCABULARY_KEYS` the harvest measured. The qids come from the manifest when it
    lists them (the harvester may have asked a subset) and from the registry otherwise -- never
    from a count, because which questions are active is a fact about the question set and moves
    with it.
    """
    version = questions_version(manifest)
    block: dict = {"questions_version": version,
                   "qids": list(manifest.get("qids") or registry.qids(version))}
    block.update({key: manifest[key] for key in VOCABULARY_KEYS
                  if manifest.get(key) is not None})
    return block


def for_rows(cfg: TrainConfig, manifest: dict) -> TrainConfig:
    """`cfg` with the path budget this manifest's question set is trained and served at.

    `--max-length` is not a tuning knob: `robojev.registry` owns it, so that the trainer
    and `robojev/policy.py` cannot disagree about how much of a candidate path the model
    ever sees (upstream raises rather than truncating, `predict_toy_decisions.py:110-112`).
    Picking it from the rows rather than from a flag is what makes "train tonight on whatever the
    harvest wrote" safe.
    """
    budget = registry.max_path_tokens(questions_version(manifest))
    return cfg if cfg.max_length == budget else dataclasses.replace(cfg, max_length=budget)


# --------------------------------------------------------------------------------------------
# the rows


#: The v2 harvest's grounding rows, merged after the task files when present.
GROUNDING_ROWS = "grounding.jsonl"


def _task_files(data_dir: pathlib.Path) -> list[pathlib.Path]:
    """`task_<i>.jsonl` in **task** order, not in `sorted()` order: `task_10` sorts between
    `task_1` and `task_2` as a string, and a suite with ten or more tasks would then be
    concatenated in an order the manifest does not describe."""
    found = []
    for path in data_dir.glob("task_*.jsonl"):
        try:
            found.append((int(path.stem.split("_", 1)[1]), path))
        except ValueError:  # pragma: no cover -- a file called task_x.jsonl
            continue
    files = [path for _, path in sorted(found)]
    # v2's expert-rollout harvest writes its BDDL grounding rows (`target`/`destination`) beside
    # the task files, with their own instruction-level split. They are training rows like any
    # other, and a merge that skips them trains a model that was never asked which bowl is meant.
    grounding = data_dir / GROUNDING_ROWS
    if files and grounding.is_file():
        files.append(grounding)
    return files


def merge_rows(data_dir, out) -> dict:
    """Concatenate every `task_*.jsonl` under `data_dir` into one `out`, and count what went in.

    The counts are by split and by question, which is what makes the two things the trainer will
    refuse visible *here*: an empty `dev` or `test` split (`:476-478`) and fewer eligible training
    questions than one batch (`:466`). Both cost a 2.4 GB model load and a tokenizer pass to
    discover upstream, and a second of stdlib to discover here.

    The one rule this re-checks rather than merely counting is the cross-split one: a `state_id`
    or a `metadata.source_group_id` that appears under two `split` values is a hard error in
    `read_training_records` (`:168-173`). The harvest splits by *task*, so it cannot happen by
    construction -- which is exactly why it is worth checking, because if it ever does happen the
    cause is a directory holding two harvests that disagree about which task is held out, and the
    message names the offending group rather than a line number in a merged file.
    """
    data_dir, out = pathlib.Path(data_dir), pathlib.Path(out)
    files = _task_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"no task_*.jsonl under {data_dir}. Harvest first:\n"
            f"    robojev harvest robojev --suite {data_dir.name}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    where: dict[tuple[str, str], tuple[str, str]] = {}   # (kind, key) -> (split, file)
    counts = {"total": 0, "by_split": {}, "by_question": {}, "by_task": {},
              "files": [f.name for f in files]}
    with out.open("w", encoding="utf-8") as sink:
        for path in files:
            task = path.stem.split("_", 1)[1] if path.name != GROUNDING_ROWS else "grounding"
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    split = row["split"]
                    for kind, key in (("state_id", row.get("state_id")),
                                      ("source_group_id", (row.get("metadata") or {}).get("source_group_id"))):
                        if key is None:
                            continue
                        seen = where.get((kind, key))
                        if seen is None:
                            where[(kind, key)] = (split, path.name)
                        elif seen[0] != split:
                            raise ValueError(
                                f"{kind} {key!r} is in split {seen[0]!r} ({seen[1]}) and split "
                                f"{split!r} ({path.name}); the trainer refuses a state or source "
                                "group that crosses splits (train_pipeline_decisions.py:168-173). "
                                "Re-harvest the directory so one task lands in one split."
                            )
                    sink.write(line if line.endswith("\n") else line + "\n")
                    counts["total"] += 1
                    counts["by_split"][split] = counts["by_split"].get(split, 0) + 1
                    counts["by_task"][task] = counts["by_task"].get(task, 0) + 1
                    for qid in row.get("questions", {}):
                        counts["by_question"][qid] = counts["by_question"].get(qid, 0) + 1
    return counts


# --------------------------------------------------------------------------------------------
# the trainer


def upstream_scripts(r: runtime.TrainerEnv) -> pathlib.Path:
    """The clone's `scripts/` -- where the trainer and every module it imports by bare name live.

    Read-only (`runtime.nanojev_clone`): `robojev train` must not be the command that silently
    clones 204 MB off GitHub. `robojev train --build-env` is, and the error says so.
    """
    pin = runtime.nanojev_pin(r)
    if pin is None:  # pragma: no cover -- only if the sidecar is removed from the recipe
        raise FileNotFoundError(f"{r.dir} pins no NanoJev checkout ({runtime.NANOJEV_PIN})")
    root = runtime.nanojev_clone(r)
    if root is None:
        raise FileNotFoundError(
            f"NanoJev's checkout is not on this box. It is pinned by {runtime.NANOJEV_PIN} in "
            f"{r.dir} ({pin[0]} @ {pin[1][:12]}) and cloned by:\n"
            f"    robojev train --build-env"
        )
    return root / pin[2]


def warm_starts_from_nanojev(cfg: TrainConfig) -> bool:
    """Whether this run loads NanoJev's released `DecisionModel` (body **and** heads).

    The one predicate three places have to agree on -- `trainer_command` (which flags), `train`
    (whether to download 2.4 GB) and `decision.slurm` (whether the job script downloads it on the
    node) -- so it is written once. False for `--from-scratch` and false for any `base_model`
    that is not the NanoJev repository, because those are backbone ids and a backbone has no
    heads to warm-start from.
    """
    return not cfg.from_scratch and cfg.base_model == BASE_CHECKPOINT_REPO


def backbone_for(cfg: TrainConfig) -> tuple[str, str]:
    """The `--model` / `--revision` pair the trainer is given.

    Under a NanoJev warm start both are inert -- the tokenizer and the body config come out of the
    checkpoint (`train_pipeline_decisions.py:448-456`) -- but upstream records them in its own
    `config.json`, so they stay the runbook's pinned pair rather than becoming a lie. Under any
    other `base_model` they are the *whole* of what the backbone is, and `--revision` defaults to
    `main` exactly as upstream's flag does; the commit the hub resolved is recorded afterwards as
    `resolved_model_revision`.
    """
    if warm_starts_from_nanojev(cfg) or cfg.from_scratch:
        # `base_revision` is a commit of `base_model`, and `base_model` here is the *NanoJev
        # repository*, not the backbone -- so it pins the warm start (`base_checkpoint`) and
        # leaves the runbook's backbone pin alone. Under `--from-scratch` there is no warm start
        # to pin and the runbook's pair is the whole answer.
        return BACKBONE_MODEL, BACKBONE_REVISION
    return cfg.base_model, cfg.base_revision or "main"


def base_checkpoint(log=log_to_stderr, revision: str | None = None) -> tuple[pathlib.Path, str]:
    """`C-Tianyu/NanoJev` in the Hugging Face cache, fetching it once, and its resolved revision.

    Only the four entries upstream's own loader requires (`local_checkpoint_files`,
    `predict_toy_decisions.py:149-162`) are fetched: the repository also carries the artefacts of
    the runs that made it, and none of them is a warm start. 2.39 GB of fp32 weights; the trainer
    copies the tokenizer and the backbone config straight out of it into our checkpoint, which is
    why the fine-tune needs no `--revision` of its own (`:448-456`).
    """
    from huggingface_hub import snapshot_download

    log(f"warm start: {BASE_CHECKPOINT_REPO}@{revision or 'main'} "
        f"(2.4 GB, into the Hugging Face cache, once)")
    directory = snapshot_download(
        repo_id=BASE_CHECKPOINT_REPO,
        revision=revision,
        allow_patterns=["config.json", "best.safetensors", "backbone_config/*", "tokenizer/*"],
    )
    directory = pathlib.Path(directory)
    # The snapshot directory is named for the commit the hub resolved, which is the only identity
    # these unversioned weights have; `robojev.json` records it so a retrain can be compared.
    return directory, directory.name


def trainer_command(r: runtime.TrainerEnv, rows, out_dir, cfg: TrainConfig,
                    init_checkpoint=None) -> tuple[list[str], str]:
    """NanoJev's own invocation, and the directory to run it from. Every flag is upstream's.

    * `--model` / `--revision` -- `backbone_for(cfg)`. The runbook's pinned Qwen3-0.6B (`:19-29`)
      under a NanoJev warm start, where both are inert but recorded; `cfg.base_model` and its
      revision when the run starts from a bare backbone instead (a design ruling's `Qwen/Qwen3-4B`),
      where they are the whole of what gets built.
    * `--init-checkpoint` -- a warm start with a **fresh optimizer**, not a resume, and it cannot
      change `set_head` (`:448-453`); `--set-head attention` is therefore what the released
      checkpoint already is, asserted rather than chosen. Passed only when
      `warm_starts_from_nanojev`; without it upstream builds the body from `--model` and the heads
      fresh, which is the whole of what "a 4B backbone" means here.
    * `--objective gold_distribution` -- the harvested rows carry `gold_probs` (docs/DESIGN.md): ρ-softened
      distributions for the two reference axes and the magnitude, a measured frequency for `grip`.
      `observed_outcome` would throw away three questions of every four, and `teacher` wants an
      API labeller this project does not use.
    * `--loss ce` -- `−Σ q log softmax(z)` per complete question, upstream's default and the loss
      every released `gold_distribution` run used.
    * `--max-length` -- `cfg.max_length`, which `for_rows` picks from the harvest manifest's
      `questions_version` rather than from a flag: see `MAX_PATH_TOKENS`.
    * `--batch-questions 12 --microbatch-questions 4` -- upstream's defaults: twelve complete
      questions per optimizer update, at most four of them in one backward.
    * `--max-microbatch-tokens 6000` -- **the runbook's number** (`pipeline_runbook.md:131-141`
      and `:194-204`), not the flag's own default of 16384. The cap is the microbatch's padded
      budget, `total candidate paths × longest padded path` (`pack_complete_questions`,
      `:236-250`): a 7-candidate LIBERO-Spatial question at 534 tokens is 3,738 padded tokens, so
      6000 admits one such question per backward and 16384 admits four. Four fits the *cap* and
      does not fit a 24 GB card -- fp32 parameters plus fp32 gradients plus AdamW's two fp32
      moments are already ~10 GB of the 0.6B model, and 15,000 tokens of activations on top of
      that is an out-of-memory error on the box this is trained on. Upstream's receipts are all
      80 GB A100s; its own runbook runs at 6000, and so does this. The effective batch is
      unchanged -- `--batch-questions` is what the loss is divided by, and the packer only decides
      how many backward passes it takes to get there.
      A single question over the cap is an error rather than a truncation (`:243-244`), which is
      why the number has to be checked against the real rows rather than assumed.
    * `--gradient-checkpointing` -- upstream's own flag, which it documents as "reduce activation
      memory for complete long maze inputs". A LIBERO-Spatial path is one of those long inputs.
      Without it a 6000-token microbatch peaks over 20 GB and a 24 GB card runs out on the second
      optimizer step; with it the run peaks at 11.9 GB -- which is where every one of upstream's
      own A100 receipts peaks too. Default **on** here, because the card this trains on is the
      card the worker serves on and a run that needs the whole of it is a run that cannot be made.
    * `--eval-every` -- upstream's default of 50 is a per-*step* interval and the cost of an
      evaluation is a property of the dev *split*: LIBERO-Spatial holds out a whole task, 4,844
      complete questions, where the runbook's own dev splits are a few hundred. At the measured
      rate an evaluation is two minutes longer than fifty training steps, so leaving it at 50
      would spend more of the run evaluating than training. `TrainConfig` keeps upstream's 50 and
      the caller raises it.
    * `--precision bf16` -- fp32 parameters, bf16 forward autocast; the same arithmetic the server
      serves under.

    `cwd` is the clone's `scripts/`, because that is the directory upstream's flat sibling imports
    (`calibrated_objectives`, `train_toy_decisions`, `predict_toy_decisions`) assume. The argv
    comes from `runtime.run_command(..., entry=…)`, so the interpreter is the trainer's Python 3.14
    and not whichever one is running the CLI.
    """
    scripts = upstream_scripts(r)
    model, revision = backbone_for(cfg)
    args = [
        "--input", str(rows),
        "--output-dir", str(out_dir),
        "--model", model,
        "--revision", revision,
        *(() if init_checkpoint is None else ("--init-checkpoint", str(init_checkpoint))),
        "--objective", cfg.objective,
        "--loss", cfg.loss,
        "--set-head", cfg.set_head,
        "--steps", str(cfg.steps),
        "--batch-questions", str(cfg.batch_questions),
        "--microbatch-questions", str(cfg.microbatch_questions),
        "--max-microbatch-tokens", str(cfg.max_microbatch_tokens),
        "--eval-every", str(cfg.eval_every),
        *(("--gradient-checkpointing",) if cfg.gradient_checkpointing else ()),
        "--max-length", str(cfg.max_length),
        "--backbone-lr", repr(cfg.backbone_lr),
        "--head-lr", repr(cfg.head_lr),
        "--head-warmup-lr", repr(cfg.head_warmup_lr),
        "--precision", cfg.precision,
        "--seed", str(cfg.seed),
    ]
    argv, _ = runtime.run_command(r, args, entry=scripts / TRAINER)
    return argv, str(scripts)


def validate_command(r: runtime.TrainerEnv, rows) -> tuple[list[str], str]:
    """`--validate-only`: upstream's stdlib schema/split/target audit, no tokenizer and no GPU
    (`:409,421-423`). It is the cheapest true statement about a harvest -- the design notes's first proof,
    "the row format ... its trainer accepts the file" -- and it is the reason the `train` verb has
    a mode that runs on a box with the card busy."""
    scripts = upstream_scripts(r)
    argv, _ = runtime.run_command(r, ["--input", str(rows), "--validate-only"],
                                  entry=scripts / TRAINER)
    return argv, str(scripts)


# --------------------------------------------------------------------------------------------
# the numbers upstream does not report


def eval_table(predictions, summary=None) -> dict:
    """Per-question accuracy and Brier from `predictions_test.jsonl` (a design ruling).

    * **accuracy** -- `argmax(student_probs) == gold_index`, the fraction over the rows of that
      question that carry a hard `gold_index`. For `translate`/`rotate`/`magnitude` that is the
      demonstrated axis; for `grip` it is one observed draw from the measured frequency, so a
      perfect model does *not* score 1.0 on it and the Brier beside it is the honest number.
    * **brier** -- `Σ_i (student_probs_i − gold_probs_i)²` per question, averaged. Over the target
      distribution the row actually carries, not a one-hot: the whole point of the ρ-softened
      targets is that the model should predict 0.85/0.025…, and scoring it against a one-hot would
      reward exactly the overconfidence NanoJev exists to avoid.
    * **target_ce** -- not recomputed: copied from `summary.json`'s `metrics_by_split.test` when a
      summary is given, so the table's own column and upstream's selection metric are the same
      number rather than two roundings of it. It is a per-split figure, so it sits on the total
      row.

    A row whose question has no target distribution contributes to `n` and to accuracy but not to
    Brier, and `brier` is `None` when no row of that question had one.
    """
    predictions = pathlib.Path(predictions)
    by_qid: dict[str, dict] = {}
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cell = by_qid.setdefault(row["qid"], {"n": 0, "hits": 0, "graded": 0,
                                              "brier_sum": 0.0, "scored": 0})
        cell["n"] += 1
        probs = row["student_probs"]
        gold_index = row.get("gold_index")
        if gold_index is not None:
            cell["graded"] += 1
            cell["hits"] += int(max(range(len(probs)), key=probs.__getitem__) == gold_index)
        target = row.get("gold_distribution_probs")
        if target is None and isinstance(row.get("gold_probs"), dict):
            # The row's own `gold_probs` is keyed by candidate id; put it in the model's order.
            target = [row["gold_probs"].get(c, 0.0) for c in row["candidate_ids"]]
        if target is not None:
            cell["brier_sum"] += sum((p - q) ** 2 for p, q in zip(probs, target))
            cell["scored"] += 1

    table: dict[str, dict] = {}
    for qid, cell in by_qid.items():
        table[qid] = {
            "n": cell["n"],
            "accuracy": (cell["hits"] / cell["graded"]) if cell["graded"] else None,
            "brier": (cell["brier_sum"] / cell["scored"]) if cell["scored"] else None,
            "graded": cell["graded"],
        }
    total = {
        "n": sum(c["n"] for c in by_qid.values()),
        "graded": sum(c["graded"] for c in by_qid.values()),
        "accuracy": None,
        "brier": None,
    }
    graded = sum(c["graded"] for c in by_qid.values())
    scored = sum(c["scored"] for c in by_qid.values())
    if graded:
        total["accuracy"] = sum(c["hits"] for c in by_qid.values()) / graded
    if scored:
        total["brier"] = sum(c["brier_sum"] for c in by_qid.values()) / scored
    if summary:
        # `evaluate_pipeline` names them `target_ce`/`target_kl`/`target_tv` at the split level
        # (the bare `ce`/`kl`/`tv` spelling is only inside `by_target_kind`), so read that.
        metrics = (summary.get("metrics_by_split") or {}).get("test") or {}
        total["target_ce"] = metrics.get("target_ce")
        total["target_kl"] = metrics.get("target_kl")
        total["target_tv"] = metrics.get("target_tv")
    return {"by_question": table, "total": total, "predictions": predictions.name}


#: The order questions are printed in: the question set's own, so the table reads the way the
#: console renders the bars, with anything unexpected after it rather than silently dropped.
def _qid_order(table: dict) -> list[str]:
    known = [q for q in registry.qids(registry.DEFAULT_VERSION) if q in table]
    return known + sorted(q for q in table if q not in known)


def format_table(table: dict) -> str:
    """The eval table as text, two decimals, one row per question plus a total.

    Printed on stdout by `robojev train` because it is the answer to the question the command was
    run to ask. `None` prints as `-`: a question with no hard gold has no accuracy, and saying so
    is better than printing 0.00.
    """
    def cell(value) -> str:
        return "    -" if value is None else f"{value:5.2f}"

    lines = [f"{'question':<10} {'n':>7} {'accuracy':>9} {'brier':>7}"]
    for qid in _qid_order(table["by_question"]):
        row = table["by_question"][qid]
        lines.append(f"{qid:<10} {row['n']:>7} {cell(row['accuracy']):>9} {cell(row['brier']):>7}")
    total = table["total"]
    lines.append(f"{'all':<10} {total['n']:>7} {cell(total['accuracy']):>9} {cell(total['brier']):>7}")
    if total.get("target_ce") is not None:
        lines.append(f"test target CE {total['target_ce']:.4f} (upstream's own selection metric)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# the command


def run_trainer(argv: list[str], cwd: str, log=log_to_stderr) -> str:
    """Run `argv` in `cwd`, forwarding its stdout to **stderr** a line at a time; return the last.

    Upstream prints one JSON object per logged step and one `{"done": …}` at the end. Those are
    progress, and this command's stdout is the eval table, so they are relayed rather than
    swallowed (a training run is minutes long and a silent one is indistinguishable from a hung
    one). Only the **last** is kept -- it is upstream's own receipt for the run, and it goes into
    `robojev.json`; buffering the other 1,200 would be a list nobody reads.

    A nonzero exit raises `CalledProcessError`, which is what keeps `train`'s `except` -- and so
    the removal of the half-written directory -- on the path of every failure.
    """
    # Long padded microbatches of varying width fragment the caching allocator badly enough to
    # fail an allocation with gigabytes still free; expandable segments is torch's own answer and
    # is what its OOM message recommends. Only set when the caller has not, so an operator
    # debugging an allocator problem keeps the last word.
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    last = ""
    with subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          text=True, bufsize=1) as child:
        for line in child.stdout:
            line = line.rstrip("\n")
            if line.strip():
                last = line
            print(line, file=sys.stderr, flush=True)
    if child.returncode:
        raise subprocess.CalledProcessError(child.returncode, argv)
    return last


def _read_json(path: pathlib.Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def data_root(policy: str, suite: str) -> pathlib.Path:
    """`$ROBOJEV_HOME/data/<policy>/<suite>` -- `dataset.out_root`'s directory."""
    return home.data_dir(policy, suite)


def checkpoint_root(policy: str, suite: str) -> pathlib.Path:
    """`$ROBOJEV_HOME/checkpoints/<policy>/<suite>` -- exactly `runtime.local_checkpoint_dir` of the
    registry's `{"local": "<policy>/<suite>"}`, so what this writes is what a
    later verification reads."""
    return home.checkpoints_dir() / policy / suite


def _backbone_from_trainer_config(config: dict) -> str | None:
    """`"<model>@<commit>"` out of upstream's own `config.json`, or `None`.

    The fallback for a `--finish` over a directory whose train-side context is gone: upstream
    records `model` (what it was asked for) and `resolved_model_revision` (what the hub gave it),
    which is the same pair `train` writes and a strictly better source for the second half of it.
    """
    model = config.get("model")
    if not model:
        return None
    resolved = config.get("resolved_model_revision") or config.get("revision")
    return f"{model}@{resolved}" if resolved else str(model)


def read_manifest(data_dir: pathlib.Path) -> dict:
    """The harvest manifest, or a `TrainError`. Never a default.

    `robojev.json` exists so a served checkpoint knows the step size its own `translate` answer
    means, and δ_t/δ_r come from exactly one place: the manifest `robojev harvest` wrote beside the
    rows it measured them over. A run given rows without their manifest would train perfectly
    happily, pass a later verification, and then compose every action at a default -- a
    policy steering at the wrong magnitude, with nothing anywhere saying so. `--data <a directory of task files someone copied>` is not a hypothetical: it is the
    shape this suite's own parallel harvest produced before its parts were assembled.

    So: refused, before the trainer is launched rather than after, and refused as a configuration
    error (exit 2) because the fix is to point `--data` somewhere else or to re-harvest.
    """
    path = pathlib.Path(data_dir) / HARVEST_MANIFEST
    manifest = _read_json(path)
    if not manifest:
        raise TrainError(
            f"no readable {HARVEST_MANIFEST} in {data_dir}. It is where delta_t/delta_r come from, "
            f"and a checkpoint trained without them serves every action at the wrong scale. "
            f"Harvest the rows and their manifest together:\n"
            f"    robojev harvest robojev --suite {pathlib.Path(data_dir).name}",
            exit_code=2,
        )
    missing = [k for k in ("delta_t", "delta_r") if manifest.get(k) is None]
    if missing:
        raise TrainError(
            f"{path} has no {', '.join(missing)}. Those are the measured step sizes the server "
            f"composes actions with; re-harvest rather than train against a manifest without them.",
            exit_code=2,
        )
    return manifest


def _context(work: pathlib.Path) -> pathlib.Path:
    return work / TRAIN_CONTEXT


def _infer_context(work: pathlib.Path, data=None) -> dict:
    """What `finish` needs, reconstructed from a `.tmp-` directory that carries no context file.

    A run started before the context existed -- or one whose context was lost -- is still a
    directory full of finished weights, and refusing to describe it would be the same mistake as
    deleting it. The name is the whole of the convention: `<checkpoints>/<policy>/<suite>.tmp-
    <stamp>`, so the policy, the suite, the destination and the stamp all read straight off the
    path. Everything else is provenance rather than structure, and is recorded as unknown (`None`)
    rather than guessed; the trainer's own `config.json` is beside the weights either way.
    """
    name, _, stamp = work.name.partition(".tmp-")
    if not stamp:
        raise TrainError(
            f"{work} is not a <suite>.tmp-<stamp> directory and has no {TRAIN_CONTEXT}, so there "
            f"is nothing here that says what was trained or where it should go.",
            exit_code=2,
        )
    policy = work.parent.name
    return {"policy": policy, "suite": name, "stamp": stamp,
            "out": str(work.parent / name),
            "data": str(pathlib.Path(data) if data else data_root(policy, name)),
            "trainer": _read_json(work / "config.json", {}).get("_argv"),
            "trainer_cwd": None, "rows": None, "base_checkpoint": None,
            # Upstream's own config.json is the fallback source: it records the backbone id it
            # was given and the commit the hub resolved for it.
            "base_model": _backbone_from_trainer_config(_read_json(work / "config.json", {}) or {}),
            "max_path_tokens": (_read_json(work / "config.json", {}) or {}).get("max_length"),
            "inferred_from_the_directory_name": True}


def train(policy: str, suite: str, cfg: TrainConfig, log=log_to_stderr) -> dict:
    """Merge, train, then `finish`. Returns the summary and writes it into the checkpoint.

    The order is what makes an interrupted run harmless. Everything lands in a sibling
    `<suite>.tmp-<stamp>/` directory: upstream's own refusal to overwrite a checkpoint (`:433`)
    stays satisfied without this command ever having to delete weights, a crash leaves a `.tmp-`
    directory that nothing reads, and the rename onto `<suite>/` is the single atomic moment at
    which this box has a trained RoboJEV.

    **The cleanup covers the trainer and nothing after it.** Up to the point where the four files
    upstream's own loader requires are all in `work`, a failure means there is no checkpoint and
    the half-written directory is removed. After that point `work` holds hours of a shared GPU, and
    *no* failure -- a full disk while `predictions_test.jsonl` is flushed, a key `eval_table`
    indexes that upstream renamed, a Ctrl-C during the `robojev.json` write -- is worth deleting it
    for. Those failures leave the directory exactly where it is and say so, and `finish` picks it
    up: everything after the trainer is a pure function of what is already on disk.
    """
    r = runtime.trainer()
    data_dir = pathlib.Path(cfg.data) if cfg.data else data_root(policy, suite)
    final = pathlib.Path(cfg.out) if cfg.out else checkpoint_root(policy, suite)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = final.parent / f"{final.name}.tmp-{stamp}"
    # Read before anything is created: a missing manifest must cost nothing, least of all a GPU.
    manifest = read_manifest(data_dir)
    # ... and the question set the rows speak decides the path budget, so a v2 harvest trains at
    # v2's 1024 without anybody having to remember a flag.
    cfg = for_rows(cfg, manifest)
    log(f"{questions_version(manifest)} rows: --max-length {cfg.max_length}")
    work.mkdir(parents=True, exist_ok=False)
    # The merged rows are the trainer's *input*, not part of its output: 58 MB of a copy of the
    # harvest, which would otherwise be renamed into the published checkpoint and carried to every
    # box the weights go to. They live in a sibling scratch directory that goes away with the run.
    scratch = final.parent / f"{final.name}.tmp-{stamp}.rows"
    scratch.mkdir(parents=True, exist_ok=False)

    try:
        rows = scratch / ROWS
        counts = merge_rows(data_dir, rows)
        log(f"{counts['total']} rows from {len(counts['files'])} task file(s) -> {rows} "
            f"({counts['by_split']})")
        init, base_revision = ((base_checkpoint(log=log, revision=cfg.base_revision))
                               if warm_starts_from_nanojev(cfg) else (None, None))
        argv, cwd = trainer_command(r, rows, work, cfg, init_checkpoint=init)
        model, model_revision = backbone_for(cfg)
        # Written before the trainer starts, so `--finish` can pick the directory up after a crash
        # in the post-training half without being told any of this again.
        _context(work).write_text(json.dumps({
            "policy": policy, "suite": suite, "data": str(data_dir), "out": str(final),
            "stamp": stamp, "started_at": time.time(), "trainer": argv, "trainer_cwd": cwd,
            "rows": counts, "max_path_tokens": cfg.max_length,
            "base_checkpoint": None if init is None else f"{BASE_CHECKPOINT_REPO}@{base_revision}",
            # The backbone, always -- under a NanoJev warm start it is what that checkpoint's body
            # is, and otherwise it is what the run built from scratch. `robojev.json` carries it so
            # a served checkpoint says which model it is a fine-tune of without opening 16 GB of
            # weights to find out.
            "base_model": f"{model}@{model_revision}",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log("running NanoJev's own trainer: " + " ".join(argv))
        done_line = run_trainer(argv, cwd, log=log)
        # Inside the guard on purpose: a trainer that exits 0 without writing weights has produced
        # nothing worth keeping, and this is the last moment at which that is true. Everything
        # after it is `finish`, which never deletes.
        missing = runtime.missing_checkpoint_files(work)
        if missing:
            raise TrainError(
                f"the trainer exited 0 but {work} is missing {', '.join(missing)}; there is no "
                f"checkpoint here, so it was removed rather than renamed into place."
            )
    except BaseException:
        # Nothing trained, or half of it: a directory a later verification might accept is worse than no
        # directory. This is the *only* place `work` is removed.
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    shutil.rmtree(scratch, ignore_errors=True)
    # Upstream's own last word on the run, recorded now because `finish` may be re-run tomorrow.
    _context(work).write_text(json.dumps(
        {**_read_json(_context(work), {}), "trainer_done": done_line},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return finish(work, log=log)


def finish(work, data=None, log=log_to_stderr) -> dict:
    """Everything after the trainer, over a `.tmp-` directory that already holds the weights.

    Split out of `train` so that it can be re-run: `robojev train <policy> --finish <dir>`, with
    `--data` when the harvest is not where the run left it. It reads
    `summary.json` and `predictions_test.jsonl`, computes the eval table, writes `robojev.json` and
    the harvest manifest beside the weights, renames the directory into place and returns the
    summary. Every input is already on disk, so running it twice costs seconds and needs no GPU.

    It never deletes `work`. A failure here is a failure to *describe* a checkpoint that exists.
    """
    work = pathlib.Path(work)
    context = _read_json(_context(work))
    if not context:
        context = _infer_context(work, data=data)
        log(f"no {TRAIN_CONTEXT} in {work}: reading policy/suite/destination off the directory "
            f"name and the harvest from {context['data']}")
    elif data is not None:
        context = {**context, "data": str(data)}
    missing = runtime.missing_checkpoint_files(work)
    if missing:
        raise TrainError(
            f"{work} is missing {', '.join(missing)}, so it is not a checkpoint: the trainer "
            f"either has not run or did not finish. Nothing was renamed into place and nothing "
            f"was deleted."
        )

    final = pathlib.Path(context["out"])
    data_dir = pathlib.Path(context["data"])
    stamp = context["stamp"]
    try:
        summary = _read_json(work / "summary.json", {}) or {}
        table = eval_table(work / "predictions_test.jsonl", summary)
        manifest = read_manifest(data_dir)
        meta = {
            **manifest,
            # Normalised over the manifest's own keys: `questions_version` is present even when
            # the harvest predates it, and `qids` name the questions rather than counting them.
            **vocabulary(manifest),
            "policy": context["policy"],
            "suite": context["suite"],
            "trainer": context["trainer"],
            "trainer_cwd": context["trainer_cwd"],
            "rows": context["rows"],
            "base_checkpoint": context["base_checkpoint"],
            "base_model": context.get("base_model"),
            "max_path_tokens": context["max_path_tokens"],
            "eval": table,
            "training_seconds": summary.get("training_seconds"),
            "best_step": summary.get("best_step"),
            "best_dev_target_ce": summary.get("best_dev_target_ce"),
            "max_gpu_allocated_gb": summary.get("max_gpu_allocated_gb"),
            "trainer_done": context.get("trainer_done"),
            # True only for a directory `finish` had to read off its own name, so a checkpoint
            # whose provenance is partly unknown says so rather than looking complete.
            "inferred_from_the_directory_name": bool(
                context.get("inferred_from_the_directory_name")),
            "harvest_manifest": HARVEST_MANIFEST,
            "trained_at": datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
        (work / CHECKPOINT_META).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
        shutil.copyfile(data_dir / HARVEST_MANIFEST, work / HARVEST_MANIFEST)
    except TrainError:
        raise
    except BaseException as exc:
        raise TrainError(
            f"the weights in {work} are finished and have NOT been touched, but describing them "
            f"failed ({type(exc).__name__}: {exc}). Fix the cause and re-run:\n"
            f"    robojev train {context['policy']} --suite {context['suite']} --finish {work}"
        ) from exc

    if final.exists():
        # Only reached once training has succeeded: the previous checkpoint is kept beside the
        # new one rather than deleted, because it is the only copy of weights nothing published.
        retired = final.parent / f"{final.name}.was-{stamp}"
        final.rename(retired)
        log(f"the previous checkpoint is now {retired}")
    _context(work).unlink(missing_ok=True)
    work.rename(final)

    revision = runtime.local_checkpoint_revision(final)
    meta["revision"] = revision
    (final / CHECKPOINT_META).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    took = time.time() - float(context.get("started_at") or time.time())
    log(f"{final} in {took:.0f}s")
    return {"checkpoint": str(final), "revision": revision, "rows": context["rows"],
            "eval": table, "summary": summary, "trainer": context["trainer"], "meta": meta}
