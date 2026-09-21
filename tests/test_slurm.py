"""`robojev train robojev --slurm` -- everything about the cluster run that is not the cluster.

No ssh, no sbatch, no GPU: what is pinned here is the job script's text, the remote path layout,
the two rsync commands, the poll and its parses, and that the dry run touches nothing. The one
claim these cannot make is that the job runs, which is why task 7d's report carries a real one.
"""
from __future__ import annotations

import json
import dataclasses
import pathlib
import shlex
import subprocess

import pytest

from robojev import cli, runtime
from robojev import slurm
from robojev import train as trainer


#: A cluster for the tests to point at. `host`, `root` and `partition` have no defaults -- naming
#: one particular site in the package is exactly what this repository must not do -- so every test
#: that builds a config goes through here and the three values are visibly made up.
HOST = "login.cluster.example"
ROOT = "/scratch/robojev"
PARTITION = "gpu"


def config(**kw):
    kw.setdefault("host", HOST)
    kw.setdefault("root", ROOT)
    kw.setdefault("partition", PARTITION)
    kw.setdefault("qos", "normal")
    return slurm.SlurmConfig(**kw)



@pytest.fixture
def robojev(tmp_path, monkeypatch):
    """The real recipe, with a stand-in NanoJev checkout under an isolated `$ROBOJEV_HOME`.

    `trainer_flags` asks `train.trainer_command` for the flags, and that resolves the clone
    (`runtime.nanojev_clone`) before it will build anything. The clone's *contents* are never read --
    only `scripts/` has to exist -- so an empty directory at the pinned commit is the whole
    fixture, and the flags it produces are the flags a box with the real 204 MB clone produces.
    """
    monkeypatch.setenv("ROBOJEV_HOME", str(tmp_path / "home"))
    r = runtime.trainer()
    url, commit, subdir = runtime.nanojev_pin(r)
    (tmp_path / "home" / "src" / f"nanojev@{commit[:12]}" / subdir).mkdir(parents=True)
    return r


@pytest.fixture
def place(robojev):
    return slurm.layout(robojev, config(), "libero_spatial-20260920T120000Z")


# -- the configuration ---------------------------------------------------------------------------


def test_the_cluster_raises_the_microbatch_cap_and_drops_the_recompute():
    cfg = slurm.cluster_config(trainer.TrainConfig())
    assert cfg.max_microbatch_tokens == slurm.CLUSTER_MICROBATCH_TOKENS == 32768
    assert cfg.gradient_checkpointing is False


def test_an_explicit_microbatch_cap_survives_the_cluster_defaults():
    """`--max-microbatch-tokens 6000 --slurm` is how the two cards are compared step for step."""
    asked = trainer.TrainConfig(max_microbatch_tokens=6000, gradient_checkpointing=True)
    assert asked.max_microbatch_tokens == trainer.TrainConfig().max_microbatch_tokens
    # 6000 *is* the default, so it cannot be told apart -- but anything else must survive:
    cfg = slurm.cluster_config(trainer.TrainConfig(max_microbatch_tokens=8192))
    assert cfg.max_microbatch_tokens == 8192
    assert cfg.gradient_checkpointing is False
    kept = slurm.cluster_config(trainer.TrainConfig(gradient_checkpointing=False))
    assert kept.gradient_checkpointing is False


def test_a_pinned_flag_survives_even_when_it_equals_the_default():
    """The gap the value comparison cannot close: `gradient_checkpointing=True` is both the local
    default and what a 4B backbone needs on a 141 GB card, so only "the operator typed it" can
    tell the two apart (spec ruling 13's run, which OOMs without the recompute)."""
    cfg = slurm.cluster_config(trainer.TrainConfig(), pinned={"gradient_checkpointing"})
    assert cfg.gradient_checkpointing is True
    assert cfg.max_microbatch_tokens == slurm.CLUSTER_MICROBATCH_TOKENS   # the other still applies
    both = slurm.cluster_config(trainer.TrainConfig(max_microbatch_tokens=6000),
                                pinned={"max_microbatch_tokens", "gradient_checkpointing"})
    assert (both.max_microbatch_tokens, both.gradient_checkpointing) == (6000, True)


def test_a_4b_run_asks_the_node_for_no_warm_start_and_downloads_none(robojev, place):
    """Spec ruling 13: the body comes from the hub through upstream's own `--model` branch, so
    there is no `DecisionModel` directory to fetch and no `--init-checkpoint` to point at one."""
    cfg = trainer.TrainConfig(steps=10, base_model="Qwen/Qwen3-4B")
    flags = slurm.trainer_flags(robojev, cfg, place)
    assert "--init-checkpoint" not in flags
    assert flags[flags.index("--model") + 1] == "Qwen/Qwen3-4B"
    script = slurm.render_sbatch(robojev, cfg, config(), place)
    assert "snapshot_download" not in script and "INIT_CHECKPOINT" not in script
    assert trainer.BASE_CHECKPOINT_REPO not in script
    # ... while the default run still downloads it on the node, into the shared HF_HOME.
    normal = slurm.render_sbatch(robojev, trainer.TrainConfig(steps=10), config(), place)
    assert "snapshot_download" in normal and trainer.BASE_CHECKPOINT_REPO in normal


def test_the_trainers_own_base_revision_reaches_the_node_when_slurms_is_not_given(robojev, place):
    cfg = trainer.TrainConfig(steps=10, base_revision="feedface")
    assert "feedface" in slurm.render_sbatch(robojev, cfg, config(), place)
    # and the cluster-specific flag wins when both are given, because only the node downloads it
    script = slurm.render_sbatch(robojev, cfg, config(base_revision="0ddba11"), place)
    assert "0ddba11" in script and "feedface" not in script


def test_nothing_else_about_the_run_changes_on_the_cluster():
    """Three fields, and each is a fact about the *card* or about what a run on a preempting
    partition can afford -- never a tuning choice. Everything that decides what the model learns
    (the learning rates, the objective, the loss, the seed, both batch numbers, the path budget)
    is the same number on the cluster as on the local card, or the two runs are not comparable."""
    local, remote = trainer.TrainConfig(steps=3350, seed=17), None
    remote = slurm.cluster_config(local)
    changed = {f.name for f in dataclasses.fields(local)
               if getattr(local, f.name) != getattr(remote, f.name)}
    assert changed == {"max_microbatch_tokens", "gradient_checkpointing", "eval_every"}


# -- the remote layout ---------------------------------------------------------------------------


def test_every_remote_path_is_under_the_root_and_none_is_under_home(place):
    root = place.root
    assert root == ROOT
    for path in (place.envs, place.env, place.venv, place.src, place.scripts, place.hf,
                 place.uv_cache, place.uv_python, place.run, place.data, place.rows,
                 place.checkpoint, place.logs, place.script, place.log("271087")):
        assert path.startswith(root + "/"), path
        assert "$HOME" not in path and "/home/" not in path


def test_the_env_is_named_for_the_cu126_lock_and_the_clone_for_the_pin(place, robojev):
    project = slurm.cluster_project(robojev)
    assert place.env_name == f"robojev@{slurm.lock_hash(project)[:12]}"
    assert place.env_name != robojev.env_name, "the cluster env must not collide with the local one"
    _, commit, subdir = runtime.nanojev_pin(robojev)
    assert place.nanojev_name == f"nanojev@{commit[:12]}"
    assert place.scripts.endswith(f"/{subdir}")


def test_the_run_directory_carries_the_run_id_and_the_log_the_job_id(place):
    assert place.run.endswith("/runs/libero_spatial-20260920T120000Z")
    assert place.log("271087") == place.logs + "/slurm-271087.out"


def test_a_recipe_without_a_cluster_project_is_a_configuration_error(tmp_path, robojev):
    stripped = dataclasses.replace(robojev, dir=tmp_path)
    with pytest.raises(slurm.SlurmError) as caught:
        slurm.cluster_project(stripped)
    assert caught.value.exit_code == 2
    assert "uv.lock" in str(caught.value)


# -- the trainer's own flags ---------------------------------------------------------------------


def test_the_cluster_runs_the_flags_train_py_builds_not_a_second_list(robojev, place):
    """The point of the module: the argument list is `train.trainer_command`'s, asked for."""
    cfg = slurm.cluster_config(trainer.TrainConfig(steps=3350, eval_every=335))
    # `--steps` is the one value the *script* decides (a resumed attempt asks for what is left),
    # so it travels as a placeholder; everything else is compared token for token.
    import dataclasses as _d
    argv, _ = trainer.trainer_command(
        robojev, place.rows, place.checkpoint, _d.replace(cfg, steps=slurm.STEPS_PLACEHOLDER),
        init_checkpoint=slurm.INIT_PLACEHOLDER)
    assert slurm.trainer_flags(robojev, cfg, place) == argv[argv.index(
        next(a for a in argv if a.endswith(trainer.TRAINER))) + 1:]
    assert slurm.STEPS_PLACEHOLDER in slurm.trainer_flags(robojev, cfg, place)


def test_the_flags_differ_from_the_3090s_only_in_the_two_memory_numbers(robojev, place):
    local = trainer.TrainConfig(steps=3350, eval_every=335)
    here = slurm.trainer_flags(robojev, local, place)
    there = slurm.trainer_flags(robojev, slurm.cluster_config(local), place)
    assert "--gradient-checkpointing" in here and "--gradient-checkpointing" not in there
    assert here[here.index("--max-microbatch-tokens") + 1] == "6000"
    assert there[there.index("--max-microbatch-tokens") + 1] == "32768"
    strip = [f for f in here if f != "--gradient-checkpointing"]
    strip[strip.index("--max-microbatch-tokens") + 1] = "32768"
    assert strip == there


def test_the_flags_point_at_the_remote_rows_and_the_remote_output(robojev, place):
    flags = slurm.trainer_flags(robojev, slurm.cluster_config(trainer.TrainConfig()), place)
    assert flags[flags.index("--input") + 1] == place.rows
    assert flags[flags.index("--output-dir") + 1] == place.checkpoint
    assert flags[flags.index("--init-checkpoint") + 1] == slurm.INIT_PLACEHOLDER
    assert flags[flags.index("--max-length") + 1] == str(trainer.MAX_PATH_TOKENS)


def test_from_scratch_asks_for_no_warm_start_anywhere(robojev, place):
    cfg = slurm.cluster_config(trainer.TrainConfig(from_scratch=True))
    assert "--init-checkpoint" not in slurm.trainer_flags(robojev, cfg, place)
    script = slurm.render_sbatch(robojev, cfg, config(), place)
    assert trainer.BASE_CHECKPOINT_REPO not in script
    assert "snapshot_download" not in script


# -- the job script ------------------------------------------------------------------------------


def test_the_script_asks_slurm_for_the_probes_answers(robojev, place):
    script = slurm.render_sbatch(robojev, slurm.cluster_config(trainer.TrainConfig()),
                                 config(), place)
    for directive in (f"#SBATCH --partition={PARTITION}", "#SBATCH --qos=normal",
                      "#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=16",
                      "#SBATCH --mem=64G", "#SBATCH --time=06:00:00",
                      "#SBATCH --signal=TERM@60", "#SBATCH --requeue",
                      f"#SBATCH --chdir={place.run}",
                      f"#SBATCH --output={place.logs}/slurm-%j.out",
                      "#SBATCH --open-mode=append"):
        assert directive in script, directive


def test_the_script_is_valid_bash(robojev, place, tmp_path):
    script = slurm.render_sbatch(robojev, slurm.cluster_config(trainer.TrainConfig()),
                                 config(), place)
    path = tmp_path / "job.sbatch"
    path.write_text(script, encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_the_script_puts_every_cache_under_the_root_and_none_in_home(robojev, place):
    script = slurm.render_sbatch(robojev, slurm.cluster_config(trainer.TrainConfig()),
                                 config(), place)
    for line in ("export UV_CACHE_DIR=", "export UV_PYTHON_INSTALL_DIR=", "export HF_HOME="):
        assigned = next(l for l in script.splitlines() if l.startswith(line))
        assert place.root in assigned, assigned
    # `$HOME` appears in exactly one instruction: the uv binary the cluster already has there.
    home = [l for l in script.splitlines() if "$HOME" in l and not l.startswith("#")]
    assert home == ["UV=$HOME/.local/bin/uv"]


def test_the_environment_and_the_clone_are_cached_behind_a_lock_and_a_stamp(robojev, place):
    script = slurm.render_sbatch(robojev, slurm.cluster_config(trainer.TrainConfig()),
                                 config(), place)
    assert "uv\" sync --frozen --project \"$ENV_DIR\"" in script.replace("$UV", "uv")
    assert f'if [ ! -f "$ENV_DIR/{slurm.ENV_STAMP}" ]; then' in script
    assert "flock 9" in script
    assert f'9>"$ROOT/envs/.{place.env_name}.lock"' in script
    assert 'if [ ! -d "$SRC/.git" ]; then' in script
    assert place.nanojev_commit in script and place.nanojev_url in script


def test_a_requeued_job_starts_over_because_nanojev_cannot_resume(robojev, place):
    """Upstream documents `--init-checkpoint` as "optimizer is new, not an exact training
    resume" and writes no optimizer state, so there is nothing to continue from."""
    script = slurm.render_sbatch(robojev, slurm.cluster_config(trainer.TrainConfig()),
                                 config(), place)
    assert 'RESTART="${SLURM_RESTART_COUNT:-0}"' in script
    assert 'if [ "$RESTART" != "0" ]; then' in script
    assert 'rm -rf "$OUT"' in script


def test_the_warm_start_is_downloaded_on_the_node_and_its_revision_recorded(robojev, place):
    script = slurm.render_sbatch(robojev, slurm.cluster_config(trainer.TrainConfig()),
                                 config(), place)
    assert f"repo_id={trainer.BASE_CHECKPOINT_REPO!r}" in script
    assert 'INIT_CHECKPOINT="$(' in script
    assert '--init-checkpoint "$INIT_CHECKPOINT"' in script
    assert slurm.INIT_PLACEHOLDER not in script
    assert f'"$OUT/{slurm.RUN_RECEIPT}"' in script
    assert '"base_checkpoint_revision": "${BASE_REVISION:-}"' in script


def test_the_warm_start_can_be_pinned_because_those_weights_move(robojev, place):
    """`C-Tianyu/NanoJev` is unversioned and its `best.safetensors` changed during plan 9a, so a
    run being compared against an earlier one has to name the commit that one used."""
    default = slurm.render_sbatch(robojev, slurm.cluster_config(trainer.TrainConfig()),
                                  config(), place)
    assert "revision=None," in default          # the hub's `main`, as a local run gets
    pinned = slurm.render_sbatch(
        robojev, slurm.cluster_config(trainer.TrainConfig()),
        config(base_revision="4a19595eada0857133c0d2be024f879a4077054b"), place)
    assert "revision='4a19595eada0857133c0d2be024f879a4077054b'," in pinned


def test_a_requeued_attempt_continues_from_the_one_it_interrupted(robojev, place):
    """The default. A preempted attempt's weights become the next attempt's warm start, and the
    next attempt asks for the steps the run has left rather than for the whole epoch again."""
    script = slurm.render_sbatch(robojev, trainer.TrainConfig(steps=2906), config(),
                                 place)
    assert 'PRIOR="$RUN/attempt-$RESTART"' in script
    assert 'mv "$OUT" "$PRIOR"' in script
    assert 'INIT_ARGS=(--init-checkpoint "$PRIOR")' in script
    assert "STEPS=$(( 2906 - DONE ))" in script
    # Counted out of the job's own stdout, so the run is one epoch across the attempts and not
    # one epoch per attempt even when an attempt was preempted (see the counter test below).
    assert '"$RUN" "$STDOUT_LOG"' in script
    assert "/slurm-$SLURM_JOB_ID.out" in script
    assert '"/attempt-*/train_log.json"' in script     # kept as a floor
    assert 'r.get("phase") == "full"' in script
    # Nothing of the interrupted attempt is deleted -- it is hours of a shared card.
    assert 'rm -rf "$PRIOR"; mv' in script and 'rm -rf "$OUT"' in script.split("else")[-1]


def test_starting_over_is_still_available_and_says_what_it_costs(robojev, place):
    script = slurm.render_sbatch(robojev, trainer.TrainConfig(steps=2906),
                                 config(resume=False), place)
    assert 'clearing $OUT and training again from the warm start' in script
    assert "attempt-$RESTART" not in script and "INIT_ARGS=(--init-checkpoint \"$PRIOR\")" not in script
    assert "STEPS=2906" in script      # the whole epoch, every attempt


def test_the_init_checkpoint_flag_is_an_array_because_it_is_not_always_there(robojev, place):
    """A bare-backbone run starts with no warm start and *resumes* with one, so the flag cannot be
    baked into the line: it is `"${INIT_ARGS[@]}"`, which expands to nothing when empty rather
    than to one empty argument."""
    for cfg in (trainer.TrainConfig(steps=10),
                trainer.TrainConfig(steps=10, base_model="Qwen/Qwen3-4B")):
        script = slurm.render_sbatch(robojev, cfg, config(), place)
        assert script.count("INIT_ARGS=()") == 1
        assert '"${INIT_ARGS[@]}"' in script
        # and never inline, which would put the two words in the argv unconditionally
        assert "--init-checkpoint '" not in script
    warm = slurm.render_sbatch(robojev, trainer.TrainConfig(steps=10), config(), place)
    assert 'INIT_ARGS=(--init-checkpoint "$INIT_CHECKPOINT")' in warm


def test_the_script_is_deterministic(robojev, place):
    args = (robojev, slurm.cluster_config(trainer.TrainConfig()), config(), place)
    assert slurm.render_sbatch(*args) == slurm.render_sbatch(*args)


# -- the commands --------------------------------------------------------------------------------


def test_ssh_goes_through_a_login_shell_because_uv_is_not_on_the_bare_path():
    argv = slurm.ssh_command(config(), "squeue")
    assert argv[:4] == ["ssh", "-o", "BatchMode=yes", HOST]
    assert argv[4] == "bash -lc squeue" and len(argv) == 5
    named = slurm.ssh_command(config(user="someone"), "squeue")
    assert named[3] == f"someone@{HOST}"


def test_the_remote_command_is_one_quoted_word_because_ssh_re_splits_it():
    """ssh joins its arguments with spaces and lets the far side's shell re-split them, so a
    command passed as separate argv entries loses its quoting: `bash -lc 'a && b'` would arrive
    as `bash -lc a && b` and run `b` in the outer shell. This cost the first real run."""
    argv = slurm.ssh_command(config(), "mkdir -p /a/b && cp '/a/b/c' /a/d")
    assert len(argv) == 5
    # Whatever the far side's shell does to this word, `bash -lc` gets the command back intact.
    assert subprocess.run(["bash", "-c", f"printf '%s\\n' {argv[4]}"],
                          capture_output=True, text=True).stdout.splitlines() == [
        "bash", "-lc", "mkdir -p /a/b && cp '/a/b/c' /a/d"]


def test_rsync_up_sends_the_directorys_contents_and_makes_the_far_side(tmp_path, place):
    argv = slurm.rsync_up_command(config(), tmp_path / "rows", place.data)
    assert argv[0] == "rsync" and "-a" in argv and "--stats" in argv
    assert argv[argv.index("--rsync-path") + 1] == f"mkdir -p {place.data} && rsync"
    assert argv[-2] == f"{tmp_path / 'rows'}/"
    assert argv[-1] == f"{HOST}:{place.data}/"


def test_rsync_down_pulls_because_the_nodes_cannot_reach_this_box(tmp_path, place):
    argv = slurm.rsync_down_command(config(), place.checkpoint, tmp_path / "work")
    assert argv[-2] == f"{HOST}:{place.checkpoint}/"
    assert argv[-1] == f"{tmp_path / 'work'}/"


def _remote(argv: list[str]) -> str:
    """The command the far side's `bash -lc` will see, back out of ssh's single word."""
    shell, flag, command = shlex.split(argv[-1])
    assert (shell, flag) == ("bash", "-lc")
    return command


def test_sbatch_is_parsable_and_submitted_from_the_run_directory(place):
    argv = slurm.sbatch_command(config(), place)
    assert _remote(argv) == f"cd {place.run} && sbatch --parsable {place.script}"


def test_one_poll_asks_squeue_and_the_log_from_a_byte_offset(place):
    argv = slurm.poll_command(config(), place, "271087", 4096)
    remote = _remote(argv)
    assert "squeue -h -j 271087 -o '%T|%R'" in remote
    assert f"tail -c +4097 {place.log('271087')}" in remote     # 1-based, so offset + 1
    assert f"head -c {slurm.LOG_CHUNK}" in remote
    assert slurm.POLL_MARK.decode().strip("\n") in remote


def test_sacct_asks_for_the_job_not_its_steps(place):
    assert "-X -n -P" in _remote(slurm.sacct_command(config(), "271087"))


# -- the parses ----------------------------------------------------------------------------------


def test_squeue_says_pending_running_or_nothing_at_all():
    assert slurm.parse_squeue("PENDING|(Resources)\n") == ("PENDING", "(Resources)")
    assert slurm.parse_squeue("RUNNING|node-6\n") == ("RUNNING", "node-6")
    assert slurm.parse_squeue("") is None
    assert slurm.parse_squeue("\n  \n") is None
    # A requeue can show the job twice for a moment; it is one job, so the first line wins.
    assert slurm.parse_squeue("PENDING|(JobRequeued)\nPREEMPTED|(null)\n")[0] == "PENDING"


def test_sacct_is_a_row_or_not_there_yet():
    row = slurm.parse_sacct(
        "271087|COMPLETED|2026-09-20T10:41:36|2026-09-20T10:42:13|2026-09-20T10:42:19|"
        "00:00:06|0:0|node-6|\n")
    assert row["job_id"] == "271087" and row["state"] == "COMPLETED"
    assert row["elapsed"] == "00:00:06" and row["nodes"] == "node-6"
    assert slurm.parse_sacct("") is None


def test_a_preempted_job_is_not_a_finished_one():
    assert slurm.is_terminal("COMPLETED") and slurm.is_terminal("CANCELLED by 1000")
    assert slurm.is_terminal("TIMEOUT") and slurm.is_terminal("FAILED")
    assert not slurm.is_terminal("PREEMPTED") and not slurm.is_terminal("RUNNING")
    assert not slurm.is_terminal("") and not slurm.is_terminal("REQUEUED")


def test_rsync_stats_are_the_bytes_that_moved():
    stats = slurm.parse_rsync_stats(
        "Number of files: 12\nTotal bytes sent: 60,313,921\nTotal bytes received: 1,204\n")
    assert stats == {"sent": 60313921, "received": 1204}
    assert slurm.parse_rsync_stats("nothing here") == {"sent": 0, "received": 0}


# -- the wait loop -------------------------------------------------------------------------------


def _poll(text: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout=text, stderr="")


def test_the_wait_measures_the_queue_and_the_run_and_notices_a_requeue(monkeypatch, place):
    mark = slurm.POLL_MARK.decode().strip("\n")
    answers = [
        f"PENDING|(Resources)\n{mark}\n",
        f"RUNNING|node-6\n{mark}\n{{\"step\": 12}}\n",
        f"PENDING|(JobRequeued)\n{mark}\n",               # preempted and on its way back
        f"RUNNING|node-1\n{mark}\n{{\"step\": 24}}\n",
        f"{mark}\n",                                      # gone from the queue
        f"{mark}\n",                                      # the final read for trailing output
        "271087|COMPLETED|s|t|u|00:41:02|0:0|node-1|\n",
    ]
    calls = []

    def fake_run(argv, *, check=True):
        calls.append(argv)
        return _poll(answers.pop(0))

    monkeypatch.setattr(slurm, "_run", fake_run)
    outcome = slurm.wait(config(), place, "271087", log=lambda _m: None,
                         sleep=lambda _s: None)
    assert outcome["restarts"] == 1
    assert outcome["state"] == "COMPLETED"
    assert outcome["sacct"]["elapsed"] == "00:41:02"
    assert outcome["log"] == place.log("271087")
    # Three clocks, and the last attempt is the only one that describes the checkpoint: a
    # requeued job starts the epoch again, so `run_seconds` includes work that was thrown away.
    assert outcome["last_run_seconds"] <= outcome["run_seconds"]
    assert outcome["queue_seconds"] >= 0.0
    # Each poll asked from the end of what it had already read.
    offsets = [int(a[-1].split("tail -c +")[1].split()[0]) for a in calls if "tail -c +" in a[-1]]
    assert offsets == sorted(offsets) and offsets[0] == 1 and offsets[-1] > 1


def test_completing_on_the_way_out_is_not_a_requeue(monkeypatch, place):
    """Every clean job passes RUNNING -> COMPLETING -> gone. Counting COMPLETING as a return to
    the queue would report one restart on every run that never restarted -- the real 271334 went
    RUNNING, COMPLETING, PENDING(BeginTime), RUNNING, and only the third of those is a requeue."""
    mark = slurm.POLL_MARK.decode().strip("\n")
    answers = [f"RUNNING|node-6\n{mark}\n", f"COMPLETING|node-6\n{mark}\n",
               f"{mark}\n", f"{mark}\n", "271087|COMPLETED|s|t|u|00:41:02|0:0|n|\n"]
    monkeypatch.setattr(slurm, "_run", lambda argv, check=True: _poll(answers.pop(0)))
    outcome = slurm.wait(config(), place, "271087", log=lambda _m: None,
                         sleep=lambda _s: None)
    assert outcome["restarts"] == 0
    assert outcome["last_run_seconds"] == outcome["run_seconds"]


def test_a_dropped_ssh_is_not_a_finished_job(monkeypatch, place):
    """The marker is printed unconditionally, so output without it is a failed connection --
    ending the wait there would abandon a job that is still training."""
    mark = slurm.POLL_MARK.decode().strip("\n")
    answers = ["", "", f"RUNNING|node-6\n{mark}\n", f"{mark}\n", f"{mark}\n",
               "271087|COMPLETED|s|t|u|00:01:00|0:0|n|\n"]
    monkeypatch.setattr(slurm, "_run", lambda argv, check=True: _poll(answers.pop(0)))
    outcome = slurm.wait(config(), place, "271087", log=lambda _m: None,
                         sleep=lambda _s: None)
    assert outcome["state"] == "COMPLETED" and outcome["restarts"] == 0


def test_a_login_node_that_stays_unreachable_says_the_job_is_still_there(monkeypatch, place):
    monkeypatch.setattr(slurm, "_run", lambda argv, check=True: _poll(""))
    with pytest.raises(slurm.SlurmError) as caught:
        slurm.wait(config(), place, "271087", log=lambda _m: None,
                   sleep=lambda _s: None)
    assert "has NOT been cancelled" in str(caught.value)
    assert place.checkpoint in str(caught.value)


# -- the dry run ---------------------------------------------------------------------------------


def test_the_dry_run_prints_the_script_and_the_commands_and_touches_nothing(
        robojev, tmp_path, monkeypatch, capsys):
    """`--slurm --dry-run` is the ssh-free mode: no connection, no directory, no rsync."""
    def no_processes(*a, **k):  # pragma: no cover -- the assertion is that it is never called
        raise AssertionError("a dry run must not run anything")

    monkeypatch.setattr(subprocess, "run", no_processes)
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        '{"questions_version": "v2", "delta_t": 0.35, "delta_r": 0.058}')
    checkpoints = pathlib.Path(tmp_path / "home" / "checkpoints")

    result = slurm.train_on_slurm(
        "robojev", "libero_spatial",
        trainer.TrainConfig(steps=3350, eval_every=335, data=data),
        config(), dry_run=True, log=lambda _m: None)

    out = capsys.readouterr().out
    assert result["dry_run"] is True
    assert f"#SBATCH --partition={PARTITION}" in out
    assert "$ rsync -a --stats" in out and "$ ssh -o BatchMode=yes" in out
    assert "sbatch --parsable" in out
    assert "--max-microbatch-tokens 32768" in out
    assert not checkpoints.exists(), "a dry run created a checkpoint directory"
    assert not list(tmp_path.glob("**/rows.jsonl"))


def test_the_cluster_is_asked_for_the_path_budget_the_rows_question_set_needs(
        robojev, tmp_path, monkeypatch, capsys):
    """The same resolution a local run makes, made before the job script is rendered: the cluster
    must not train at 1536 what this box would serve at 1024."""
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: pytest.fail("a dry run must not run anything"))
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        '{"questions_version": "v2", "cm_per_unit": 5.0, "delta_t": 1.0, "delta_r": 0.058}')
    slurm.train_on_slurm("robojev", "libero_spatial",
                         trainer.TrainConfig(steps=3350, data=data), config(),
                         dry_run=True, log=lambda _m: None)
    assert "--max-length 1024" in capsys.readouterr().out


def test_a_missing_harvest_manifest_costs_no_queue_time(robojev, tmp_path):
    empty = tmp_path / "rows"
    empty.mkdir()
    with pytest.raises(trainer.TrainError) as caught:
        slurm.train_on_slurm("robojev", "libero_spatial",
                             trainer.TrainConfig(data=empty), config(),
                             dry_run=True, log=lambda _m: None)
    assert caught.value.exit_code == 2


def test_the_cli_flag_reaches_the_module_with_the_operators_overrides(monkeypatch, capsys):
    seen = {}

    def fake(policy, suite, cfg, config, *, dry_run=False, **kw):
        seen.update(policy=policy, suite=suite, cfg=cfg, config=config, dry_run=dry_run)
        return {"dry_run": True}

    monkeypatch.setattr(slurm, "train_on_slurm", fake)
    code = cli.main(["train", "--suite", "libero_spatial", "--slurm", "--dry-run",
                     "--steps", "3350", "--eval-every", "335",
                     "--slurm-host", HOST, "--slurm-root", ROOT,
                     "--slurm-partition", "other-gpu", "--slurm-cpus", "8"])
    assert code == 0
    assert seen["dry_run"] is True
    assert seen["cfg"].steps == 3350 and seen["cfg"].eval_every == 335
    assert seen["config"].partition == "other-gpu" and seen["config"].cpus == 8
    # Untouched flags keep the module's own defaults rather than becoming None.
    assert seen["config"].host == HOST
    assert seen["config"].mem == config().mem
    # The shared-GPU warning is about *this* box's card, and this run does not use it.
    assert cli.TRAIN_WARNING not in capsys.readouterr().err


# ----------------------------------------------- the resume counter (diagnosis note §R4.5)

def resume_counter(script: str) -> str:
    """The python the job runs to work out how many steps it has already taken."""
    body = script.split("<<'ROBOJEV_COUNT'\n", 1)[1]
    return body.split("\nROBOJEV_COUNT", 1)[0]


def count_steps(robojev, place, tmp_path, stdout_lines, train_logs=()) -> int:
    """Run that counter for real, against a synthetic run directory and stdout."""
    import subprocess
    import sys as _sys

    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    for attempt, records in train_logs:
        directory = run / f"attempt-{attempt}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "train_log.json").write_text(json.dumps(records))
    log = tmp_path / "slurm-271734.out"
    log.write_text("\n".join(stdout_lines) + "\n")
    script = slurm.render_sbatch(robojev, trainer.TrainConfig(steps=2906), config(),
                                 place)
    out = subprocess.run([_sys.executable, "-c", resume_counter(script), str(run), str(log)],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def logged(steps, every=12, phase="full"):
    """Upstream's own stdout shape: one JSON record per logged step."""
    return [json.dumps({"step": s, "phase": phase, "loss": 0.5, "questions": 12})
            for s in range(every, steps + 1, every)]


def test_a_preempted_attempt_is_counted_from_stdout_not_from_a_file_it_never_wrote(
        robojev, place, tmp_path):
    """**The run-4 failure, as a test** (diagnosis note §R4.5).

    Job 271734 was preempted at step 2329 of 2906. Upstream writes `train_log.json` once, after
    the final evaluation, so the interrupted attempt left none -- and the old counter, which read
    only that file, answered `0 of 2906` and asked for the whole epoch again. On a partition that
    preempts about hourly, a run that resets its own progress never finishes.
    """
    assert count_steps(robojev, place, tmp_path, logged(2329)) == 2328      # the last record at 12-step spacing
    # ...and the old source alone, which is what the bug read, has nothing to say.
    assert count_steps(robojev, place, tmp_path, [], train_logs=()) == 0


def test_steps_are_summed_across_attempts_because_each_ones_counter_restarts(
        robojev, place, tmp_path):
    """Every attempt's `step` counts from zero within itself, so the total is the sum of the
    high-water marks -- an attempt boundary is a step that did not increase.

    Two preemptions, at 1200 and 900 logged steps, followed by an attempt 792 steps in: the run
    has taken 2892 of its 2906 and has 14 left. Not 2906 (the bug), and not 792 (the last attempt
    alone), and not 1200 (the largest)."""
    lines = logged(1200) + logged(900) + logged(800)
    assert count_steps(robojev, place, tmp_path, lines) == 1200 + 900 + 792 == 2892


def test_a_completed_attempts_train_log_is_a_floor_and_the_two_agree(robojev, place, tmp_path):
    """Where both sources exist they say the same thing, and the larger is taken -- so a source
    going missing can only ever under-resume, never skip work."""
    records = [{"step": s, "phase": "full"} for s in range(1, 501)]
    # stdout logs every twelfth step (496), the completed attempt's own log has all 500.
    assert count_steps(robojev, place, tmp_path, logged(500), train_logs=[(0, records)]) == 500


def test_the_head_warmup_is_not_counted_as_training(robojev, place, tmp_path):
    """A fresh-head run does upstream's head-only warm-up first (`--head-steps`, `phase: "head"`).
    Counting it would make the resume think it had trained twelve steps it has not."""
    lines = logged(60, phase="head") + logged(240)
    assert count_steps(robojev, place, tmp_path, lines) == 240 - 240 % 12


def test_junk_in_the_stdout_is_ignored_rather_than_crashing_the_job(robojev, place, tmp_path):
    """The log also carries `nvidia-smi`, uv's output, tqdm bars and SLURM's own preemption
    notice. A counter that raised on any of them would take the job down on its restart."""
    lines = ["robojev: job 1234 on node-4, restart 0",
             "Loading weights:  9%|9         | 37/398",
             "{not json at all",
             json.dumps({"phase": "full"}),                    # no step
             json.dumps({"step": "1200", "phase": "full"}),    # not an int
             *logged(600),
             "*** JOB 1234 CANCELLED AT 2026-09-20T21:41:16 DUE to preemption ***"]
    assert count_steps(robojev, place, tmp_path, lines) == 600


def test_a_missing_stdout_is_not_a_crash_either(robojev, place, tmp_path):
    """The path is built from `$SLURM_JOB_ID` and a restart runs before the new attempt has
    written anything; if it is not there, the answer is zero, not a traceback."""
    import subprocess
    import sys as _sys

    script = slurm.render_sbatch(robojev, trainer.TrainConfig(steps=2906), config(),
                                 place)
    out = subprocess.run(
        [_sys.executable, "-c", resume_counter(script), str(tmp_path / "nope"),
         str(tmp_path / "also-nope.out")], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "0"


# ------------------------------------------- what the cluster costs per step (diagnosis §R4.4)

def test_the_cluster_drops_the_recompute_for_a_named_backbone_too():
    """A 4B run is exactly the one that must not pay for activation recompute it has 141 GB of
    room to avoid: run 4 measured 3.142 s/step against run 3's 1.11, and the recompute is part of
    that. `--base-model` is not a reason to keep it -- the card is the reason, and the card is the
    same one."""
    for base in (trainer.TrainConfig(), trainer.TrainConfig(base_model="Qwen/Qwen3-4B")):
        assert slurm.cluster_config(base).gradient_checkpointing is False
    # And an operator who asks for it explicitly still gets it, on either backbone: comparing the
    # cluster against the 3090 step for step is a real reason to want it.
    pinned = slurm.cluster_config(trainer.TrainConfig(base_model="Qwen/Qwen3-4B"),
                                  pinned={"gradient_checkpointing"})
    assert pinned.gradient_checkpointing is True


def test_the_cluster_spaces_the_evaluations_over_the_run():
    """The default cadence is 50 steps, which over a 2906-step epoch is **58** dev evaluations.
    Run 4 measured one 4B dev evaluation at 299 s, ~60 s of it writing 16 GB: 58 of those is 4.8
    hours of evaluation bolted to 2.5 hours of training, on a partition that preempts hourly and
    only writes the held-out table after the last one."""
    assert slurm.cluster_config(trainer.TrainConfig(steps=2906)).eval_every == 291
    assert 2906 / 291 <= slurm.CLUSTER_EVALS
    # A short run keeps the default rather than evaluating more often than it.
    assert slurm.cluster_config(trainer.TrainConfig(steps=100)).eval_every == 50
    # And a caller who named a cadence keeps it -- including a tighter one.
    assert slurm.cluster_config(trainer.TrainConfig(steps=2906, eval_every=20)).eval_every == 20


def test_the_job_samples_the_cards_own_memory_while_it_runs(robojev, place):
    """`summary.json`'s `max_gpu_allocated_gb` is torch's high-water mark and it is written in the
    final block, after the evaluations -- so a *preempted* attempt reports none, which is exactly
    the attempt whose memory one wants to know about. This is the card's own number, sampled, and
    it lands in the receipt."""
    script = slurm.render_sbatch(robojev, trainer.TrainConfig(steps=10), config(),
                                 place)
    assert "nvidia-smi --query-gpu=memory.used" in script
    assert f"sleep {slurm.VRAM_SAMPLE_SECONDS}" in script
    assert 'trap \'kill $VRAM_SAMPLER 2>/dev/null || true\' EXIT' in script
    assert '"peak_vram_mib": $PEAK_VRAM_MIB' in script
    # Killed before the receipt is written, so the sampler cannot outlive the trainer.
    assert script.index("kill $VRAM_SAMPLER 2>/dev/null || true\nPEAK_VRAM_MIB=") < \
        script.index('"peak_vram_mib"')
