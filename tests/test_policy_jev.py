"""The learned policy with the hosted engine: `api="typesafe"`, and no socket anywhere.

`test_policy_model.py` is what a *local* checkpoint gets; this is the other engine, and
the point of most of it is that the two are the same. The last test is the one that matters: given
the same distributions, the hosted engine and the local one produce the same state text and the
same action, because everything between the answer and the arm is shared code.

Every call goes through a stub transport. The key is a fake, and the assertions that mention it
are asserting where it does *not* appear.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from robojev import jev_api
from robojev import policy as policy_mod
from test_policy_model import (                                         # noqa: F401
    FakeV2Predictor, INSTRUCTION, PRIVILEGED, QIDS, STATE, fake_upstream, home, obs, server,
)

#: Not shaped like anybody's real key, on purpose. Every assertion that mentions it is asserting
#: where it does *not* appear.
FAKE_KEY = "test-key-not-real"
MODEL = "jev-latest"
VERSION = "jev-1.13.0"


class StubApi:
    """The endpoint, answering whatever it is asked exactly as `FakeV2Predictor` does.

    Same `_distribution` as the local fake, so a test can put the two engines side by side and
    the only thing that differs is the transport. A `boolean` asked as a `noul` comes back as one
    number, which is the whole of the shape difference between the two engines.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, url, body, headers, timeout):
        payload = json.loads(body.decode())
        self.calls.append({"url": url, "headers": headers, "body": payload})
        answers = {}
        for qid, q in payload["questions"].items():
            if q["type"] == "noul":
                # `_distribution`'s head lands on the first candidate, which for our boolean is
                # `true`; the noul is P(true) by definition.
                answers[qid] = {"type": "noul",
                                "noul": FakeV2Predictor._distribution(qid, ["true", "false"])["true"]}
            else:
                probabilities = FakeV2Predictor._distribution(qid, list(q["criteria"]))
                answers[qid] = {
                    "type": "choice",
                    "choice": max(probabilities, key=probabilities.get),
                    "confidence": max(probabilities.values()),
                    "probabilities": probabilities,
                }
        return jev_api.Response(200, {}, json.dumps({
            "model": VERSION, "answers": answers,
            "usage": {"input_tokens": 2336, "output_tokens": 4},
        }).encode())


@pytest.fixture
def api(monkeypatch):
    """A fake key in the environment and a stub transport under every client built from it."""
    monkeypatch.setenv(jev_api.API_KEY_ENV, FAKE_KEY)
    FakeV2Predictor.pinned = {}
    stub = StubApi()
    monkeypatch.setattr(jev_api, "urllib_transport", stub)
    return stub


def make_api(server, **kw):
    s = server.ModelPolicy("libero_spatial", MODEL, api=jev_api.PROVIDER, **kw)
    s.reset(INSTRUCTION)
    return s


# -------------------------------------------------------------------------------- the launch

def test_without_the_key_the_launch_fails_naming_the_variable_and_the_file(server, monkeypatch,
                                                                          tmp_path):
    monkeypatch.delenv(jev_api.API_KEY_ENV, raising=False)
    monkeypatch.setenv("ROBOJEV_HOME", str(tmp_path))
    with pytest.raises(jev_api.JevApiError) as exc:
        server.ModelPolicy("libero_spatial", MODEL, api=jev_api.PROVIDER)
    assert jev_api.API_KEY_ENV in str(exc.value)
    assert str(tmp_path / "env") in str(exc.value)


def test_an_api_this_server_does_not_speak_is_refused_at_construction(server, api):
    with pytest.raises(policy_mod.PolicyError, match="unknown decision API"):
        server.ModelPolicy("libero_spatial", MODEL, api="some-other-provider")


def test_a_hosted_launch_needs_no_checkpoint_directory_and_serves_v2(server, api, tmp_path,
                                                                     monkeypatch):
    """There is no `$ROBOJEV_HOME/checkpoints` at all here: a hosted model has no manifest, and
    the numbers it is served at are the harness's."""
    monkeypatch.setenv("ROBOJEV_HOME", str(tmp_path))
    s = make_api(server)
    assert s.questions_version == "v2"
    assert tuple(s._qids) == QIDS
    assert s.cm_per_unit == jev_api.checkpoint_meta()["cm_per_unit"]
    assert s.delta_t == 1.0


# ------------------------------------------------------------------------------- what it says

def test_describe_records_the_provider_the_model_and_the_version_that_answered(server, api):
    d = make_api(server).describe()
    assert d["checkpoint"] == {"repo": "typesafe://jev-latest", "revision": VERSION}
    assert d["family"] == "decision"
    assert d["protocol"]["privileged_state"] is True
    assert d["protocol"]["questions_version"] == "v2"
    assert d["versions"]["policy_repo"] == "typesafe://jev-latest"
    assert d["versions"]["api_endpoint"] == jev_api.API_URL
    # No NanoJev commit anywhere: none of upstream's code runs for a hosted answer, and naming
    # it would credit a file that was never imported.
    assert "torch" not in d["versions"] and "policy_repo" in d["versions"]
    # The probe is one call and it is cached: a second describe asks nothing.
    before = len(api.calls)
    assert before == 1
    make_api(server).describe()
    assert len(api.calls) == before + 1


def test_the_questions_are_the_ones_the_local_v2_server_asks(server, api, home, fake_upstream):
    hosted = make_api(server).describe()["questions"]
    local = server.ModelPolicy("libero_spatial", str(home)).describe()["questions"]
    assert hosted == local


def test_the_nondeterminism_notes_say_the_state_leaves_the_box(server, api):
    notes = " ".join(make_api(server).describe()["nondeterminism"])
    assert jev_api.API_URL in notes
    assert "CUDA" not in notes


def test_neither_describe_nor_its_json_carries_the_key(server, api):
    d = make_api(server).describe()
    assert FAKE_KEY not in json.dumps(d, default=str)


# ----------------------------------------------------------------------------- what it serves

def test_the_first_decision_grounds_and_then_moves_which_is_two_calls(server, api):
    s = make_api(server)
    chunk, decisions = s.act(obs())
    assert np.asarray(chunk).shape == (5, 7)
    assert len(api.calls) == 2                       # grounding, then motion
    assert set(decisions) >= set(QIDS) | {"target", "destination", "meta"}
    assert decisions["meta"]["grounding"] is not None

    chunk, decisions = s.act(obs())
    assert len(api.calls) == 3                       # one call per decision afterwards
    assert decisions["meta"]["grounding"] is None


def test_every_question_is_asked_in_one_request_with_the_same_criteria_we_declare(server, api):
    s = make_api(server)
    s.act(obs())
    motion = api.calls[-1]["body"]
    assert set(motion["questions"]) == set(QIDS)
    assert motion["model"] == MODEL
    assert motion["questions"]["grip"]["type"] == "noul"
    assert set(motion["questions"]["grip"]["criteria"]) == {"true", "false"}
    assert motion["questions"]["move_x"]["type"] == "choice"
    assert motion["questions"]["move_x"]["criteria"] == s._questions_block()["move_x"]["criteria"]


def test_the_state_sent_is_the_state_recorded(server, api):
    s = make_api(server)
    _, decisions = s.act(obs())
    assert api.calls[-1]["body"]["state"] == decisions["meta"]["state"]


def test_meta_api_carries_the_cost_and_the_version_and_not_the_key(server, api):
    s = make_api(server)
    _, decisions = s.act(obs())
    meta = decisions["meta"]["api"]
    assert meta["model"] == VERSION
    assert meta["input_tokens"] == 2336
    assert meta["latency_s"] >= 0.0
    # Cumulative over the episode, so the once-per-episode grounding call is counted too.
    assert meta["usage"]["calls"] == 2
    assert meta["usage"]["input_tokens"] == 2 * 2336
    assert FAKE_KEY not in json.dumps(decisions, default=str)


def test_the_grip_latch_is_the_same_guard_under_both_engines(server, api):
    s = make_api(server)
    _, decisions = s.act(obs())
    latch = decisions["meta"]["grip_latch"]
    # The stub answers `true` with p 0.9 on the first decision, which is in `reach`: the guard
    # refuses it, exactly as it does for the local model.
    assert latch["asked"] is True
    assert latch["closed"] is False
    assert latch["refused"] is True
    assert decisions["grip"]["choice"] == "false"


def test_ground_forward_commits_the_models_own_answer_through_the_api_too(server, api):
    s = make_api(server, ground="forward")
    _, decisions = s.act(obs())
    assert decisions["meta"]["grounding"]["mode"] == "forward"
    assert decisions["meta"]["target"] is not None


def test_a_failed_call_fails_the_decision_rather_than_composing_a_half_answer(server, api,
                                                                              monkeypatch):
    s = make_api(server)

    def refuse(url, body, headers, timeout):
        return jev_api.Response(429, {"retry-after": "0"}, b'{"error": "rate limit exceeded"}')

    monkeypatch.setattr(jev_api, "urllib_transport", refuse)
    s._engine.client._transport = refuse
    s._engine.client._sleep = lambda _: None
    with pytest.raises(jev_api.JevApiError) as exc:
        s.act(obs())
    assert "rate limit" in str(exc.value)
    assert FAKE_KEY not in str(exc.value)


# ------------------------------------------------------------- the launcher and the registry

def test_the_repo_word_is_the_provider_and_the_model_and_nothing_else():
    """What a run records where a repository would go. It is the two facts that decide what
    answers, joined, so a run made against one hosted model can never be read as another."""
    assert jev_api.repo_word(jev_api.PROVIDER, MODEL) == "typesafe://jev-latest"
    assert jev_api.repo_word(jev_api.PROVIDER, "jev-1.13.0") == "typesafe://jev-1.13.0"


# ------------------------------------------------------- the comparison the whole thing is for

def test_the_two_engines_compose_the_same_action_from_the_same_answers(server, api, home,
                                                                       fake_upstream):
    """Same questions, same state text, same plan, same guard, same composer: given identical
    distributions, the only thing a run can be measuring is the model."""
    hosted = make_api(server)
    local = server.ModelPolicy("libero_spatial", str(home))
    local.reset(INSTRUCTION)

    for _ in range(3):
        hosted_chunk, hosted_decisions = hosted.act(obs())
        local_chunk, local_decisions = local.act(obs())
        assert hosted_decisions["meta"]["state"] == local_decisions["meta"]["state"]
        assert np.allclose(np.asarray(hosted_chunk), np.asarray(local_chunk))
        for qid in QIDS:
            assert hosted_decisions[qid]["choice"] == local_decisions[qid]["choice"]
            assert hosted_decisions[qid]["probabilities"] == pytest.approx(
                local_decisions[qid]["probabilities"])
        assert hosted_decisions["meta"]["target"] == local_decisions["meta"]["target"]
        assert hosted_decisions["meta"]["subgoal"] == local_decisions["meta"]["subgoal"]
