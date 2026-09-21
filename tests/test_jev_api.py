"""The hosted Jev engine: the two translations, the retry rule, and the key that never appears.

No test here opens a socket. The transport is a stub that returns a scripted list of responses,
which is the only way the retry rule's branches (a 429 with a `Retry-After`, a 500 that runs out
of attempts, a 401 that must not be retried at all, a connection that never answered) can be
exercised at all, and it keeps `pytest -q` free and offline.

The key in every test is `FAKE_KEY`, and a good half of this file is one assertion: that string
must not appear in a repr, a message, a returned payload or a record. It is asserted against
directly rather than through a "redacted?" helper so that a test which stops checking the real
thing fails rather than passing vacuously.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from robojev import jev_api
from robojev import grounding as v2grounding
from robojev.questions import ACTIVE_QIDS, questions_block, request as v2_request

#: Shaped like a real key and emphatically not one.
FAKE_KEY = "test-key-not-real"

INSTRUCTION = "pick up the black bowl between the plate and the ramekin and place it on the plate"
PRIVILEGED = {
    "akita_black_bowl_1": {"pos": np.array([-0.06, -0.15, 0.97], np.float32),
                           "quat": np.array([1, 0, 0, 0], np.float32)},
    "akita_black_bowl_2": {"pos": np.array([-0.19, 0.20, 0.97], np.float32),
                           "quat": np.array([1, 0, 0, 0], np.float32)},
    "plate_1": {"pos": np.array([0.05, 0.19, 0.97], np.float32),
                "quat": np.array([1, 0, 0, 0], np.float32)},
    "glazed_rim_porcelain_ramekin_1": {"pos": np.array([-0.20, -0.30, 0.97], np.float32),
                                       "quat": np.array([1, 0, 0, 0], np.float32)},
}


class StubTransport:
    """A scripted list of `(status, headers, body)`, one per call, recording what it was sent.

    An entry that is an exception instance is raised instead of returned, which is how a
    connection failure and a timeout are spelled.
    """

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.timeouts: list[float] = []

    def __call__(self, url, body, headers, timeout):
        self.calls.append({"url": url, "headers": headers, "body": json.loads(body.decode())})
        self.timeouts.append(timeout)
        item = self.responses.pop(0) if self.responses else self.responses
        if isinstance(item, BaseException):
            raise item
        status, headers_out, payload = item
        return jev_api.Response(status, headers_out, json.dumps(payload).encode())


def ok(answers: dict, *, model: str = "jev-1.13.0", input_tokens: int = 2336,
       output_tokens: int = 4):
    return (200, {}, {"model": model, "answers": answers,
                      "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}})


def client(*responses, **kwargs):
    sleeps: list[float] = []
    transport = StubTransport(*responses)
    c = jev_api.JevApiClient(
        FAKE_KEY, transport=transport, sleep=sleeps.append,
        **{"attempts": 3, **kwargs},
    )
    c.stub, c.sleeps = transport, sleeps      # for the assertions; not part of the interface
    return c


# ------------------------------------------------------------------------------ the request map

def test_a_choice_question_travels_as_a_choice_with_the_same_criteria():
    block = questions_block(("move_x",))
    mapped = jev_api.to_api_questions(block)
    assert mapped["move_x"]["type"] == "choice"
    assert mapped["move_x"]["criteria"] == block["move_x"]["criteria"]
    assert mapped["move_x"]["instructions"] == block["move_x"]["instructions"]


def test_our_boolean_becomes_their_noul_and_keeps_its_true_false_criteria():
    block = questions_block(("grip",))
    mapped = jev_api.to_api_questions(block)["grip"]
    assert mapped["type"] == "noul"
    assert mapped["instructions"] == block["grip"]["instructions"]
    # A noul's criteria are optional in their schema and ours exist: the latch's two
    # descriptions are what say the answer commits the fingers, so they must travel.
    assert set(mapped["criteria"]) == {"true", "false"}
    assert mapped["criteria"]["true"] == block["grip"]["criteria"]["true"]


def test_a_question_type_the_api_has_no_primitive_for_is_refused_by_name():
    with pytest.raises(jev_api.JevApiError) as exc:
        jev_api.to_api_questions({"q": {"type": "score", "instructions": "x", "criteria": {}}})
    assert "score" in str(exc.value)


def test_the_request_body_is_one_state_the_model_name_and_the_questions():
    state = v2_request("STATE TEXT", state_id="libero_spatial:0:0")["states"][0]
    body = jev_api.to_api_request(state, "jev-latest")
    assert set(body) == {"state", "model", "questions"}
    assert body["state"] == "STATE TEXT"
    assert body["model"] == "jev-latest"
    assert set(body["questions"]) == set(ACTIVE_QIDS)


def test_the_grounding_request_maps_too_and_carries_the_scene_as_choices():
    """The once-per-episode forward is the *same* engine call, in a different frame."""
    request = v2grounding.grounding_request(INSTRUCTION, PRIVILEGED, state_id="s:ground")
    state = request["states"][0]
    body = jev_api.to_api_request(state, "jev-latest")
    assert set(body["questions"]) == set(state["questions"])
    for qid, q in body["questions"].items():
        assert q["type"] == "choice"
        assert q["criteria"] == state["questions"][qid]["criteria"]
        assert len(q["criteria"]) >= 2


# ------------------------------------------------------------------------------ the answer map

def test_a_choice_answer_comes_back_as_the_whole_distribution_in_the_questions_own_order():
    block = questions_block(("move_x",))
    payload = {"answers": {"move_x": {
        "type": "choice", "choice": "+", "confidence": 0.74,
        # Deliberately not in the question's declared order.
        "probabilities": {"hold": 0.2, "+": 0.74, "-": 0.06},
    }}}
    answers = jev_api.from_api_answers(payload, block)
    assert list(answers["move_x"]["probabilities"]) == list(block["move_x"]["criteria"])
    assert answers["move_x"]["probabilities"]["+"] == pytest.approx(0.74)
    assert answers["move_x"]["choice"] == "+"


def test_a_noul_answer_becomes_the_two_candidates_the_boolean_declares():
    block = questions_block(("grip",))
    answers = jev_api.from_api_answers({"answers": {"grip": {"type": "noul", "noul": 0.09}}},
                                       block)
    assert answers["grip"]["probabilities"] == {"true": pytest.approx(0.09),
                                                "false": pytest.approx(0.91)}
    assert answers["grip"]["choice"] == "false"
    assert set(answers["grip"]["probabilities"]) == set(block["grip"]["criteria"])


def test_a_noul_at_one_closes_the_latch():
    block = questions_block(("grip",))
    answers = jev_api.from_api_answers({"answers": {"grip": {"type": "noul", "noul": 1.0}}},
                                       block)
    assert answers["grip"]["choice"] == "true"
    assert answers["grip"]["probabilities"]["false"] == pytest.approx(0.0)


def test_a_missing_answer_is_refused_rather_than_composed_around():
    block = questions_block(("move_x", "grip"))
    with pytest.raises(jev_api.JevApiError) as exc:
        jev_api.from_api_answers({"answers": {"grip": {"type": "noul", "noul": 0.5}}}, block)
    assert "move_x" in str(exc.value)


def test_a_choice_answered_over_other_candidates_is_refused():
    block = questions_block(("move_x",))
    payload = {"answers": {"move_x": {"type": "choice", "choice": "up",
                                      "probabilities": {"up": 0.6, "down": 0.4}}}}
    with pytest.raises(jev_api.JevApiError) as exc:
        jev_api.from_api_answers(payload, block)
    assert "move_x" in str(exc.value)


def test_a_choice_without_a_distribution_is_refused_rather_than_made_one_hot():
    block = questions_block(("move_x",))
    with pytest.raises(jev_api.JevApiError):
        jev_api.from_api_answers(
            {"answers": {"move_x": {"type": "choice", "choice": "+", "confidence": 0.9}}}, block)


def test_a_noul_outside_zero_to_one_is_refused():
    block = questions_block(("grip",))
    with pytest.raises(jev_api.JevApiError):
        jev_api.from_api_answers({"answers": {"grip": {"type": "noul", "noul": 1.4}}}, block)


def test_temperature_flattens_the_distribution_the_way_the_local_engine_does():
    block = questions_block(("move_x",))
    payload = {"answers": {"move_x": {
        "type": "choice", "choice": "+",
        "probabilities": {"-": 0.1, "hold": 0.2, "+": 0.7},
    }}}
    hot = jev_api.from_api_answers(payload, block, temperature=2.0)["move_x"]["probabilities"]
    cold = jev_api.from_api_answers(payload, block, temperature=0.5)["move_x"]["probabilities"]
    assert sum(hot.values()) == pytest.approx(1.0)
    assert sum(cold.values()) == pytest.approx(1.0)
    assert hot["+"] < 0.7 < cold["+"]
    # T = 1 is the identity, to the bit: an argmax run must be the answer as it arrived.
    same = jev_api.from_api_answers(payload, block, temperature=1.0)["move_x"]["probabilities"]
    assert same == {"-": 0.1, "hold": 0.2, "+": 0.7}


# ---------------------------------------------------------------------------------- the engine

def test_one_decision_is_one_call_and_reports_what_it_cost():
    block = questions_block(ACTIVE_QIDS)
    answers = {qid: ({"type": "noul", "noul": 0.2} if q["type"] == "boolean" else
                     {"type": "choice", "choice": list(q["criteria"])[0], "confidence": 0.5,
                      "probabilities": {c: 1.0 / len(q["criteria"]) for c in q["criteria"]}})
               for qid, q in block.items()}
    engine = jev_api.JevApiEngine(client(ok(answers)))
    out = engine.predict(v2_request("STATE", state_id="libero_spatial:0:3"))

    assert list(out["states"][0]["answers"]) == list(block)
    assert out["states"][0]["id"] == "libero_spatial:0:3"
    assert out["execution"]["forward_passes"] == 1
    assert out["api"]["model"] == "jev-1.13.0"
    assert out["api"]["input_tokens"] == 2336
    assert out["api"]["usage"]["calls"] == 1
    assert out["api"]["latency_s"] >= 0.0
    assert len(engine.client.stub.calls) == 1


def test_the_engine_refuses_a_request_carrying_more_than_one_state():
    engine = jev_api.JevApiEngine(client())
    with pytest.raises(jev_api.JevApiError):
        engine.predict({"states": [{"id": "a", "state": "x", "questions": {}},
                                   {"id": "b", "state": "y", "questions": {}}]})


def test_the_probe_reports_the_version_that_answered_not_the_alias_that_was_asked_for():
    engine = jev_api.JevApiEngine(client(ok({"ok": {"type": "noul", "noul": 0.9}})))
    assert engine.client.model == "jev-latest"
    assert engine.probe() == "jev-1.13.0"


# ----------------------------------------------------------------------------- the retry rule

def test_a_rate_limit_is_retried_and_the_servers_own_delay_is_honoured():
    c = client((429, {"Retry-After": "2"}, {"error": "slow down"}),
               ok({"q": {"type": "noul", "noul": 0.5}}))
    c.post({"state": "x", "model": "jev-latest", "questions": {}})
    assert c.sleeps == [2.0]
    assert c.usage["retries"] == 1
    assert c.usage["calls"] == 1


def test_retry_after_ms_wins_over_retry_after_and_is_capped():
    c = client((429, {"retry-after-ms": "250", "retry-after": "9"},
                {"error": "slow down"}),
               ok({"q": {"type": "noul", "noul": 0.5}}))
    c.post({"state": "x", "model": "jev-latest", "questions": {}})
    assert c.sleeps == [0.25]


def test_a_server_error_backs_off_exponentially_and_then_fails_clearly():
    body = {"error": "overloaded"}
    c = client((529, {}, body), (500, {}, body), (503, {}, body))
    with pytest.raises(jev_api.JevApiError) as exc:
        c.post({"state": "x", "model": "jev-latest", "questions": {}})
    assert exc.value.status == 503
    assert exc.value.attempts == 3
    assert "overloaded" in str(exc.value)
    assert len(c.sleeps) == 2 and c.sleeps[0] < c.sleeps[1]


def test_a_bad_key_is_not_retried_because_a_second_attempt_cannot_fix_it():
    c = client((401, {}, {"error": "invalid api key"}))
    with pytest.raises(jev_api.JevApiError) as exc:
        c.post({"state": "x", "model": "jev-latest", "questions": {}})
    assert exc.value.status == 401
    # And it says one attempt, not three: "failed after 3 attempts" about a key would read as a
    # network problem.
    assert exc.value.attempts == 1
    assert "after 1 attempt" in str(exc.value)
    assert len(c.stub.calls) == 1
    assert c.sleeps == []


def test_a_request_the_api_rejects_is_not_retried_either():
    c = client((422, {}, {"error": "criteria must have at least two options"}))
    with pytest.raises(jev_api.JevApiError):
        c.post({"state": "x", "model": "jev-latest", "questions": {}})
    assert len(c.stub.calls) == 1


def test_a_connection_that_never_answered_is_retried_like_a_5xx():
    c = client(jev_api.TransportError("timed out after 30s"),
               ok({"q": {"type": "noul", "noul": 0.5}}))
    c.post({"state": "x", "model": "jev-latest", "questions": {}})
    assert c.usage["retries"] == 1


def test_every_attempt_timing_out_fails_with_a_message_that_says_so():
    c = client(*[jev_api.TransportError("timed out after 30s")] * 3)
    with pytest.raises(jev_api.JevApiError) as exc:
        c.post({"state": "x", "model": "jev-latest", "questions": {}})
    assert exc.value.status is None
    assert "timed out" in str(exc.value)


def test_the_timeout_is_passed_to_every_attempt():
    c = client(ok({"q": {"type": "noul", "noul": 0.5}}), timeout=12.5)
    c.post({"state": "x", "model": "jev-latest", "questions": {}})
    assert c.stub.timeouts == [12.5]


def test_a_body_that_is_not_json_is_a_clear_failure_not_a_traceback():
    class Garbage:
        def __call__(self, url, body, headers, timeout):
            return jev_api.Response(200, {}, b"<html>502 Bad Gateway</html>")

    c = jev_api.JevApiClient(FAKE_KEY, transport=Garbage(), sleep=lambda _: None, attempts=1)
    with pytest.raises(jev_api.JevApiError) as exc:
        c.post({"state": "x", "model": "jev-latest", "questions": {}})
    assert "not JSON" in str(exc.value)


def test_usage_accumulates_over_the_calls_an_episode_makes():
    c = client(ok({"q": {"type": "noul", "noul": 0.5}}, input_tokens=100, output_tokens=2),
               ok({"q": {"type": "noul", "noul": 0.5}}, input_tokens=200, output_tokens=3))
    c.post({"state": "x", "model": "jev-latest", "questions": {}})
    c.post({"state": "y", "model": "jev-latest", "questions": {}})
    assert c.usage["calls"] == 2
    assert c.usage["input_tokens"] == 300
    assert c.usage["output_tokens"] == 5


# -------------------------------------------------------------------------------------- the key

def test_the_key_is_sent_as_a_bearer_header_and_nowhere_else():
    c = client(ok({"q": {"type": "noul", "noul": 0.5}}))
    c.post({"state": "x", "model": "jev-latest", "questions": {}})
    call = c.stub.calls[0]
    assert call["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in json.dumps(call["body"])
    assert FAKE_KEY not in call["url"]


def test_the_key_is_not_in_the_repr_of_the_client():
    c = client()
    assert FAKE_KEY not in repr(c)
    assert FAKE_KEY not in str(c)
    assert FAKE_KEY not in f"{c}"


def test_the_key_is_not_in_anything_the_engine_returns():
    block = questions_block(("grip",))
    engine = jev_api.JevApiEngine(client(ok({"grip": {"type": "noul", "noul": 0.3}})))
    out = engine.predict(v2_request("STATE", state_id="s", qids=("grip",)))
    assert FAKE_KEY not in json.dumps(out, default=str)


def test_a_server_that_echoes_the_key_back_does_not_get_it_into_the_exception():
    """Nothing should echo a bearer token. An error message is read by strangers, so assume it."""
    c = client((400, {}, {"error": f"unexpected token {FAKE_KEY} in header"}))
    with pytest.raises(jev_api.JevApiError) as exc:
        c.post({"state": "x", "model": "jev-latest", "questions": {}})
    assert FAKE_KEY not in str(exc.value)
    assert jev_api.API_KEY_ENV in str(exc.value)


def test_a_transport_failure_that_quotes_the_key_is_scrubbed_too():
    c = client(jev_api.TransportError(f"proxy rejected Bearer {FAKE_KEY}"), attempts=1)
    with pytest.raises(jev_api.JevApiError) as exc:
        c.post({"state": "x", "model": "jev-latest", "questions": {}})
    assert FAKE_KEY not in str(exc.value)


def test_a_missing_key_names_the_variable_and_the_file_and_no_value(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOJEV_HOME", str(tmp_path))
    with pytest.raises(jev_api.JevApiError) as exc:
        jev_api.api_key({})
    message = str(exc.value)
    assert jev_api.API_KEY_ENV in message
    assert str(tmp_path / "env") in message


def test_a_blank_key_is_a_missing_key():
    with pytest.raises(jev_api.JevApiError):
        jev_api.api_key({jev_api.API_KEY_ENV: "   "})


def test_the_engine_factory_reads_the_key_from_the_environment():
    engine = jev_api.engine("jev-latest", env={jev_api.API_KEY_ENV: FAKE_KEY},
                            transport=StubTransport(), sleep=lambda _: None)
    assert isinstance(engine, jev_api.JevApiEngine)
    assert FAKE_KEY not in repr(engine.client)


# ------------------------------------------------------------- the manifest a hosted model lacks

def test_the_hosted_checkpoint_meta_is_v2_at_the_measured_scale():
    """A hosted model has no `robojev.json`; these are the harness's numbers, not its own."""
    from robojev.compose import MEASURED_CM_PER_UNIT, MEASURED_STEPS

    meta = jev_api.checkpoint_meta()
    assert meta["questions_version"] == "v2"
    assert meta["cm_per_unit"] == MEASURED_CM_PER_UNIT
    assert meta["delta_t"] == MEASURED_STEPS.units("large") == 1.0
    assert meta["steps"]["cm"] == dict(MEASURED_STEPS.cm)
