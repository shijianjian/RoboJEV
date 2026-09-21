"""The hosted Jev behind the same engine interface the local fine-tune is served through.

`robojev` is a NanoJev checkpoint `robojev train` produced here; `jev` is TypeSafe's own hosted
System One model answering the *same* questions about the *same* state text. One is a 0.6B model
fine-tuned on this robot's decisions, the other a general-purpose model that has never seen a
robot, and the only way that comparison means anything is if literally nothing else differs: the
same question block, the same serialised state, the same plan, the same grip latch, the same
grounding rule, the same composer. So this module is deliberately *small*. It is a transport and
two translations, and everything else is the recipe server's, unchanged.

**The engine interface** is the one `predict_toy_decisions.DecisionPredictor` already defines and
`robojev/policy.py` already calls, because the server is written against a shape rather
than against a class:

    engine.predict(request, batch_questions=..., temperature=...) -> {"states": [{"answers": {...}}]}

`request` is `robojev.questions.request`'s shape -- one state, its id, and the
`{qid: {type, instructions, criteria}}` block -- which is also what `v2.grounding.grounding_request`
builds for the once-per-episode grounding forward. Both reach this engine unchanged, so the two
forwards a v2 episode makes are two calls here and nothing in the server has to know which is
which.

**The two translations.**

*Out*: one state per call (`POST /v1/systemone` takes exactly one `state`), our `choice`
questions as their `choice` with the same `criteria` dict, and our `boolean` question (`grip`,
the latch) as their `noul` -- a yes/no whose answer is a probability. A noul may carry optional
`true`/`false` criteria and ours does, so they travel too: the latch's two candidate descriptions
are the text that says what "true" commits the fingers to, and dropping them would be asking a
different question than the fine-tune was trained on.

*Back*: a `choice` answer already carries a full `probabilities` map and it is returned as-is (in
the question's own candidate order, which the server's `_ordered` then pins); a `noul` answer is
one number, `P(true)`, and becomes `{"true": p, "false": 1 - p}` -- the two candidate ids our
boolean question declares, so the composer, the latch, the console and the record all read it
exactly as they read the local model's.

**What is deliberately not here.** No SDK: this is `urllib` and `json` from the standard library,
so nothing has to be installed for it and the comparison cannot be blamed on a client library. No caching, no
concurrency: one decision is one call, and a run's cost is therefore exactly its decision count.

**The key** is `$JEV_API_KEY`, read from the process environment, or from `$ROBOJEV_HOME/env`
(one `NAME=value` per line, mode 600, outside the checkout) when it is not already there. It is held in one private attribute, never logged, never put in a record, never in
`describe()`, and scrubbed out of every message this module raises (`_scrub`): an exception that
carried it would end up in a job's `error` column and on a public page.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import time
import urllib.error
import urllib.request

from robojev import home
from robojev.compose import MEASURED_CM_PER_UNIT, MEASURED_STEPS

#: The endpoint. One state per call; every question in the request is answered against it in one
#: pass, which is the same property the local engine's `batch_questions=0` has and the reason a
#: decision costs one call rather than eight.
API_URL: str = "https://api.typesafe.ai/v1/systemone"

#: The provider word. It is the registry checkpoint's `api` field and the first half of the
#: reserved repo string a run records (`typesafe://jev-latest`).
PROVIDER: str = "typesafe"

#: The environment variable the key arrives in, named in the failure when it is missing.
API_KEY_ENV: str = "JEV_API_KEY"

#: The model name sent in the request when the registry pins none. An alias, not a version: it
#: moves when TypeSafe ships a release, which is why the registry pins `revision: null` for it
#: and the *response's* `model` is what a run records as its revision.
DEFAULT_MODEL: str = "jev-latest"

#: Seconds one call may take before it is abandoned and retried. Their SDK's own budget is 30 s
#: per call; a decision that takes longer than that has already cost the episode more than a
#: retry will.
DEFAULT_TIMEOUT: float = 30.0

#: Total attempts, first one included: their SDK retries twice by default and so do we.
DEFAULT_ATTEMPTS: int = 3

#: What is worth trying again: their documented 429 (rate limit) and 529 (overloaded), 408, and
#: the rest of 5xx. Everything else -- 401 (the key), 422 (the request shape) -- is a fact about
#: this launch that a second attempt cannot change, so it fails immediately and says so.
RETRY_STATUSES: frozenset = frozenset({408, 429}) | frozenset(range(500, 600))

#: Exponential backoff, their SDK's own numbers: 0.5 s doubling to a 5 s cap, with up to 25 % of
#: each delay subtracted at random so a suite of episodes that all hit a rate limit at once does
#: not retry in lockstep. A `Retry-After` on the response outranks all of it.
BACKOFF_BASE: float = 0.5
BACKOFF_MAX: float = 5.0
BACKOFF_JITTER: float = 0.25

#: How long a `Retry-After` is allowed to park a decision for. A server asking for ten minutes is
#: not something an episode with a step budget can wait out.
RETRY_AFTER_MAX: float = 30.0


class JevApiError(RuntimeError):
    """A call that cannot be completed: the key, the request, or the service.

    `status` is the HTTP status when there was a response and `None` when there was not (a
    connection that failed, a timeout). The message never contains the key -- see `_scrub`.
    """

    def __init__(self, message: str, status: int | None = None, attempts: int = 1):
        super().__init__(message)
        self.status = status
        self.attempts = attempts


def repo_word(provider: str = PROVIDER, model: str = DEFAULT_MODEL) -> str:
    """What a run records where a repository would go: `typesafe://jev-latest`.

    A hosted model has no repository, no directory and no digest, and the two reserved words the
    registry already has do not fit it: it is not `none` (there *are* weights, and a great many)
    and it is not `local` (nobody here made them). So it gets a word of its own, in the shape the
    site already reads as "not a Hub repo, nothing to link to" -- openpi's `gs://` checkpoints go
    down the same branch (`app/src/lib/weights.ts weightUrl`).

    It is the two registry fields joined and nothing else, so the word cannot drift from the
    request: `api` says who answers and `model` says what was asked for. Pinning a version
    (`jev-1.13.0`) in the registry instead of the alias changes this word, which is correct --
    that is a different thing to ask for. What *answered* is the revision beside it, and under
    the moving alias it changes without anything here changing.

    Every consumer must agree on this string, because a run's checkpoint is compared by equality
    against the one it claims to be.
    """
    return f"{provider}://{model}"


def env_file() -> str:
    """`$ROBOJEV_HOME/env` -- the one file a key may be read out of.

    Deliberately **not** `home.home()`: that honours an older variable for the sake of harvests
    and checkpoints somebody already has, and a secret is not something to go looking for under a
    second name. One variable, one path, or the process environment.
    """
    root = os.environ.get(home.HOME_ENV) or home.DEFAULT_HOME
    return str(pathlib.Path(root).expanduser() / "env")


def _key_from_file() -> str:
    """`JEV_API_KEY=<key>` out of `$ROBOJEV_HOME/env`, or `""`. Never raises, never logs."""
    path = pathlib.Path(env_file())
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == API_KEY_ENV:
            return value.strip().strip("'\"")
    return ""


def api_key(env: dict | None = None) -> str:
    """`$JEV_API_KEY`, else `$ROBOJEV_HOME/env`, else a failure that names both and no value.

    Called at construction, so a run without the key fails before a scene is reset rather than on
    the first decision.
    """
    key = (env if env is not None else os.environ).get(API_KEY_ENV, "").strip()
    if not key:
        key = _key_from_file()
    if not key:
        raise JevApiError(
            f"{API_KEY_ENV} is not set, and the hosted Jev policy cannot be served without it. "
            f"Put it in {env_file()} as `{API_KEY_ENV}=<key>` (keep that file mode 600 and "
            f"outside the checkout), or export it before launching by hand."
        )
    return key


def _scrub(text: str, key: str) -> str:
    """`text` with the key replaced, whatever produced it.

    Nothing here is expected to echo the key -- it travels in a header, not in the URL or the
    body -- but an exception's text ends up in a job's `error` column and on a run page, and
    "expected" is not the standard that deserves. Cheap, and total.
    """
    return text.replace(key, f"<{API_KEY_ENV}>") if key else text


# --------------------------------------------------------------------------------- the transport

class Response:
    """One HTTP response, in the three parts the retry rule reads: status, headers, body."""

    __slots__ = ("status", "headers", "body")

    def __init__(self, status: int, headers: dict, body: bytes):
        self.status = int(status)
        self.headers = {str(k).lower(): v for k, v in dict(headers or {}).items()}
        self.body = body or b""


class TransportError(Exception):
    """The request never produced a response: DNS, a refused connection, a timeout."""


def urllib_transport(url: str, body: bytes, headers: dict, timeout: float) -> Response:
    """`POST` with the standard library, and nothing else.

    An HTTP error status is a *response*, not an exception, so `HTTPError` is caught and turned
    back into one: the retry rule below is written over statuses, and a 429 that arrived is not
    the same event as a socket that never opened. Everything that really is a transport failure
    (`URLError`, a timeout, a truncated read) becomes `TransportError`, which is retried the same
    way a 5xx is.
    """
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Response(response.status, dict(response.headers), response.read())
    except urllib.error.HTTPError as exc:                   # a response, with a status
        try:
            payload = exc.read()
        except Exception:                                   # pragma: no cover - a broken body
            payload = b""
        return Response(exc.code, dict(exc.headers or {}), payload)
    except urllib.error.URLError as exc:
        raise TransportError(str(exc.reason)) from None
    except TimeoutError as exc:                             # socket timeout, not an HTTP status
        raise TransportError(f"timed out after {timeout:g}s: {exc}") from None
    except OSError as exc:
        raise TransportError(str(exc)) from None


def _retry_after(response: Response) -> float | None:
    """The delay the server asked for, in seconds, or None. `retry-after-ms` wins when both are
    present because it is the more precise of the two; both are capped (`RETRY_AFTER_MAX`)."""
    raw_ms = response.headers.get("retry-after-ms")
    raw_s = response.headers.get("retry-after")
    for raw, scale in ((raw_ms, 0.001), (raw_s, 1.0)):
        if raw is None:
            continue
        try:
            delay = float(str(raw).strip()) * scale
        except ValueError:                  # an HTTP-date form: fall through to the backoff
            continue
        if delay >= 0.0:
            return min(delay, RETRY_AFTER_MAX)
    return None


class JevApiClient:
    """The transport, the retry rule and the running usage tally. Holds the key; shows nobody.

    `transport` and `sleep` are injected so every branch of the retry rule is testable without a
    socket: the tests drive a stub that returns a scripted list of responses.
    """

    def __init__(self, key: str, *, model: str = DEFAULT_MODEL, url: str = API_URL,
                 timeout: float = DEFAULT_TIMEOUT, attempts: int = DEFAULT_ATTEMPTS,
                 transport=None, sleep=time.sleep, rng=None):
        if not key:
            raise JevApiError(f"{API_KEY_ENV} is empty")
        self._key = key
        self.model = model or DEFAULT_MODEL
        self.url = url
        self.timeout = float(timeout)
        self.attempts = max(1, int(attempts))
        # Resolved here rather than as a default argument so that a test which replaces
        # `jev_api.urllib_transport` replaces it for a client the *server* built, which is the
        # only way to drive the recipe server end to end without a socket.
        self._transport = transport if transport is not None else urllib_transport
        self._sleep = sleep
        self._rng = rng or random.Random(0)
        #: Everything a run wants to say about what it spent, and the only thing this object
        #: accumulates: one number per decision is not worth a second object.
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "retries": 0}
        #: What the last call reported, for `meta.api`.
        self.last: dict = {}

    def __repr__(self) -> str:                       # never the key, not even by accident
        return f"JevApiClient(model={self.model!r}, url={self.url!r})"

    # `__str__` follows `__repr__`; spelled out so a future edit to one cannot leak through the
    # other, and so `f"{client}"` in a log line is safe.
    __str__ = __repr__

    def post(self, payload: dict) -> dict:
        """One request, retried within the rule, and the decoded body.

        Sets `self.last` (model, latency, tokens, attempts) and adds to `self.usage`.
        """
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        started = time.monotonic()
        last_detail = ""
        last_status: int | None = None
        attempt = 0
        for attempt in range(1, self.attempts + 1):
            delay: float | None = None
            try:
                response = self._transport(self.url, body, dict(headers), self.timeout)
            except TransportError as exc:
                last_status, last_detail = None, _scrub(str(exc), self._key)
            else:
                if 200 <= response.status < 300:
                    decoded = self._decode(response, attempt, time.monotonic() - started)
                    return decoded
                last_status = response.status
                last_detail = _scrub(self._error_detail(response), self._key)
                if response.status not in RETRY_STATUSES:
                    break
                delay = _retry_after(response)
            if attempt == self.attempts:
                break
            self.usage["retries"] += 1
            self._sleep(self._backoff(attempt) if delay is None else delay)
        # `attempt`, not `self.attempts`: a 401 or a 422 is not retried at all, and a failure
        # that says "after 3 attempts" about one attempt is a failure that reads as a network
        # problem when it is a key or a request shape.
        where = f"HTTP {last_status}" if last_status is not None else "no response"
        raise JevApiError(
            f"{self.url} failed after {attempt} attempt(s) ({where}): {last_detail}",
            status=last_status, attempts=attempt,
        )

    def _backoff(self, attempt: int) -> float:
        base = min(BACKOFF_BASE * (2 ** (attempt - 1)), BACKOFF_MAX)
        return base * (1.0 - BACKOFF_JITTER * self._rng.random())

    @staticmethod
    def _error_detail(response: Response) -> str:
        """The server's own words, bounded. A body that is not JSON is still worth showing -- a
        proxy's HTML error page is the difference between "their 500" and "ours"."""
        text = response.body.decode("utf-8", "replace").strip()
        try:
            parsed = json.loads(text)
        except ValueError:
            return text[:500] or "(empty body)"
        if isinstance(parsed, dict):
            for key in ("error", "message", "detail"):
                if key in parsed:
                    return json.dumps(parsed[key])[:500]
        return text[:500]

    def _decode(self, response: Response, attempt: int, latency: float) -> dict:
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except ValueError as exc:
            raise JevApiError(
                f"{self.url} answered HTTP {response.status} with a body that is not JSON: "
                f"{_scrub(str(exc), self._key)}", status=response.status, attempts=attempt,
            ) from None
        if not isinstance(payload, dict) or "answers" not in payload:
            raise JevApiError(
                f"{self.url} answered without an `answers` object (keys: "
                f"{sorted(payload) if isinstance(payload, dict) else type(payload).__name__})",
                status=response.status, attempts=attempt,
            )
        usage = payload.get("usage") or {}
        self.last = {
            # The *versioned* id that answered, which is what a run records as its revision: the
            # request names an alias and the alias moves.
            "model": str(payload.get("model") or self.model),
            "latency_s": round(float(latency), 4),
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "attempts": attempt,
        }
        self.usage["calls"] += 1
        self.usage["input_tokens"] += self.last["input_tokens"]
        self.usage["output_tokens"] += self.last["output_tokens"]
        return payload


# -------------------------------------------------------------------------------- the two maps

def to_api_questions(questions: dict) -> dict:
    """Our question block in their vocabulary: `choice` stays `choice`, `boolean` becomes `noul`.

    A boolean's optional `criteria` are carried across because ours are not decoration: the
    latch's `true`/`false` descriptions are what say the answer commits the *fingers* for the
    next five control steps rather than predicting that the grasp will hold.
    """
    out: dict = {}
    for qid, q in questions.items():
        kind = q.get("type")
        if kind == "choice":
            criteria = {str(cid): str(text) for cid, text in q["criteria"].items()}
            if len(criteria) < 2:
                raise JevApiError(f"question {qid!r}: a choice needs at least two criteria")
            out[qid] = {"type": "choice", "instructions": q["instructions"], "criteria": criteria}
        elif kind == "boolean":
            mapped = {"type": "noul", "instructions": q["instructions"]}
            criteria = q.get("criteria") or {}
            if criteria:
                if set(criteria) - {"true", "false"}:
                    raise JevApiError(
                        f"question {qid!r}: a boolean's criteria keys must be 'true'/'false', "
                        f"got {sorted(criteria)}"
                    )
                mapped["criteria"] = {k: str(v) for k, v in criteria.items()}
            out[qid] = mapped
        else:
            raise JevApiError(
                f"question {qid!r}: {kind!r} is not a question type this engine can ask. "
                f"robojev asks `choice` and `boolean`; the API takes `choice`, `noul` "
                f"and `score`."
            )
    return out


def to_api_request(state: dict, model: str) -> dict:
    """One `{"id", "state", "questions"}` entry as their request body.

    The state id does not travel: their request has no field for it, and it is ours anyway (it
    names the suite, the task and the decision). It comes back on the answer from the entry we
    still hold.
    """
    return {
        "state": state["state"],
        "model": model,
        "questions": to_api_questions(state["questions"]),
    }


def _temper(probabilities: dict, temperature: float) -> dict:
    """A distribution at temperature T: `p^(1/T)`, renormalised.

    The local engine takes `temperature` into `DecisionPredictor.predict`, which divides the
    logits by it before the softmax; a hosted model returns the distribution already normalised,
    so the same operation is applied here instead. It is the same function of the same answer --
    `softmax(logits/T)` and `p^(1/T)/sum` are identical -- so `--selection sample@T` means the
    same thing under both engines, which is the only reason this exists.
    """
    if temperature is None or abs(float(temperature) - 1.0) < 1e-9:
        return probabilities
    t = float(temperature)
    if t <= 0.0:
        raise JevApiError(f"temperature must be positive, got {t!r}")
    scaled = {k: float(v) ** (1.0 / t) for k, v in probabilities.items()}
    total = sum(scaled.values())
    if not total > 0.0:
        return probabilities
    return {k: v / total for k, v in scaled.items()}


def from_api_answers(payload: dict, questions: dict, temperature: float | None = None) -> dict:
    """Their answers in our shape: `{qid: {"type", "choice", "probabilities", "confidence"}}`.

    Every asked question must come back, and a `choice` must come back with its distribution:
    the server composes an action out of these and the console renders them, so a question
    silently missing its probabilities would be a decision made on a shape nobody checked. A
    `noul` is one number and becomes the two candidates our boolean declares, in the order it
    declares them.
    """
    answers = payload.get("answers") or {}
    out: dict = {}
    for qid, q in questions.items():
        answer = answers.get(qid)
        if answer is None:
            raise JevApiError(
                f"the API answered {sorted(answers)} but was asked {sorted(questions)}: "
                f"{qid!r} is missing"
            )
        if q["type"] == "boolean":
            if "noul" not in answer:
                raise JevApiError(
                    f"question {qid!r} was asked as a noul and came back as "
                    f"{answer.get('type')!r} with no `noul` value"
                )
            p = float(answer["noul"])
            if not 0.0 <= p <= 1.0:
                raise JevApiError(f"question {qid!r}: a noul must be in [0, 1], got {p!r}")
            # The two candidate ids the question itself declares, in its own order -- `criteria`
            # is where they are written down, so a boolean whose ids ever changed changes here
            # and nowhere else.
            ids = list(q.get("criteria") or {"true": "", "false": ""})
            probabilities = {cid: (p if cid == "true" else 1.0 - p) for cid in ids}
            probabilities = _temper(probabilities, temperature)
            chosen = max(probabilities, key=probabilities.get)
            out[qid] = {
                "type": "boolean", "choice": chosen, "probabilities": probabilities,
                # A noul has no confidence field of its own (it *is* the probability); the
                # distance from the coin flip is the same statement in the shape the console
                # renders for every other question.
                "confidence": abs(p - 0.5) * 2.0,
            }
            continue
        raw = answer.get("probabilities") or {}
        if not raw:
            raise JevApiError(
                f"question {qid!r}: the API answered {answer.get('choice')!r} without a "
                f"`probabilities` map. The composer and the console read the whole distribution, "
                f"not the argmax, so there is nothing to fall back to."
            )
        want = list(q["criteria"])
        got = set(raw)
        if got != set(want):
            raise JevApiError(
                f"question {qid!r}: the API answered over {sorted(got)} but the question "
                f"declares {want}"
            )
        probabilities = _temper({cid: float(raw[cid]) for cid in want}, temperature)
        out[qid] = {
            "type": "choice",
            "choice": str(answer.get("choice") or max(probabilities, key=probabilities.get)),
            "probabilities": probabilities,
            "confidence": answer.get("confidence"),
        }
    return out


# -------------------------------------------------------------------------------- the engine

class JevApiEngine:
    """`DecisionPredictor`'s interface over the hosted model.

    One method matters -- `predict(request, batch_questions=..., temperature=...)` -- and it is
    called twice per episode's first decision (grounding, then motion) and once per decision
    afterwards, by a server that does not know which engine it is holding.
    """

    def __init__(self, client: JevApiClient):
        self.client = client

    @property
    def model(self) -> str:
        return self.client.model

    def predict(self, request: dict, batch_questions: int = 0, temperature: float | None = None):
        """One state, one call, every question answered against it.

        `batch_questions` is the local engine's padding knob (every candidate path of every
        question in one tensor) and has no counterpart here: the API evaluates the state once and
        every question against it in parallel, which is the same property, so the argument is
        accepted and ignored. `execution` reports what that means in the local engine's own
        terms -- one forward pass, no candidate paths -- so a record written under either engine
        answers `meta.forward_passes` with a number that means the same thing.
        """
        states = request.get("states") or []
        if len(states) != 1:
            raise JevApiError(
                f"the API takes exactly one state per call and this request carries "
                f"{len(states)}"
            )
        state = states[0]
        questions = state["questions"]
        payload = self.client.post(to_api_request(state, self.client.model))
        answers = from_api_answers(payload, questions, temperature)
        return {
            "states": [{"id": state.get("id"), "answers": answers}],
            "execution": {
                "forward_passes": 1,
                "candidate_paths": None,
                "questions": len(questions),
            },
            # What this call cost and what answered it. The server copies it into `meta.api`; the
            # key is not in it, because it is not in anything this module returns.
            "api": {**self.client.last, "usage": dict(self.client.usage)},
        }

    def probe(self) -> str:
        """The versioned model id that answers for this alias, from the cheapest call there is.

        A run records the *version* that answered (`jev-1.13.0`), not the alias it asked for
        (`jev-latest`), because the alias moves when TypeSafe ships a release and two runs a
        month apart are then two different models under one name. The version is only ever
        reported on a response, so `describe()` asks for one: a five-word state and a single
        yes/no question, tens of input tokens, and it doubles as the proof that the key works
        before an episode is claimed rather than on its first decision.
        """
        payload = self.client.post({
            "state": "ready",
            "model": self.client.model,
            "questions": {"ok": {"type": "noul", "instructions": "Is this state readable?"}},
        })
        return str(payload.get("model") or self.client.model)


def engine(model: str = DEFAULT_MODEL, *, env: dict | None = None, **kwargs) -> JevApiEngine:
    """The engine a policy server holds, with the key read from the environment at construction.

    Raises `JevApiError` naming `$JEV_API_KEY` and `$ROBOJEV_HOME/env` when there is no key, which
    is what makes a keyless launch fail at launch.
    """
    return JevApiEngine(JevApiClient(api_key(env), model=model, **kwargs))


# ------------------------------------------------------------- what a hosted model has no file for

def checkpoint_meta() -> dict:
    """The `robojev.json` this policy has no checkpoint directory to carry.

    A local checkpoint states the vocabulary it was trained under and the scale its answers were
    labelled at, and the server refuses to serve it under any other (`robojev.policy.ModelPolicy`).
    A hosted model was not trained here at all: it has no manifest, and the numbers it must be
    served at are not *its* numbers but **the harness's** -- the question set the site asks, the
    step sizes the expert measured (`robojev.compose.MEASURED_STEPS`), the tracker the state text is
    rendered from. They are stated here, in the same shape a trained checkpoint states them, so
    the server's guards run unchanged and there is exactly one code path that decides what a v2
    decision means.

    `delta_r` is the wrist's `large` step in normalised units. `yaw` is not asked on this suite
    (it is `hold` in 100 % of the expert's decisions), so it multiplies nothing; it is reported
    because the record carries it.
    """
    return {
        "questions_version": "v2",
        "delta_t": MEASURED_STEPS.units("large"),
        "delta_r": MEASURED_STEPS.yaw_units("large"),
        "cm_per_unit": MEASURED_CM_PER_UNIT,
        "steps": {"cm": dict(MEASURED_STEPS.cm), "cm_per_unit": MEASURED_CM_PER_UNIT},
    }


__all__ = [
    "API_KEY_ENV", "API_URL", "DEFAULT_MODEL", "PROVIDER", "JevApiClient", "JevApiEngine",
    "JevApiError", "Response", "TransportError", "api_key", "checkpoint_meta", "engine",
    "env_file", "from_api_answers", "repo_word", "to_api_questions", "to_api_request",
    "urllib_transport",
]
