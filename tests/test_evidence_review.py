import json
from pathlib import Path

import pytest

from assembly_monitor.evidence_review import EvidenceReviewAdapter, combine, validate_observation
from assembly_monitor.vlm import Prompt, RawOutput, make_adapter, load_model_configs, load_experiment


def test_registered_config():
    root = Path(__file__).resolve().parents[1]
    models = load_model_configs([root / "configs/model_evidence_review.yaml"])
    exp = load_experiment(root / "configs/experiment_evidence_review.yaml")
    assert exp["include"] == ["dev"]
    assert all(key in models for key in exp["models"])


IMAGES = [{"i": 1, "camera": "hama1", "time_sec": 0, "input": "a", "source": "a"},
          {"i": 2, "camera": "hama1", "time_sec": 1, "input": "b", "source": "b"}]


def obs(result="success", ids=None):
    return {"result": result, "evidence_images": [2] if ids is None else ids, "observation": "Part visibly seated"}


@pytest.mark.parametrize("value", ["bad json", "[]", json.dumps(obs(ids=[99])), json.dumps(obs(ids=[True])),
                                  json.dumps(obs(ids=[1])), json.dumps(obs(ids=[]))])
def test_invalid_or_early_evidence_abstains(value):
    assert validate_observation(value, IMAGES)["result"] == "uncertain"


def test_agreement_and_conflict():
    assert combine(obs(), obs(), IMAGES)["result"] == "success"
    assert combine(obs(), obs("failure"), IMAGES)["result"] == "uncertain"
    assert combine(obs("failure"), obs("failure"), IMAGES)["failure_reason"] == "unknown"


def test_independent_calls_and_audit(monkeypatch):
    adapter = make_adapter("test", {"type": "transformers", "skill": "evidence_review"})
    assert isinstance(adapter, EvidenceReviewAdapter)
    messages = []
    def generate(msg, budget):
        messages.append(msg)
        return RawOutput(json.dumps(obs()), 0.1, input_tokens=10, output_tokens=5)
    monkeypatch.setattr(adapter, "_generate", generate)
    prompt = Prompt("LABEL MUST NOT ENTER", [], "v", "h", (0, 1), IMAGES, {"part": "BNC"})
    answer = adapter.predict(prompt, max_time=10)
    assert json.loads(answer.text)["result"] == "success"
    assert answer.input_tokens == 20 and answer.extra["skill"]["model_calls"] == 2
    assert "Part visibly seated" not in json.dumps(messages[1])
    assert "LABEL MUST NOT ENTER" not in json.dumps(messages)


def test_timeout_never_calls_again(monkeypatch):
    adapter = EvidenceReviewAdapter("test", {})
    calls = []
    def generate(*args):
        calls.append(1)
        return RawOutput(json.dumps(obs()), 0.1, stopped_by_time=True)
    monkeypatch.setattr(adapter, "_generate", generate)
    answer = adapter.predict(Prompt("", [], "v", "h", (0, 1), IMAGES), max_time=1)
    assert len(calls) == 1 and answer.stopped_by_time
    assert json.loads(answer.text)["result"] == "uncertain"
