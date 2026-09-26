"""CLI contract for ``opendoge-eval``: every condition flag reaches the scorer.

The transition and per-skill scorers take the same measurement-condition
arguments. If ``main`` forgets to forward one, the run silently measures a
different condition than the one recorded next to the number, which is exactly
how the documented transition conditions drifted before.
"""

import sys
from typing import Any

import pytest
from src.toolchain import evaluate as evaluate_module


def _patch_scorers(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
  def fake_transitions(
    task_id: str, checkpoint: object, **kwargs: Any
  ) -> dict[str, object]:
    captured.update(scorer="transitions", task_id=task_id, kwargs=kwargs)
    return {"scorer": "transitions"}

  def fake_evaluate(
    task_id: str, checkpoint: object, **kwargs: Any
  ) -> dict[str, object]:
    captured.update(scorer="evaluate", task_id=task_id, kwargs=kwargs)
    return {"scorer": "evaluate"}

  monkeypatch.setattr(evaluate_module, "transitions", fake_transitions)
  monkeypatch.setattr(evaluate_module, "evaluate", fake_evaluate)


def _run(monkeypatch: pytest.MonkeyPatch, *extra: str) -> dict[str, Any]:
  captured: dict[str, Any] = {}
  _patch_scorers(monkeypatch, captured)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "opendoge-eval",
      "LainLab-OpenDoge-Skills-Flat",
      "--checkpoint",
      "baseline/opendoge/skills/model.pt",
      *extra,
    ],
  )
  evaluate_module.main()
  return captured


def test_transitions_forwards_condition_flags(monkeypatch: pytest.MonkeyPatch) -> None:
  captured = _run(
    monkeypatch, "--transitions", "--seed", "2", "--corruption", "--train-config"
  )

  assert captured["scorer"] == "transitions"
  assert captured["kwargs"]["corruption"] is True
  assert captured["kwargs"]["train_config"] is True
  assert captured["kwargs"]["seed"] == 2


def test_per_skill_eval_forwards_condition_flags(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  captured = _run(monkeypatch, "--seed", "1", "--corruption")

  assert captured["scorer"] == "evaluate"
  assert captured["kwargs"]["corruption"] is True
  assert captured["kwargs"]["train_config"] is False
  assert captured["kwargs"]["seed"] == 1


def test_transitions_default_conditions_are_clean(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  captured = _run(monkeypatch, "--transitions")

  assert captured["kwargs"]["corruption"] is False
  assert captured["kwargs"]["train_config"] is False
  assert captured["kwargs"]["seed"] is None
