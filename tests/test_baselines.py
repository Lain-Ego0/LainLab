"""Contract tests for versioned baseline policy discovery."""

import pytest
from src.baselines import resolve_baseline


def test_opendoge_baselines_resolve() -> None:
  for terrain, suffix in (("flat", "Flat"), ("rough", "Rough"), ("getup", "Getup")):
    baseline = resolve_baseline("opendoge", terrain)
    assert baseline.robot == "opendoge"
    assert baseline.terrain == terrain
    assert baseline.task_id == f"LainLab-OpenDoge-{suffix}"
    assert baseline.checkpoint.is_file()
    assert baseline.onnx is not None
    assert baseline.onnx.is_file()


def test_unknown_baseline_raises() -> None:
  with pytest.raises(FileNotFoundError):
    resolve_baseline("opendoge", "stairs")
