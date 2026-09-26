"""Contract tests for versioned baseline policy discovery."""

import hashlib
import json

import pytest
from src.baselines import BASELINE_ROOT, resolve_baseline


def test_opendoge_baselines_resolve() -> None:
  cases = (
    ("flat", "Flat"),
    ("rough", "Rough"),
    ("getup", "Getup"),
    # The single-policy multi-skill baseline: one network, four skills.
    ("skills", "Skills-Flat"),
  )
  for terrain, suffix in cases:
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


def test_skills_baseline_records_its_sota_reference() -> None:
  """The shipped single-policy baseline is the SOTA reference.

  It is the thing every future attempt is measured against, so the manifest has
  to pin the artifact by hash and carry both the reference protocol and the
  adoption rule. Without the hash check a re-exported model could silently
  invalidate every number recorded next to it.
  """
  manifest_path = BASELINE_ROOT / "opendoge" / "skills" / "baseline.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

  assert manifest["status"] == "sota"
  assert manifest["reference_protocol"]["per_skill"]
  assert manifest["reference_protocol"]["transitions"]
  assert manifest["reference_metrics"]["transitions"]["mean_off_diagonal"] > 0
  assert len(manifest["sota_adoption_rule"]) >= 2

  for name, expected in manifest["sha256"].items():
    digest = hashlib.sha256((manifest_path.parent / name).read_bytes()).hexdigest()
    assert digest == expected, f"{name} changed without updating its recorded hash"
