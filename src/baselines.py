"""Discovery for versioned, directly playable baseline policies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src import PACKAGE_ROOT

BASELINE_ROOT = PACKAGE_ROOT / "baseline"


@dataclass(frozen=True)
class Baseline:
  """A bundled baseline policy and its manifest metadata."""

  robot: str
  terrain: str
  task_id: str
  method: str
  checkpoint: Path
  onnx: Path | None
  iterations: int | None


def resolve_baseline(robot: str, terrain: str) -> Baseline:
  """Resolve ``baseline/<robot>/<terrain>/baseline.json``.

  Robot and terrain names are case-insensitive and use the directory layout
  ``baseline/<robot>/<terrain>/``.
  """
  robot = robot.strip().lower()
  terrain = terrain.strip().lower()
  if not robot or not terrain:
    raise ValueError("robot and terrain must be non-empty")

  root = BASELINE_ROOT / robot / terrain
  manifest_path = root / "baseline.json"
  if not manifest_path.is_file():
    raise FileNotFoundError(f"未找到 baseline：{robot}/{terrain}")

  data = json.loads(manifest_path.read_text(encoding="utf-8"))
  checkpoint = root / str(data.get("checkpoint", "model.pt"))
  if not checkpoint.is_file():
    raise FileNotFoundError(f"baseline checkpoint 不存在：{checkpoint}")

  onnx_name = data.get("onnx")
  onnx = root / str(onnx_name) if onnx_name else None
  if onnx is not None and not onnx.is_file():
    onnx = None

  return Baseline(
    robot=robot,
    terrain=terrain,
    task_id=str(data["task_id"]),
    method=str(data.get("method", "")),
    checkpoint=checkpoint,
    onnx=onnx,
    iterations=data.get("iterations"),
  )


def available_baselines() -> tuple[tuple[str, str], ...]:
  """Return every ``(robot, terrain)`` pair with a valid manifest."""
  if not BASELINE_ROOT.is_dir():
    return ()
  found: list[tuple[str, str]] = []
  for manifest in sorted(BASELINE_ROOT.glob("*/*/baseline.json")):
    found.append((manifest.parent.parent.name, manifest.parent.name))
  return tuple(found)
