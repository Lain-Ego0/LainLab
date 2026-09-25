"""Runner used by the multi-skill task.

Nothing about the optimisation changes; the only difference is how a checkpoint
is loaded. The stock training path calls ``runner.load(path)`` with no
``load_cfg``, which demands a complete training checkpoint -- critic, optimiser
and iteration counter. A behaviour-cloning artifact deliberately contains only
the actor (the critic is cheap to refit and the optimiser state is meaningless
across the cloning boundary), so this runner defaults to loading the actor.

That keeps the fine-tune on the framework's normal path:
``uv run train <task> --agent.resume True --agent.load-run <dir> ...``.
"""

from __future__ import annotations

from mjlab.rl import MjlabOnPolicyRunner


class SkillOnPolicyRunner(MjlabOnPolicyRunner):
  """``MjlabOnPolicyRunner`` that can resume from an actor-only checkpoint."""

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    if load_cfg is None:
      load_cfg = {"actor": True}
    return super().load(
      path, load_cfg=load_cfg, strict=strict, map_location=map_location
    )
