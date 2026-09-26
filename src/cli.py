"""Thin CLI adapters that register LainLab tasks before delegating to mjlab."""


def _register_tasks() -> None:
  import src.tasks  # noqa: F401


def train() -> None:
  """Train an LainLab or built-in mjlab task."""
  _register_tasks()
  from mjlab.scripts.train import main

  main()


def play() -> None:
  """Evaluate an LainLab or built-in mjlab task."""
  _register_tasks()
  from mjlab.scripts import play as mjlab_play

  from src.viewer import TerrainLevelViserPlayViewer

  mjlab_play.ViserPlayViewer = TerrainLevelViserPlayViewer
  mjlab_play.main()


def play_baseline() -> None:
  """Play a versioned baseline policy directly in Viser.

  Usage: baseline-play <robot> <terrain> [mjlab play options...]
  """
  import sys

  from src.baselines import resolve_baseline

  args = sys.argv[1:]
  if len(args) < 2 or args[0] in ("-h", "--help"):
    print("usage: baseline-play <robot> <terrain> [play options...]")
    return

  robot, terrain = args[0], args[1]
  play_args = list(args[2:])
  baseline = resolve_baseline(robot, terrain)

  _register_tasks()
  from mjlab.scripts import play as mjlab_play

  from src.viewer import TerrainLevelViserPlayViewer

  mjlab_play.ViserPlayViewer = TerrainLevelViserPlayViewer
  if not any(arg == "--viewer" or arg.startswith("--viewer=") for arg in play_args):
    play_args = ["--viewer", "viser", *play_args]
  sys.argv = [
    sys.argv[0],
    baseline.task_id,
    "--checkpoint-file",
    str(baseline.checkpoint),
    *play_args,
  ]
  mjlab_play.main()


def list_envs() -> None:
  """List all registered LainLab and built-in mjlab tasks."""
  _register_tasks()
  from mjlab.scripts.list_envs import main

  main()


def csv_to_npz() -> None:
  """Convert a G1 motion CSV to a local tracking NPZ."""
  import mjlab
  import tyro

  from src.motion_conversion import main

  tyro.cli(main, config=mjlab.TYRO_FLAGS)


def gmr_to_npz() -> None:
  """Convert a GMR retargeting pickle to a local tracking NPZ."""
  import mjlab
  import tyro

  from src.gmr_conversion import main

  tyro.cli(main, config=mjlab.TYRO_FLAGS)
