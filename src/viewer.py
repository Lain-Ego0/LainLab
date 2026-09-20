"""LainLab viewer extensions for mjlab play sessions."""

from typing import Any

from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import ViewerAction


class TerrainLevelViserPlayViewer(ViserPlayViewer):
  """Viser viewer with a terrain-level slider for curriculum playback.

  The slider updates the ``randomize_terrain`` reset event's ``fixed_level``
  parameter, then resets all environments so the change is applied immediately.
  """

  def setup(self) -> None:
    self._disable_zero_command_gui()
    super().setup()
    self._add_manual_control_gui()
    self._add_terrain_level_gui()

  def _disable_zero_command_gui(self) -> None:
    """Hide command sliders for command terms with an all-zero range.

    Viser cannot create a slider when ``min == max == 0``. Get-up tasks keep the
    velocity command identically zero, so skipping their GUI is both safer and
    semantically correct.
    """
    command_manager = self.env.unwrapped.command_manager
    for name in command_manager.active_terms:
      term = command_manager.get_term(name)
      cfg = command_manager.get_term_cfg(name)
      ranges = getattr(cfg, "ranges", None)
      if ranges is None:
        continue
      bounds = [
        getattr(ranges, axis, None) for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
      ]
      if all(bound is not None and tuple(bound) == (0.0, 0.0) for bound in bounds):
        setattr(term, "create_gui", lambda *args, **kwargs: None)  # noqa: B010

  def _add_manual_control_gui(self) -> None:
    env = self.env.unwrapped
    command_manager = env.command_manager
    velocity_terms = [
      command_manager.get_term(name)
      for name in command_manager.active_terms
      if hasattr(command_manager.get_term(name), "vel_command_b")
      and hasattr(command_manager.get_term(name), "_joystick_sliders")
    ]
    if not velocity_terms:
      return

    with self._server.gui.add_folder("Manual control"):
      self._invert_lin_vel_y = self._server.gui.add_checkbox(
        "Y: right positive",
        initial_value=True,
        hint="Only changes the Viser joystick sign; training/body-frame definitions are unchanged.",
      )
      self._invert_ang_vel_z = self._server.gui.add_checkbox(
        "Yaw: clockwise positive",
        initial_value=True,
        hint="Only changes the Viser joystick sign; training/body-frame definitions are unchanged.",
      )

    for term in velocity_terms:
      if getattr(term, "_lainlab_manual_sign_wrapped", False):
        continue
      original_compute = term.compute

      def _compute(
        dt: float | Any,
        env_ids: Any | None = None,
        _term=term,
        _original=original_compute,
      ) -> None:
        _original(dt, env_ids)
        joystick_enabled = getattr(_term, "_joystick_enabled", None)
        get_env_idx = getattr(_term, "_joystick_get_env_idx", None)
        if (
          joystick_enabled is None or not joystick_enabled.value or get_env_idx is None
        ):
          return
        idx = int(get_env_idx())
        if self._invert_lin_vel_y.value:
          _term.vel_command_b[idx, 1] = -_term.vel_command_b[idx, 1]
        if self._invert_ang_vel_z.value:
          _term.vel_command_b[idx, 2] = -_term.vel_command_b[idx, 2]

      setattr(term, "compute", _compute)  # noqa: B010
      setattr(term, "_lainlab_manual_sign_wrapped", True)  # noqa: B010

  def _add_terrain_level_gui(self) -> None:
    env = self.env.unwrapped
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
      return
    generator = terrain.cfg.terrain_generator
    if generator is None or not generator.curriculum:
      return
    try:
      term_cfg = env.event_manager.get_term_cfg("randomize_terrain")
    except ValueError:
      return
    if "fixed_level" not in term_cfg.params:
      return

    max_level = terrain.terrain_origins.shape[0] - 1
    if max_level < 0:
      return
    initial_level = int(term_cfg.params.get("fixed_level", 0))
    initial_level = min(max(initial_level, 0), max_level)

    slider = self._server.gui.add_slider(
      "Terrain level",
      0,
      max_level,
      1,
      initial_level,
      hint="Reset environments to the selected curriculum row.",
    )

    @slider.on_update
    def _(_event) -> None:
      self.request_action(
        "SET_TERRAIN_LEVEL",
        {"kind": "terrain_level", "level": int(slider.value)},
      )

  def _handle_custom_action(self, action: ViewerAction, payload: Any | None) -> bool:
    if (
      action == ViewerAction.CUSTOM
      and isinstance(payload, dict)
      and payload.get("kind") == "terrain_level"
    ):
      env = self.env.unwrapped
      terrain = env.scene.terrain
      if terrain is None:
        return True
      try:
        term_cfg = env.event_manager.get_term_cfg("randomize_terrain")
      except ValueError:
        return True
      if "fixed_level" not in term_cfg.params:
        return True
      num_rows = (
        terrain.terrain_origins.shape[0] if terrain.terrain_origins is not None else 1
      )
      level = min(max(int(payload["level"]), 0), num_rows - 1)
      term_cfg.params["fixed_level"] = level
      self._handle_gui_reset(all_envs=True)
      return True
    return super()._handle_custom_action(action, payload)
