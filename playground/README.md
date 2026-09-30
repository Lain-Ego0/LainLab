# LainLab Playground

Standalone browser demos for selected, verified LainLab policies.  This project
does not import the workbench, run Python, or communicate with the training
workflow at runtime.

## Run

```bash
cd playground
npm install
npm run dev
```

Use the robot selector in the top bar to switch between **Unitree Go2** and
**Unitree G1 (29 DOF)**. The policy selector lists only policies for that robot.
Switching pauses and resets the simulation, including observation history,
command inputs, and the camera. Click **开始** to run.

`npm run prepare-assets` prepares both robot scenes and meshes in
`public/robot/`, using the project asset library and the G1 overrides in
`assets/g1/`. Policy ONNX files are versioned in
`public/policies/`; clone with Git LFS enabled before running the demo.
Asset preparation skips unchanged files and preserves existing directories.
The Vite dev/preview servers read robot and policy files directly from disk;
missing assets return HTTP 404 instead of the app's HTML entry point.

## Included policies

### Go2

- Front handstand (48-D observation)
- Rear stand (45-D observation)
- Trot (470-D / 10-frame history)
- Jump (470-D / 10-frame history)
- Spring jump (470-D / 10-frame history)
- Arena flat walk (270-D / 6-frame history, sourced from ArenaX)
- DreamWaQ terrain gait (270-D / 6-frame history)
- AMP-CTS gait (270-D / 6-frame history)

The handstand artifact is LainLab's own exported policy (from a
`logs/rsl_rl/go2_handstand/` run); the others were converted once from the
corresponding TorchScript policies in the external `My_unitree_go2_gym-main`
reference project. The browser only ever loads the checked-in `.onnx` files:
there is no export step in the build, and refreshing them is an offline
operation on a machine that has those sources, PyTorch and ONNX installed.

### G1

G1 includes one **AMP flat-walking** policy (`g1Amp`, `policies/g1-amp.onnx`).
It uses `wbc_fsm-main/model/loco/Unitree-G1-AMP-Flat_model_30000.onnx`,
originally called MJ AMP. The checkpoint is copied unchanged; its 384-value
input uses four 96-value history frames in motor order. It produces 29 joint
actions at 50 Hz, while MuJoCo runs at 500 Hz. Default poses, PD gains,
torque-normalized action scales and command ranges follow `State_MJAmp`.
Motor torques are limited by the robot MJCF.

W/S, A/D, Q/E and the virtual joysticks control the policy. Start with low
speeds using the CMD panel.

The G1 scene is the source project's `g1_29dof_mjamp.xml`. Shared meshes reuse
LLoco's G1 assets byte-for-byte; five differing mesh files are bundled under
`assets/g1/meshes/`. No runtime Python process or external source checkout is needed.

To refresh the imported assets from a local source checkout:

```bash
npm run import-g1 -- /path/to/wbc_fsm-main
npm run build
```

This offline import copies the AMP ONNX file unchanged; it does not train
or re-export the policy. Run `npm test` for history and action-scaling checks.

Browser checks cover G1 AMP, robot switching during playback, reset, terrain
recompilation, failed-load recovery, mobile controls, and development asset refresh:

```bash
npx playwright install chromium
npm run test:browser
```

Alternatively, set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to a locally installed
Chrome/Chromium executable.

## Terrain editor

The **Terrain** button opens a browser-native editor implemented as part of
the playground UI; it has no PyQt, Python server, or workbench dependency.
It provides flat, slope, stairs, and seeded obstacle profiles, plus placed
platforms, stairs, ramps, stepping stones, and low walls. Selecting **Apply
to scene** recompiles only the active MuJoCo terrain geometry set; the robot
mesh assets remain cached in the browser VFS so collision topology is current
without reloading those assets.

**Export JSON** creates a portable playground scene draft for future online
publishing. **Import** accepts that JSON and ArenaX-style `scene.json` files;
it also imports simple MuJoCo XML files containing box geoms. Heightfields and
external mesh assets intentionally remain out of scope for browser import.

The current release is a public-facing interactive demo, not a replacement for
native MuJoCo/Viser acceptance checks.
