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

`npm run prepare-assets` copies the Go2 scene and meshes from the project asset
library into `public/robot/`. Policy ONNX files are versioned in
`public/policies/`; clone with Git LFS enabled before running the demo.

## Included policies

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
