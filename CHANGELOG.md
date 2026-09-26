# Changelog

## Unreleased

### Breaking

- Python 包由 `lloco` 平铺为 `src.*`（`src/lloco/**` → `src/**`），发行包名由 `lloco`
  改为 `lainlab`（包路径仍是 `src.*`，不再占用 PyPI 上已被使用的 `src` 名称）；
  `lloco-train` / `lloco-play` / `lloco-list-envs` / `lloco-webui` /
  `lloco-workbench` 入口移除，改用 `src-train` / `src-play` / `src-list-envs` /
  `src-webui` / `src-workbench`（`train` / `play` / `list-envs` / `webui` 保持不变）。
- 移除 `simulate/` 下的 Unitree MuJoCo / SDK2 仿真桥源码；仓库不再内置实机桥接。
- 模块路径重命名（无功能变化）：`src.skills` → `src.toolchain`（离线工具链，不是任务
  包），`src.tasks.skills` → `src.tasks.multiskill`（统一多技能任务类型）。
  `opendoge-*` 命令名与 `LainLab-OpenDoge-*` / `Unitree-Go2-*` 任务 ID 均不变。
- Unitree 速度任务（A2、As2、Go2、G1、G1-23Dof、H1_2、H2、R1）的 RSL-RL
  `experiment_name` 由 `<robot>_velocity` 改为
  `<task_group>_<robot>_<terrain>_velocity`，Flat 与 Rough 不再共用同一实验目录；
  续训旧实验需显式指定 `--agent.load-run`。

### Changed

- 将仓库从 mjlab 源码副本重构为依赖 mjlab 1.6.0 的下游项目。
- 将 Unitree 资产、任务、CLI 与部署代码分层。
- 用共享 robot profile 取代各机器人重复的速度任务目录。
- `play` 改用带地形层级滑块和手动摇杆的 Viser 查看器（对所有任务生效）；Rough
  回放命令区间改由各 profile 的 `play_command_ranges` 提供，`Unitree-*-Rough` 由
  mjlab 默认的 x(-1.0, 1.0) / y(-1.0, 1.0) / yaw(-0.5, 0.5) 变为
  x(-1.0, 1.5) / y(-0.5, 0.5) / yaw(-0.7, 0.7)（仅影响回放/评测，不影响训练）。
- 版本化 baseline 由仓库根移入包内 `src/baseline/`，随 wheel 分发：`baseline-play`
  以及 `opendoge-collect` 的专家来源不再依赖“当前工作目录是仓库根”这一假设。
- 移除开发期的静态类型检查（pyright）：`make type` 目标、`[tool.pyright]` 配置、
  `pyright` 开发依赖与手写的 `typings/mujoco` 桩一并删除；`make check` 现在只跑
  `lint + test`。此前该检查长期有 200+ 条误报（`mujoco` 未带类型、测试与 vendored
  代码占了绝大多数），既当不了门禁也没人维护。

### Added

- 增加 A2、As2、Go2、G1、G1-23Dof、H1_2、H2、R1 的任务注册。
- 增加资产编译、任务注册和依赖版本兼容性测试。
- 增加 OpenDoge（LainLab 自研四足）资产、Flat/Rough 速度任务，getup / handstand /
  jump 单技能任务，以及单策略多技能任务 `LainLab-OpenDoge-Skills-Flat`（57 维观测）。
- 增加版本化 baseline（`src/baseline/<robot>/<terrain>/`）与 `baseline-play` 入口。
- 增加 OpenDoge 多技能工具链 `opendoge-calibrate` / `opendoge-eval` /
  `opendoge-collect` / `opendoge-bc` / `opendoge-dagger` / `opendoge-export`。

### Fixed

- 修复 OpenDoge rough 地形覆盖通过 mjlab 全局共享的
  `ROUGH_TERRAINS_CFG.sub_terrains` 泄漏到其他 rough 任务的问题：此前按注册顺序，
  Unitree 全部 Rough 速度任务、`Unitree-Go2-DreamWaQ-Rough` 以及 mjlab 内建的
  Go1/G1 Rough 速度任务的 `pyramid_stairs.step_width` 会被静默改为 0.22、
  `random_rough.scale_with_difficulty` 会被改为 True。
- 修复 `opendoge-eval --transitions` 未透传 `--seed` / `--corruption` /
  `--train-config`，导致过渡指标在非声明条件下被测量的 bug。
- 修正与实现不符的文档：`docs/opendoge_multiskill.md` 的观测维度（54 维 → 57 维）
  和 `src/tasks/multiskill/rl.py` 的默认架构描述（扁平共享输出层）。
- 修正 README：目录树补齐 `tasks/{getup,handstand,jump,multiskill}`、`toolchain/`、
  `workbench/`、`baseline/`、`viewer.py`、`docs/`、`playground/` 等实际目录，并移除
  指向已删除 `deploy/`（含 `deploy/README.md` 死链）的错误部署说明。
