# LainLab V0.1

**LainLab** 是一个面向多厂商机器人的强化学习、仿真验证与部署项目，基于 **mjlab 1.6.0** 构建。`mjlab` 作为固定版本的通用基础依赖，LainLab 在其上维护机器人资产、任务差异、训练验证契约和部署适配，并由上游持续提供仿真框架能力。

当前仓库同时包含 Unitree 资产与 LainLab 自研 OpenDoge 等机器人资产。项目边界覆盖 LainLab 自己设计的机器人以及其他厂商的机器人资产；新增资产遵循统一的“资产 → profile/任务 → 验证 → 部署适配”路径，保持框架层的厂商中立性。

> 说明：仓库与项目名称为 **LainLab**；源码根目录为 `src/`，Python 包路径统一为 `src.*`。

### 名称与愿景

**LainLab** 由 **Lain** 与 **Lab（实验室）** 组成。Lain 的命名灵感来自《**玲音 / Serial Experiments Lain**》对连接、身份与现实/网络边界的思考；Lab 代表一个开放的机器人实验平台：把不同形态的机器人、动作数据、训练策略和部署验证放进同一套可重复的实验流程。项目希望连接不同机器人形态与同一套训练/验证节奏：保留每台机器人的物理与控制个性，同时让资产接入、技能训练和部署验证有统一、清晰的接口。项目与该作品及其权利方没有官方关联。

### 项目定位与差异

| 维度 | 原版 `mjlab` | `unitree_rl_mjlab` 参考项目 | LainLab |
| --- | --- | --- | --- |
| 角色 | 通用 MuJoCo/Warp 仿真与强化学习框架 | 面向 Unitree 机器人的训练与实机部署项目 | 建立在 `mjlab` 上的多厂商机器人资产、任务与部署层 |
| 机器人范围 | 通用框架能力 | 以 Unitree 机型为中心 | 当前包含 Unitree、LainLab OpenDoge 等起点；设计上预留更多自研与其他厂商资产接入 |
| 框架维护 | 上游维护仿真、manager、MDP、runner 与 viewer | 项目同时承载机型任务和部署流程 | 固定依赖 `mjlab==1.6.0`，以薄适配层降低框架复制与升级成本 |
| 任务组织 | 提供通用任务与扩展机制 | 速度跟踪、动作模仿和实机控制为主 | 用机器人 profile、独立资产和可注册任务表达差异；Go2 还提供来源可追溯的 skill 迁移 |
| 可验证性 | 提供框架级能力 | 以训练/部署工作流为主 | 为资产、配置、观测/奖励契约和 CPU smoke 保留项目级测试，并公开记录已知限制 |
| 动作数据 | 提供通用工具与接口 | 面向 Unitree 工作流准备动作数据 | 在本地将 CSV 转为 NPZ，并按目标机器人校验关节列与自由度，不依赖 W&B |

LainLab 将“接入一台机器人”组织为可复用的产品路径：资产有归属、任务差异有边界、训练接口一致、验证结果可复现、部署代码与模型产物分离。新厂商或自研机器人经过资产与控制参数审查、smoke 测试和任务验证后，进入可用任务集合。

## 目录结构

```text
LainLab/
├── src/
│   ├── assets/          # 各机器人 MJCF、网格和示例动作
│   ├── tasks/
│   │   ├── velocity/    # 通用速度跟踪 MDP
│   │   ├── tracking/    # 通用动作跟踪 MDP
│   │   ├── amp/         # 预留通用 AMP core
│   │   └── robots/      # 机器人组成、profile 与任务注册
│   └── cli.py           # train / play / list-envs 入口
├── tests/               # LainLab 自身的兼容性测试
└── deploy/              # 实机部署源码（不提交模型和预编译运行库）
```

分层原则：

- `mjlab` 负责仿真、manager、通用 MDP、runner 和 viewer。
- `src.assets` 负责本项目机器人描述。
- `src.tasks` 只表达机器人名称、接触点、动作缩放等差异。
- CLI 先注册 LainLab 任务，再复用 mjlab 1.6 的训练和回放实现。

## 安装

需要 Python 3.10–3.13；训练需要 NVIDIA GPU。

```bash
uv sync --extra cu128
```

仅做 CPU 配置检查时：

```bash
uv sync --extra cpu
```

## 使用

```bash
# 查看任务
uv run list-envs

# 训练 OpenDoge（自研机器人归入 LainLab 分类）
uv run train LainLab-OpenDoge-Flat --env.scene.num-envs 4096

# 直接播放内置 baseline
uv run baseline-play opendoge flat --device cuda:0 --num-envs 1
uv run baseline-play opendoge rough --device cuda:0 --num-envs 1 --no-terminations True
uv run baseline-play opendoge getup --device cuda:0 --num-envs 1

# 用随机动作做配置冒烟测试
uv run play LainLab-OpenDoge-Flat --agent random --num-envs 1

# 单策略多技能：walk / handstand / getup / jump
uv run train LainLab-OpenDoge-Handstand --env.scene.num-envs 4096
uv run train LainLab-OpenDoge-Jump --env.scene.num-envs 4096
uv run train LainLab-OpenDoge-Skills-Flat --env.scene.num-envs 4096
uv run opendoge-calibrate --write          # 倒立/跳跃目标标定
uv run opendoge-eval LainLab-OpenDoge-Skills-Flat --checkpoint <ckpt>
uv run opendoge-eval LainLab-OpenDoge-Skills-Flat --checkpoint <ckpt> --transitions

# 回放本地策略
uv run play LainLab-OpenDoge-Flat --checkpoint-file logs/.../model_1000.pt

# 把 G1 CSV 动作转换为 mjlab 跟踪格式
uv run csv-to-npz --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1-subject2 --robot g1
```

### 独立工作台

```bash
# 在 LainLab 根目录启动（webui / src-webui 兼容入口仍可用）
uv run --extra workbench src-workbench
```

浏览器打开 `http://127.0.0.1:7860`。工作台按固定顺序组织：

1. **项目内 URDF/MJCF 查看器**：加载项目模型，或导入包含网格的目录；调节关节，显示坐标系、关节轴、碰撞体和惯量。
2. **策略选择**：Velocity / Tracking 互斥展开；Tracking 内置 GMR 的 LAFAN1 BVH → G1 29-DoF 重定向，从统一动作资产库选择输入，也支持已有 GMR PKL → NPZ；实时查看重定向进度、人体骨架与 G1 姿态，完成后通过时间轴回放。
3. **算法选择**：Velocity / Tracking 分组；Go2 Skill 单独提供 DreamWaQ、AMP-DreamWaQ、CTS、TS Teacher 和动作模板。
4. **训练**：设置迭代轮次、保存间隔、并行环境数；查看实时日志、停止任务，在内嵌 TensorBoard 中查看奖励和训练指标。
5. **导出与验证**：浏览、选择、下载 checkpoint/ONNX，手动导出 ONNX，启动 CPU 单环境 Viser checkpoint 回放。

工作台源码在 `src/workbench/`，不改变 `tasks/` 中的训练算法与环境。
查看器由 LainLab 自行实现：MuJoCo 解析模型，精简 Three.js 页面负责绘制；仅参考 `robot_viewer` 的功能思路，不复制其整套应用。查看器与 GMR 子集都在项目内运行。
模型导入用于检查；训练仍使用已注册任务的机器人配置。GMR 当前限 LAFAN1 骨架和 G1 29-DoF，23-DoF 任务需使用对应 NPZ。Viser 当前回放 checkpoint，ONNX 是独立导出产物。

动作集中在 `src/assets/motions/`。页面支持导入和选择文件，转换输出自动命名、入库，并选为训练动作。

前端开发、模块边界和验证说明见 [工作台文档](docs/workbench.md)。

### 独立试玩区

`playground/` 是与工作台和训练流程隔离的静态浏览器应用：MuJoCo WASM 在浏览器内运行物理，ONNX Runtime Web 执行已挑选策略。它仅用于交互展示，不作为训练验收工具，也不会连接工作台端口或读取训练日志。

```bash
cd playground
npm install
npm run dev
```

当前包含 Go2 前倒立和后立试玩策略。构建生产静态站点使用 `npm run build`；详细的资产准备方式见 [`playground/README.md`](playground/README.md)。

`csv-to-npz` 直接在本地生成 NPZ，不需要 Weights & Biases。`--output-name` 只传文件名时，输出会保存到输入 CSV 的同一目录；也可传入完整路径。G1-23DoF 动作使用 `--robot g1_23dof`。

任务命名统一使用分类前缀，例如 `Unitree-<Robot>-Flat` / `Unitree-<Robot>-Rough` 和 `LainLab-<Robot>-Flat` / `LainLab-<Robot>-Rough`。OpenDoge 属于自研机器人，归入 `LainLab` 分类，提供 Flat、Rough、Getup、Handstand、Jump 版本，以及把前四类技能合成单策略的 `LainLab-OpenDoge-Skills-Flat`。速度任务支持 A2、As2、Go2、OpenDoge、G1、G1-23Dof、H1_2、H2 和 R1；动作跟踪任务支持 G1 与 G1-23Dof；当前平地倒地自恢复任务为 `LainLab-OpenDoge-Getup`。单策略多技能的完整设计与验收见 [docs/opendoge_multiskill.md](docs/opendoge_multiskill.md)。

## 运行验证与已知限制

截至 2026-09-09，项目的 CPU 单环境 smoke 验证覆盖了已注册任务的环境构建、`reset` 与一步零动作执行，并检查了观测和奖励是否为有限值。所有 Go2 任务（含 Go2 skill）、基础速度任务、Cartpole、Yam 操作任务和 29-DoF G1 跟踪任务均已通过。

- 跟踪任务必须提供与机器人自由度匹配的 NPZ 动作文件。命令行训练/回放时请设置 `motion_file`；仓库本地的 29-DoF G1 示例不能用于 `G1-23Dof` 跟踪任务。
- `Unitree-H2-Rough` 和 `Unitree-R1-Rough` 当前的 MuJoCo-Warp 接触缓冲容量不足：分别至少需要 `nconmax=80` 和 `nconmax=141`。提高该配置值后，两项任务可完成 smoke；在修复默认配置前请勿直接用默认设置启动训练。
- Smoke 只验证初始化和单步数值稳定性，不代表策略已经收敛，也不替代长时间训练、checkpoint 回放或真机验收。

## 开发

```bash
make format
make check
```

机器人差异集中在 `src/tasks/robots/` 下的机器人包，通用 MDP 位于 `src/tasks/velocity/core.py`、`tracking/core.py` 等任务类型包。新增同类机器人时，通常在对应机器人包中增加 profile 和注册函数，无需复制整套 MDP。

## 部署

`deploy/` 保留参考项目中的源码，但不再内置 ONNX Runtime、MuJoCo 二进制或训练好的策略。请按 [deploy/README.md](deploy/README.md) 配置系统依赖并把导出的策略放到对应机器人目录。

## 上游与许可

本项目参考 `unitree_rl_mjlab` 的项目边界和部署代码，并基于 mjlab 1.6 API 重新组织。代码使用 Apache-2.0 许可；第三方组件保留各自许可。

---

# LainLab (English)

**LainLab** is a reinforcement-learning, simulation-validation, and deployment project for robots from multiple vendors, built on **mjlab 1.6.0**. `mjlab` remains the pinned, general-purpose dependency; LainLab owns robot assets, task-specific behavior, training-validation contracts, and deployment adapters, while upstream supplies the simulation framework.

The repository currently includes Unitree assets together with LainLab-owned OpenDoge as validated starting points. LainLab continues to add its own robot designs and assets from other vendors. Every new asset follows the same path—asset, profile/task, validation, and deployment adapter—while the framework layer remains vendor-neutral.

> Note: the repository and project are named **LainLab**; the source root is `src/`, and Python package paths use the `src.*` namespace.

### Name and intent

**LainLab** combines **Lain** and **Lab**. The name Lain is inspired by *Serial Experiments Lain* (玲音), particularly its reflection on connection, identity, and the boundary between networked and physical reality. Lab represents an open robotics experimentation platform that brings different robot morphologies, motion data, training policies, and deployment validation into one repeatable process. The project connects different robot morphologies to one training and validation cadence while preserving each robot's physical and control characteristics. It has no official affiliation with the work or its rights holders.

### Positioning and differentiators

| Dimension | Upstream `mjlab` | `unitree_rl_mjlab` reference project | LainLab |
| --- | --- | --- | --- |
| Role | General MuJoCo/Warp simulation and RL framework | Training and real-robot deployment project centered on Unitree robots | Multi-vendor robot asset, task, and deployment layer built on `mjlab` |
| Robot scope | General framework capabilities | Unitree-centered | Unitree and LainLab OpenDoge are current starting points; more self-designed and other-vendor assets are planned by design |
| Framework maintenance | Upstream owns simulation, managers, MDPs, runners, and viewers | The project carries robot tasks and deployment workflows | Pins `mjlab==1.6.0` and uses thin adapters to avoid framework duplication and reduce upgrade cost |
| Task organization | General tasks and extension mechanisms | Primarily velocity tracking, motion imitation, and real-robot control | Robot profiles, isolated assets, and registrable tasks express differences; Go2 also includes source-traceable skill migrations |
| Verifiability | Framework-level capabilities | Training and deployment workflows | Project-level asset, configuration, observation/reward contract, and CPU-smoke tests; known limitations are documented |
| Motion data | General tooling and interfaces | Motion preparation for Unitree workflows | CSV-to-NPZ conversion runs locally and validates joint columns and DoF for the target robot, without W&B |

LainLab turns robot onboarding into a repeatable product path: assets have clear ownership, task differences have boundaries, training interfaces stay consistent, validation outcomes are reproducible, and deployment code is kept separate from model artifacts. A new vendor or self-designed robot enters the usable task set after asset/control review, smoke testing, and task validation.

## Layout

```text
LainLab/
├── src/
│   ├── assets/          # Per-robot MJCF files, meshes, and example motions
│   ├── tasks/
│   │   ├── velocity/    # Shared velocity-tracking MDP
│   │   ├── tracking/    # Shared motion-tracking MDP
│   │   ├── amp/         # Reserved generic AMP core
│   │   └── robots/      # Robot composition, profiles, and registration
│   └── cli.py           # train / play / list-envs entry points
├── tests/               # LainLab compatibility tests
└── deploy/              # Real-robot deployment source (no models or bundled runtimes)
```

Layering principles:

- `mjlab` provides simulation, managers, common MDPs, runners, and viewers.
- `src.assets` owns project-specific robot descriptions.
- `src.tasks` expresses only task differences such as robot names, contacts, and action scales.
- The CLI registers LainLab tasks before reusing mjlab 1.6 training and playback.

## Installation

Python 3.10–3.13 is required. Training requires an NVIDIA GPU.

```bash
uv sync --extra cu128
```

For CPU-only configuration checks:

```bash
uv sync --extra cpu
```

## Usage

```bash
# List tasks
uv run list-envs

# Train OpenDoge (in-house robots use the LainLab group)
uv run train LainLab-OpenDoge-Flat --env.scene.num-envs 4096

# Play a bundled baseline directly in Viser
uv run baseline-play opendoge flat --device cuda:0 --num-envs 1
uv run baseline-play opendoge rough --device cuda:0 --num-envs 1 --no-terminations True
uv run baseline-play opendoge getup --device cuda:0 --num-envs 1

# Run a random-action configuration smoke test
uv run play LainLab-OpenDoge-Flat --agent random --num-envs 1

# Single-policy multi-skill: walk / handstand / getup / jump
uv run train LainLab-OpenDoge-Handstand --env.scene.num-envs 4096
uv run train LainLab-OpenDoge-Jump --env.scene.num-envs 4096
uv run train LainLab-OpenDoge-Skills-Flat --env.scene.num-envs 4096
uv run opendoge-calibrate --write          # handstand / jump target calibration
uv run opendoge-eval LainLab-OpenDoge-Skills-Flat --checkpoint <ckpt>
uv run opendoge-eval LainLab-OpenDoge-Skills-Flat --checkpoint <ckpt> --transitions

# Replay a local policy
uv run play LainLab-OpenDoge-Flat --checkpoint-file logs/.../model_1000.pt

# Convert a G1 CSV motion into mjlab tracking format
uv run csv-to-npz --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1-subject2 --robot g1
```

`csv-to-npz` creates an NPZ locally and does not require Weights & Biases. If `--output-name` is only a filename, the output is written next to the input CSV; an absolute or relative output path may also be used. Use `--robot g1_23dof` for G1-23DoF motions.

Tasks use a category prefix, for example `Unitree-<Robot>-Flat` / `Unitree-<Robot>-Rough` and `LainLab-<Robot>-Flat` / `LainLab-<Robot>-Rough`. OpenDoge is an in-house robot and belongs to `LainLab`; Flat, Rough, Getup, Handstand, and Jump variants are registered, plus `LainLab-OpenDoge-Skills-Flat`, which folds the first four skills into a single policy. Velocity tasks support A2, As2, Go2, OpenDoge, G1, G1-23Dof, H1_2, H2, and R1. Motion tracking tasks support G1 and G1-23Dof. The current flat-ground fall-recovery task is `LainLab-OpenDoge-Getup`. The full single-policy multi-skill design and acceptance record is in [docs/opendoge_multiskill.md](docs/opendoge_multiskill.md).

## Validation status and known limitations

As of 2026-09-09, CPU single-environment smoke checks cover environment construction, `reset`, and one zero-action step for registered tasks, while checking that observations and rewards remain finite. All Go2 tasks (including Go2 skills), baseline velocity tasks, Cartpole, Yam manipulation tasks, and 29-DoF G1 tracking tasks passed.

- Tracking tasks need an NPZ motion file whose degrees of freedom match the robot. Set `motion_file` for training or playback. The local 29-DoF G1 example cannot be used for `G1-23Dof` tracking.
- The default MuJoCo-Warp contact-buffer capacity is currently insufficient for `Unitree-H2-Rough` and `Unitree-R1-Rough`: they require at least `nconmax=80` and `nconmax=141`, respectively. Both tasks pass smoke checks after this temporary configuration adjustment; do not start training with their defaults until the configuration is fixed.
- Smoke tests validate initialization and single-step numerical stability only. They do not demonstrate policy convergence and do not replace long training, checkpoint playback, or real-robot validation.

## Development

```bash
make format
make check
```

Robot differences are centralized under `src/tasks/robots/`. Shared MDPs live in task-type packages such as `velocity/core.py` and `tracking/core.py`. Adding a similar robot typically consists of a profile and registration in its robot package, without copying the shared MDP implementation.

## Deployment

`deploy/` retains source from the reference projects but does not bundle ONNX Runtime, MuJoCo binaries, or trained policies. Follow [deploy/README.md](deploy/README.md) to configure system dependencies, then place exported policies in the corresponding robot directory.

## Upstream and license

This project follows the project boundary and deployment source of `unitree_rl_mjlab`, reorganized around the mjlab 1.6 API. LainLab is licensed under Apache-2.0; third-party components retain their own licenses.
