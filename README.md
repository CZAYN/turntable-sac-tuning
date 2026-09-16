# CGS 转台三环控制参数智能整定

当前方法：**CrossQ + 阶段条件 Batch Renormalization（BRN）+ 难度感知模型采样 + 四阶段课程训练**。控制对象为三环 PIDF 与速度环并联 DOBC，训练协议为 Protocol 7。

本仓库提供当前方法的源码、配置、训练/验证物理模型数据和核心测试。不提供预训练权重、设计 Word/PDF、历史实验、演示材料或发布安装包。仓库名称中的 SAC 和部分 Python 文件名沿用历史命名；默认配置和训练算法均为 CrossQ。

## 方法框架

```text
40 个训练电机模型 → 难度采样 → 每个 episode 选择一个模型
    ↓
146 维观测 → CrossQ Actor → 11 维参数增量 → 阶段掩码
    ↓
位置 PIDF → 速度 PIDF + DOBC → 电流 PIDF → 电机物理仿真
    ↓
频域与时域六指标 → Cost / Reward → Replay Buffer
    ↓
CrossQ 双 Critic / Actor 更新（阶段条件 BRN）
    ↓
current → speed → position → joint
    ↓
训练/验证集合审计 → 约束优先候选选择 → 参数与四阶段策略快照
```

- **控制与仿真**：电流环 25 μs（40 kHz），速度环、位置环及 DOBC 200 μs（5 kHz）；包含 LuGre 摩擦和多采样率离散频域评价。
- **状态与动作**：观测由 96 维频响、6 维摩擦上下文、11 维参数、18 维指标误差、11 维掩码和 4 维阶段编码组成；动作是 11 维连续参数增量。
- **CrossQ**：256/256 隐藏层、双 Critic，无目标 Critic；当前与下一状态/动作拼接前向，TD 目标停止梯度；每 3 次 Critic 更新执行一次 Actor/熵更新。
- **阶段 BRN**：每层为四个阶段分别保存运行统计，共享层内仿射参数；处理跨阶段回放数据的分布差异。
- **难度采样**：每阶段前 80 个完成的 episode 使用均匀采样（40×2，并不保证每个模型恰好访问两次）；随后混合 0.3 均匀概率和 0.7 难度概率，EMA 更新率 0.1，单模型概率上限 0.1。

| 阶段 | 调整参数 | 每种子 transitions |
|---|---|---:|
| current | 电流 PIDF，3 个 | 30,000 |
| speed | 速度 PIDF 与 DOBC，5 个 | 70,000 |
| position | 位置 PIDF，3 个 | 40,000 |
| joint | 全部 11 个 | 120,000 |

每种子共 260,000 transitions，四个并行环境；默认三个独立种子为 20260801、20260802、20260803。阶段之间保留网络、优化器与回放状态。

参数顺序固定为：

```text
kppos, kipos, kdpos, kpspeed, kispeed, kdspeed,
kgspeed, tauspeed, kpcurr, kicurr, kdcurr
```

## 安装与运行检查

使用 Python 3.11 或更新版本，在仓库根目录运行：

```bash
python -m pip install -e ".[training,test]"
python -m pip check
python scripts/check_server_runtime.py --device cpu
python -m pytest -q
```

GPU 训练需要与机器驱动匹配的 CUDA 版 PyTorch；CPU 运行可显式传入 `--device cpu`。依赖范围见 `pyproject.toml`。运行检查会构造环境、执行一个环境步、创建策略并检查动作与网络结构，不启动正式训练。

## 训练、续训与参数评估

唯一随附的训练配置是 `config/crossq_training.json`，省略 `--config` 时自动使用它。

**单种子训练：**

```bash
python scripts/train_sac.py --seed 20260801 --device cuda --n-envs 4 --output-dir outputs/crossq_training/seed_20260801
```

**三个种子训练：**

```bash
python scripts/train_all_seeds.py --device cuda --n-envs 4 --parallel-seeds 3 --output-root outputs/crossq_training
```

显存或内存不足时可将 `--parallel-seeds` 降为 1。每种子的环境数会影响交互轨迹，应在一次运行及其续训过程中保持一致。

**续训：**在原训练命令后追加 `--resume`，保持输出路径、配置和环境数一致。默认每 10,000 transitions 保存检查点，保留最近两份，包含模型、优化器、Replay Buffer、环境、采样器和随机状态。仅有模型 ZIP 无法精确续训。

**多种子候选选择与完整参数审计：**

```bash
python scripts/select_final_candidate.py --runs-root outputs/crossq_training --output-dir outputs/crossq_training/selection
python scripts/audit_physics_candidate.py outputs/crossq_training/selection/final_candidate.npz
```

选择要求至少三个完成的正式种子候选。排序优先考虑有效性、全指标达标、最大越限和越限数量，再比较验证 Cost、训练 Cost 与种子号。使用 `--engineering-check-steps-per-stage` 生成的短运行候选不能用于正式选择。

## 冻结策略推理

先完成训练，再使用该运行生成的 `run_manifest.json`、`seed_summary.json` 和 `models/` 下四阶段快照：

```bash
python scripts/infer_frozen_policy.py --run-dir outputs/crossq_training/seed_20260801 --device cpu --output outputs/frozen_inference.json
```

默认每阶段 4 个 episode，步数沿用训练配置。可增加 `--episodes-per-stage 1 --steps-per-episode 2` 做短调用检查；这不代表完整性能验证。输出路径必须尚不存在。

推理按运行清单选择正确的 CrossQ 加载器，固定网络权重和 BRN 统计，并通过现有候选审计选择参数。训练输入指纹必须匹配；旧版本训练产物需要其原始源码与配置，不能直接当作本版本产物使用。

## 评价定义与数据边界

| 环路 | 带宽目标 / Hz | 最小增益裕度 / dB | 最小相位裕度 / ° | 最大超调 | 最大上升时间 / s | 最大调节时间 / s |
|---|---:|---:|---:|---:|---:|---:|
| 电流 | 1500 | 5 | 65 | 20% | 0.05 | 0.1 |
| 速度 | 100 | 5 | 65 | 50% | 0.5 | 1 |
| 位置 | 20 | 2 | 70 | 20% | 0.5 | 1 |

带宽验收容差分别为 ±10%、±20%、±10%。Cost 的带宽误差采用对数归一化，其余指标采用单边越限误差，再经过 Huber 损失与频域/时域聚合。正常步骤的奖励为 `10 × (上一 Cost − 当前 Cost) − 0.02 × 当前 Cost`；数值无效或仿真发散等情况使用 -100 并终止 episode。

`physics_motor_ensemble.npz` 保存 **40 个训练模型和 16 个验证模型的物理参数**，不是神经网络权重。训练模型用于交互与难度统计；验证模型用于审计与候选选择，不是未接触的独立最终测试集。`training_anchor.json` 是工程初始化点，不是最终性能结论。

本仓库不随附历史结果与独立测试集，不将已有验证指标解释为实机性能或任意新电机的整定成功率。当前任务是仿真参数整定；LuGre 等物理参数包含仿真假设。

## 代码导航

| 路径 | 职责 |
|---|---|
| `src/elc_rl/crossq.py` | CrossQ、阶段 BRN、Actor 与双 Critic |
| `src/elc_rl/sac_training.py` | 四阶段训练、恢复、候选池与排序 |
| `src/elc_rl/tuning_env.py` | Gymnasium 环境、观测、动作与 Reward |
| `src/elc_rl/plant_sampling.py`、`sampling_vec_env.py` | 难度统计、概率约束与并行同步 |
| `src/elc_rl/physics_motor_model.py`、`simulation_kernel.py` | 电机模型、LuGre、PIDF、DOBC 与时域仿真 |
| `src/elc_rl/discrete_loop_model.py`、`physics_evaluator.py` | 离散频域模型与性能审计 |
| `src/elc_rl/performance_targets.py`、`controller_parameters.py` | 目标、Cost、验收与参数映射 |
| `src/elc_rl/frozen_policy.py` | 冻结推理与训练来源检查 |
| `config/` | 当前方法配置、控制目标及物理配置 |
| `data/processed/` | 必需物理模型、参数空间与初始化数据 |
| `scripts/` | 运行检查、训练、选择、审计与推理入口 |
| `tests/` | 核心算法、物理评价与训练恢复测试 |

内部保留 SB3 SAC 的兼容加载和训练基础设施；仓库不再提供旧 SAC 对照配置或历史消融入口。Git 历史保留，旧文件可从历史提交查阅。
