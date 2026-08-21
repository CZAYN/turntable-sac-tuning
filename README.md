# CGS 转台三环 SAC 调参

本项目使用 Soft Actor-Critic（SAC）在电机数学物理模型上联合调节三环 PIDF 与速度环 DOBC，共 11 个控制参数。当前正式协议只评价用户提供的六类控制性能指标：闭环带宽、增益裕度、相位裕度、超调、上升时间和调节时间。频域评价已经改为与时域内核一致的多采样率离散小信号状态空间：电流环 40 kHz，速度环、位置环与 DOBC 5 kHz，不再使用独立的连续域控制代理。

当前工作入口是 Windows 本机环境 `D:\OtherSoftware\elc_RL`。本轮只完成多采样率离散模型统一、40 kHz电流环非训练可行性扫描、依赖检查、自动化测试、环境快速检查和极短工程链路检查，**不启动正式训练或最终测试**。

## 文档导航

- [项目流程](PROJECT_WORKFLOW.md)：唯一六指标目标、四阶段环境、observation、SAC 网络、Reward、数据隔离与兼容边界。
- [本机环境与训练前检查](LOCAL_SETUP.md)：使用 `D:\OtherSoftware\elc_RL` 完成安装和训练前验证。
- [LuGre 参数辨识](docs/LUGRE_IDENTIFICATION.md)：当前临时摩擦参数的适用边界与未来实机辨识要求。

`SERVER_TRAINING.md` 和 `SERVER_RELEASE.md` 仅保留为历史服务器操作记录，不是当前协议的执行入口。

## 唯一性能目标

| 环路 | 闭环带宽 | 最小增益裕度 | 最小相位裕度 | 最大超调 | 最大上升时间 | 最大调节时间 |
|---|---:|---:|---:|---:|---:|---:|
| 电流环 | 1500 Hz | 5 dB | 65° | 20% | 0.05 s | 0.1 s |
| 速度环 PIDF + DOBC | 100 Hz | 5 dB | 65° | 50% | 0.5 s | 1 s |
| 位置环 | 20 Hz | 2 dB | 70° | 20% | 0.5 s | 1 s |

闭环带宽是需要逼近的双向目标；两个裕度是下限；超调、上升时间和调节时间是上限。DOBC 没有独立 Cost，它的两个参数与速度 PIDF 三个参数共同依据速度环六项指标调整。

## 当前训练协议

- 四阶段：`current → speed → position → joint`。
- 11 维连续动作：9 个 PIDF 参数和 2 个 DOBC 参数。
- 146 维纯物理 observation：96 维物理模型频响上下文、6 维摩擦上下文、11 维参数状态、18 维归一化性能误差、11 维动作掩码和4维阶段标识。
- Actor：两层 `256 × 256`；双 Critic：各两层 `256 × 256`。
- 有效候选的单步奖励：

  \[
  r_t=10\left(C_{t-1}-C_t\right)-0.02C_t
  \]

- 数值无效、闭环发散或仿真终止时奖励为 `-100` 并终止 episode。

训练阶段只从40个训练物理模型中抽样。16个验证模型独立报告六指标泛化与有效性，不混入在线 Reward 或训练模型聚合 Cost。隔离的24个最终测试模型必须按新协议重新生成和封存后才能使用。

## 训练前快速检查

在 PowerShell 中执行：

```powershell
$ElcPython = "D:\OtherSoftware\elc_RL\python.exe"

& $ElcPython -m pip install --no-deps --no-build-isolation -e .
& $ElcPython -m pip check
& $ElcPython -m pytest -q
& $ElcPython scripts/check_tuning_env.py --quick
& $ElcPython scripts/benchmark_parallel_env.py --n-envs 2 --steps-per-env 64
```

这些命令只执行安装、测试、环境交互检查和不含 SAC 梯度更新的仿真基准，不会启动训练。详细说明见 [LOCAL_SETUP.md](LOCAL_SETUP.md)。

## 兼容边界

本次修改改变了性能目标、Cost、Reward、训练阶段、observation 结构、频域动力学和最终测试报告结构，训练协议版本已升级。因此此前生成的模型、Replay Buffer、checkpoint、候选参数、排行榜及最终测试消费记录均不可恢复或复用；必须使用新的输出目录开启一次全新实验。

实测频响目前只作为训练之外的模型误差诊断，不进入 observation、Reward 或 Replay Buffer。LuGre 当前仍含未经实机辨识的临时仿真参数，任何仿真候选都不能直接宣称为实机最优参数。
