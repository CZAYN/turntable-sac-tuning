# CGS 转台正式 SAC 服务器训练（Legacy）

> **历史文档，不是当前执行入口。** 本文记录旧服务器环境与操作方式，尚未按当前六指标、四阶段、146维 observation 协议重新验证。当前项目转为 Windows 本机 `D:\OtherSoftware\elc_RL`，训练前操作以 `LOCAL_SETUP.md` 为准。旧服务器 checkpoint、Replay Buffer、候选与输出不可用于当前协议。

本训练包只用于基于 `physics` 的正式长程训练。它不包含封存测试数据，也不使用冒烟训练产生的模型或输出。

## 1. 解压和建立环境

以下示例使用 Linux Bash；所有项目路径均为相对路径，核心程序同时兼容 Windows 和 Linux Python。

```bash
unzip elc_rl_server_training.zip -d elc_rl_server_training
cd elc_rl_server_training
unset PYTHONPATH
source /data/l50063953/miniconda3/etc/profile.d/conda.sh
conda env create -f environment-server.yml
conda activate elc-rl-server
```

当前服务器环境组合为 Python 3.11、PyTorch 2.5.1、CUDA 12.1、
Stable-Baselines3 2.9.0、Gymnasium 1.2.3、Numba 0.66.0 和
TensorBoard 2.20.0。
`environment-server.yml` 通过 Conda 通道安装这些依赖，不访问被公司代理拦截的
PyPI 或 `download.pytorch.org`。环境建立后只需安装项目本身：

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

## 2. 服务器预检

```bash
python scripts/check_server_runtime.py --device cuda
```

预检必须确认 CUDA 可用、训练输入哈希有效、磁盘空间充足，并且环境能够完成一次安全交互。

可选的环境吞吐基准：

```bash
python scripts/benchmark_parallel_env.py --n-envs 4 --steps-per-env 64
```

该基准只测物理环境采样，不包含 SAC 梯度更新。

## 3. 启动并行训练

```bash
python -u scripts/train_all_seeds.py --device cuda
```

默认配置同时运行 3 个随机种子，每个种子建立 4 个独立环境进程，共使用约
12 个 CPU 核心。物理仿真在 CPU 上并行，Actor/Critic 更新共用 GPU。默认值可在
`config/sac_training.json` 的 `parallelism` 中调整，也可在命令行临时覆盖：

```bash
python -u scripts/train_all_seeds.py \
  --device cuda \
  --parallel-seeds 2 \
  --n-envs 2
```

每增加一个环境，SAC 梯度更新数按 transition 数等比例增加，因此不会因为批量采样
而降低训练强度。环境数、阶段步数、checkpoint 间隔和验证间隔必须可以整除。

需要单独运行一个种子时：

```bash
python -u scripts/train_sac.py --seed 20260801 --device cuda --n-envs 4
```

每个种子的默认输出位于：

```text
outputs/sac_training/seed_<seed>/
```

## 4. 中断后续训

程序接收到正常的终止信号后，会在当前环境步结束时保存模型、Replay Buffer、训练状态和随机状态。

```bash
python -u scripts/train_all_seeds.py --device cuda --resume
```

checkpoint 还会保存每个并行环境的采样状态和随机状态。续训时配置、源码、训练数据
哈希和环境数必须与原运行完全一致。

## 5. TensorBoard

```bash
tensorboard --logdir outputs/sac_training
```

## 6. 选择唯一候选

三个种子全部完成后运行：

```bash
python scripts/select_final_candidate.py
```

输出位于：

```text
outputs/sac_training/selection/
├─ candidate_leaderboard.json
├─ final_candidate.npz
└─ final_candidate_audit.json
```

候选选择只使用40个训练电机和16个验证电机。将整个 `selection` 目录以及三个种子的 `seed_summary.json` 下载回本机，再进行候选锁定和一次性封存测试。

## 7. 工程检查模式

工程检查只验证正式训练程序，不产生可参与候选选择的参数：

```bash
python scripts/train_sac.py \
  --seed 20260801 \
  --device cuda \
  --n-envs 4 \
  --engineering-check-steps-per-stage 4 \
  --output-dir outputs/formal_engineering_check
```

正式训练不得添加这个参数。

也可以一次检查全部种子的并行启动和日志隔离：

```bash
python -u scripts/train_all_seeds.py \
  --device cpu \
  --n-envs 4 \
  --parallel-seeds 3 \
  --engineering-check-steps-per-stage 4
```

该模式默认输出到 `outputs/sac_training_engineering_check/`，不会污染正式训练目录。
