# Windows 本机环境与训练前检查

## 1. 当前环境

项目使用已经安装在以下位置的本机 Python 环境：

```text
D:\OtherSoftware\elc_RL
```

PowerShell 中统一通过绝对解释器路径运行，避免误用系统 Python 或其他 Conda 环境：

```powershell
$ElcPython = "D:\OtherSoftware\elc_RL\python.exe"
& $ElcPython --version
```

当前环境目标组合记录在 `environment-local.yml`，包括 Python 3.11、PyTorch CUDA、Stable-Baselines3、Gymnasium、Numba、NumPy、SciPy、TensorBoard 与 pytest。

## 2. 安装当前源码

在仓库根目录执行：

```powershell
$ElcPython = "D:\OtherSoftware\elc_RL\python.exe"

& $ElcPython -m pip install `
  --no-deps `
  --no-build-isolation `
  -e .
```

这是 editable 安装；后续修改 `src/` 后无需重复复制源码。

## 3. 训练前检查

### 3.1 依赖一致性

```powershell
& $ElcPython -m pip check
```

### 3.2 CUDA与主要库

```powershell
& $ElcPython -c "import torch, gymnasium, stable_baselines3, numba; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'gymnasium': gymnasium.__version__, 'sb3': stable_baselines3.__version__, 'numba': numba.__version__})"
```

预期本机 GPU 为 RTX 4060 Laptop GPU，`cuda` 应为 `True`。

### 3.3 自动化测试

```powershell
& $ElcPython -m pytest -q
```

旧版封存最终测试集与六指标协议不兼容，保留时只能作为历史归档。当前代码使用带版本号的 `physics_motor_six_metric_test_v2` 测试集和清单；不得把旧候选、旧锁文件或旧消费记录混入新版流程。

### 3.4 环境快速检查

```powershell
& $ElcPython scripts/check_tuning_env.py --quick
```

该检查验证：

- 四阶段环境能够初始化；
- 146维 observation 合法；
- 11维动作与阶段掩码一致；
- Gymnasium 与 Stable-Baselines3 接口检查通过；
- SAC 能在 CUDA 上创建网络；
- 每个阶段可以完成少量环境 transition。

### 3.5 仿真并行基准

```powershell
& $ElcPython scripts/benchmark_parallel_env.py `
  --n-envs 2 `
  --steps-per-env 64
```

该命令只测量物理环境 transition，不进行 SAC 梯度更新，不属于训练。8 GB 显存的本机建议从2个环境开始，再根据基准决定未来是否提高到4个。

## 4. 当前停止点

完成上述检查后停止。当前轮不得执行以下命令：

```text
scripts/train_sac.py
scripts/train_all_seeds.py
scripts/select_final_candidate.py
scripts/run_final_test.py
```

只有用户明确确认开始训练后，才建立新的实验目录并启动单种子工程检查。由于当前协议与历史实验不兼容，不得使用 `--resume` 载入旧 checkpoint。

## 5. 未来训练原则

未来获准训练时：

1. 先运行单种子、少量步数的工程检查；
2. 检查四阶段日志、Reward方向、六指标与GPU/CPU资源；
3. 本机8 GB显存优先使用 `parallel-seeds=1`，环境数从2开始；
4. 工程检查输出与正式训练输出严格分目录；
5. 正式训练、候选选择和最终测试全部使用新协议指纹，不复用任何历史训练状态。
