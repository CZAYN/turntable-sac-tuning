# CGS 转台单次上传服务器流程（Legacy）

> **历史文档，不是当前发布流程。** 本文对应旧训练与旧封存最终测试结构。当前六指标协议改变了配置、observation、Reward、阶段、参数空间和最终测试 schema；旧发布包、候选锁与消费记录均不兼容。重新启用服务器前必须重新生成发布包与封存测试清单。

发布包只需上传一次，内部包含两个相互隔离的 ZIP：

- `training/elc_rl_server_training.zip`：正式训练、验证和候选选择。
- `final_test/elc_rl_server_final_test.zip`：封存最终测试；唯一候选冻结前不得解压或使用。

## 1. 解压训练包

```bash
unzip elc_rl_server_release.zip -d elc_rl_server_release
cd elc_rl_server_release
mkdir -p runtime/training
unzip training/elc_rl_server_training.zip -d runtime/training
cd runtime/training
```

更新或建立服务器环境。当前版本新增 Numba，旧环境不能只更新源码：

```bash
unset PYTHONPATH
source /data/l50063953/miniconda3/etc/profile.d/conda.sh
conda env update -n elc-rl-server -f environment-server.yml
conda activate elc-rl-server
python -m pip install --no-deps --no-build-isolation -e .
python -c "import numba; print(numba.__version__)"
```

## 2. 预检与工程检查

```bash
python scripts/check_server_runtime.py --device cuda
python scripts/benchmark_parallel_env.py --n-envs 4 --steps-per-env 64

python scripts/train_sac.py \
  --seed 20260801 \
  --device cuda \
  --n-envs 4 \
  --engineering-check-steps-per-stage 4 \
  --output-dir outputs/formal_engineering_check
```

工程检查结果不能参与正式候选选择。

## 3. 启动正式训练

默认同时启动 3 个种子，每个种子使用 4 个并行环境：

```bash
python -u scripts/train_all_seeds.py --device cuda
```

监控：

```bash
cat logs/sac_training/launcher_state.json
tail -f logs/sac_training/seed_20260801.log
```

正常中断后使用完全相同的代码、配置、数据和环境数恢复：

```bash
python -u scripts/train_all_seeds.py --device cuda --resume
```

## 4. 选择并冻结唯一候选

三个种子全部完成后：

```bash
python scripts/select_final_candidate.py
sha256sum outputs/sac_training/selection/final_candidate.npz \
  > outputs/sac_training/selection/FINAL_CANDIDATE.sha256
chmod 444 outputs/sac_training/selection/final_candidate.npz
```

从此停止训练、验证和候选选择，不得根据最终测试结果更换候选。

## 5. 此时才解压封存最终测试包

回到发布包根目录：

```bash
cd ../..
mkdir -p runtime/final_test
unzip final_test/elc_rl_server_final_test.zip -d runtime/final_test
cp runtime/training/outputs/sac_training/selection/final_candidate.npz \
  runtime/final_test/final_candidate.npz
cd runtime/final_test
sha256sum final_candidate.npz
```

该哈希必须与训练目录中 `FINAL_CANDIDATE.sha256` 的值一致。

## 6. 锁定并执行唯一一次最终测试

```bash
python scripts/lock_final_candidate.py \
  final_candidate.npz \
  --output final_candidate_training_lock.json \
  --declare-training-complete

python scripts/run_final_test.py \
  final_candidate.npz \
  --candidate-lock final_candidate_training_lock.json
```

最终输出：

```text
outputs/final_test_v2/
├── final_test_report.json
└── FINAL_TEST_CONSUMED.json
```

测试结果是最终结果，不得用于重新训练或重新选择候选。
