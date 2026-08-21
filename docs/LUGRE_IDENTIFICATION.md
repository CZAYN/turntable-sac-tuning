# LuGre 摩擦参数辨识与启用说明

## 1. 当前状态

时域内核已经实现完整 LuGre 方程、稳定离散状态更新以及摩擦转矩、鬃毛状态和鬃毛状态变化率轨迹。当前仿真配置已启用 `lugre`，以下两个参数暂时使用工程启动假设：

- `coulomb_friction_nm = 0.015 N·m`：库仑摩擦转矩 `τc`；
- `static_friction_nm = 0.020 N·m`：最大静摩擦转矩 `τs`；
- 二者不确定度均为 `±10%`。

这些值由当前 `0.5 A` 训练限流和 `Kt=0.1633 N·m/A` 的转矩裕量选取，不是实机辨识结果。现有 `σ0=187`、`σ1=2.42` 和 `ωs=0.05061 rad/s` 同样需要实验确认，其中 `σ1` 的瞬态阻尼影响较大，当前标记为高风险临时参数。完整 LuGre 只允许用于仿真与训练验证，禁止把当前数值解释为硬件参数结论。

## 2. 需要采集的数据

建议记录统一时间戳下的：

| 字段 | 单位 | 必需性 | 说明 |
|---|---|---|---|
| `time_s` | s | 必需 | 单调递增时间 |
| `iq_a` | A | 必需 | 实际转矩电流，不是电流指令 |
| `omega_rad_s` | rad/s | 必需 | 实际机械角速度 |
| `theta_rad` | rad | 建议 | 用于校核低速与反转数据 |
| `load_torque_nm` | N·m | 条件必需 | 存在已知外部负载时记录 |
| `direction` | — | 建议 | 正转或反转标记 |

至少进行以下三组实验：

1. 正反向多档恒速运行，覆盖 Stribeck 低速区和较高稳定速度；
2. 从静止开始缓慢增加正、负方向电流，记录刚发生连续运动时的启动电流；
3. 小幅低速往复与速度反转，用于检查预滑动、零速记忆和动态摩擦。

所有实验都应在限流、限速和急停保护下进行。优先使用空载或已知负载；未知负载会与摩擦转矩混在一起。

## 3. 参数计算关系

机械方程为：

$$
J\dot\omega=K_t i_q-\tau_f-\tau_L
$$

因此动态数据中的等效摩擦转矩为：

$$
\tau_f=K_t i_q-J\dot\omega-\tau_L
$$

恒速稳态时 `dω/dt` 接近零，可以减少加速度微分噪声：

$$
\tau_f\approx K_t i_q-\tau_L
$$

利用正反向恒速数据拟合稳态 Stribeck 曲线：

$$
\tau_{f,\mathrm{ss}}(\omega)=
\operatorname{sgn}(\omega)
\left[\tau_c+(\tau_s-\tau_c)
\exp\left(-\left|\frac{\omega}{\omega_s}\right|^\alpha\right)\right]
+\sigma_2\omega
$$

缓慢电流爬升的连续启动点用于校核 `τs`。正向和反向数据应先分别拟合；只有两侧差异处于实验重复误差内，才使用对称 LuGre 参数。

现有三环频响数据不能单独确定 `τc` 和 `τs`，因为它不包含静止启动和低速反转所需的信息。

## 4. 写入配置

取得实机辨识数据后修改 `config/motor_physics.json`：

1. 用辨识值替换或确认 `nominal_parameters.coulomb_friction_nm` 和 `static_friction_nm`；
2. 将两个 `parameter_status` 改为可追溯的实验编号或数据来源；
3. 填写有依据的 `uncertainty_fraction`；
4. 确认最坏范围仍满足：

   $$
   \tau_s(1-u_s)\ge\tau_c(1+u_c)>0
   $$

5. 将 `friction_model.active` 改为 `lugre`。

配置校验还要求：

$$
|z(0)|\le\frac{\tau_s}{\sigma_0}
$$

## 5. 重建与验证

参数写入后依次执行：

```bash
python scripts/build_physics_model.py \
  --current-feasibility-report \
  data/processed/current_pidf_feasibility_40khz/multirate_feasibility_report.json
python scripts/build_final_test_dataset.py --overwrite-unconsumed
python -m pytest -q
python scripts/check_tuning_env.py --quick
```

随后必须使用新的实验目录重新训练。146维 observation、13维物理模型集合、Replay Buffer、transition 数据和环境状态都不能与旧训练协议混用。

启用前必须检查：

- 正负速度下摩擦方向正确；
- 零速状态能够保留合理的预滑动记忆；
- 反转过程连续、无 NaN 或数值爆炸；
- `25 us` 基础步与更小时间步结果收敛，外环仍按 `200 us` 更新；
- 低速启动、小位置运动及扰动场景不违反电流、速度和电压限制。
