# CGS 转台三环 PIDF + DOBC SAC 调参流程

## 1. 当前协议与范围

项目针对位置外环、速度中环和电流内环构成的串级伺服系统，使用 SAC 调整9个 PIDF 参数与2个 DOBC 参数。物理电机、LuGre 摩擦、测量延迟、执行延迟和控制器限幅都在环境内部计算。

当前阶段只执行训练前准备：配置校验、代码测试、环境快速检查、数值有效性检查和无梯度仿真基准。未经用户再次确认，不执行工程训练、正式多种子训练、候选选择或最终测试。

## 2. 控制结构与11维参数

```text
位置指令 → 位置 PIDF → 速度指令
                       ↓
速度反馈 → 速度 PIDF ─────────┐
电流与速度反馈 → DOBC补偿 ────┼→ 电流指令 → 电流 PIDF → 电机
                              │
                              └→ 速度、位置反馈
```

DOBC 是速度控制器内部的补偿通道，不是独立的第四个控制环。线性频域模型将 DOBC 返回路径并入速度开环与速度闭环；位置环继续使用已经闭合的“速度 PIDF + DOBC”动态。因此改变 `kgspeed` 或 `tauspeed` 会影响速度环和位置环指标，但不会改变独立电流环指标。

| 模块 | 参数 |
|---|---|
| 位置 PIDF | `kppos, kipos, kdpos` |
| 速度 PIDF + DOBC | `kpspeed, kispeed, kdspeed, kgspeed, tauspeed` |
| 电流 PIDF | `kpcurr, kicurr, kdcurr` |

参数顺序固定为：

```text
[kppos, kipos, kdpos,
 kpspeed, kispeed, kdspeed, kgspeed, tauspeed,
 kpcurr, kicurr, kdcurr]
```

## 3. 唯一评价指标

正式评价只使用下表中的六项指标。

| 环路 | 闭环带宽 \(B^*\) | 最小增益裕度 \(G^*\) | 最小相位裕度 \(\phi^*\) | 最大超调 \(O^{\max}\) | 最大上升时间 \(t_r^{\max}\) | 最大调节时间 \(t_s^{\max}\) |
|---|---:|---:|---:|---:|---:|---:|
| 电流环 | 1500 Hz | 5 dB | 65° | 20% | 0.05 s | 0.1 s |
| 速度环 PIDF + DOBC | 100 Hz | 5 dB | 65° | 50% | 0.5 s | 1 s |
| 位置环 | 20 Hz | 2 dB | 70° | 20% | 0.5 s | 1 s |

所有数值按表头字面解释：闭环带宽单位为 Hz，增益裕度单位为 dB，相位裕度单位为度。闭环带宽需要逼近目标；两个裕度为下限；三个时域指标为上限。

### 3.1 频域定义

频域评价与时域仿真统一采用多采样率离散结构：电流环周期
\(T_i=25\,\mu\mathrm{s}\)（40 kHz），速度环、位置环与 DOBC 周期
\(T_o=200\,\mu\mathrm{s}\)（5 kHz）。电流 PIDF、电压执行状态、电气状态、
机械状态与 LuGre 状态每个25 µs基础步更新一次；每8个基础步依次更新
位置环、速度环与DOBC，并在其余7步保持外环指令。三个已给定的200 µs
执行或测量延迟保持不变，不随采样周期缩短。对环路 \(l\)，在零工作点对
相同的未饱和多采样率更新进行小信号线性化：

\[
x_{k+1}=A_lx_k+B_le_{l,k},\qquad
y_{\mathrm{fb},k}=C_{\mathrm{fb},l}x_k,\qquad
y_k=C_{y,l}x_k
\]

记本环频域宏步周期为 (h_l)：电流环 (h_i=25\,\mu\mathrm{s})，
速度环和位置环 (h_\omega=h_\theta=200\,\mu\mathrm{s})。在环路误差
求和点得到离散开环返回比与实际输出闭环响应：

\[
L_l(z_l)=C_{\mathrm{fb},l}(z_lI-A_l)^{-1}B_l
\]

\[
T_l(z_l)=C_{y,l}\left[z_lI-\left(A_l-B_lC_{\mathrm{fb},l}\right)\right]^{-1}B_l,
\qquad z_l=e^{j2\pi fh_l}
\]

闭环带宽取实际输出闭环响应相对低频增益首次下降3 dB的频率：

\[
\left|T_l(e^{j2\pi B_lh_l})\right|
=\frac{|T_l(e^{j2\pi f_0h_l})|}{\sqrt{2}}
\]

其中 \(f_0\) 为评价网格的最低正频率。增益裕度与相位裕度使用离散
开环 \(L_l(e^{j2\pi fh_l})\) 的经典定义，并在存在多个交点时取最不利值。
没有 \(-180^\circ\) 相位交点时，增益裕度视为无穷，并以有限上限值
写入数值报告。每个环路的评价频率严格低于本环 Nyquist 频率：电流环
20 kHz，速度环和位置环2.5 kHz。饱和、编码器量化与硬终止不属于
小信号裕度模型，仍由非线性时域评价单独检查。

若评价网格内没有0 dB交点，相位裕度使用未达标哨兵值并产生较大 Cost；
若没有闭环−3 dB交点，带宽使用网格上界并判定目标未通过。二者都不等于
数值无效，SAC 可以继续调整并恢复；只有非有限响应或离散闭环 I/O 模态
不稳定才会终止当前 episode。

### 3.2 时域定义

对正阶跃参考值 \(r_\infty\)，归一化输出为：

\[
y_n(t)=\frac{y(t)}{r_\infty}
\]

超调为：

\[
O_l=\max\left(0,\max_t y_n(t)-1\right)
\]

上升时间使用首次10%到90%的时间差：

\[
t_{r,l}=t_{90\%}-t_{10\%}
\]

调节时间使用最终值±2%误差带：

\[
t_{s,l}=\min\left\{t_0:\ |y_n(t)-1|\le0.02,\ \forall t\ge t_0\right\}
\]

若仿真窗口内未达到90%或未进入并保持在±2%误差带中，报告会明确标记未达标，并使用仿真窗口形成越限 Cost，不能把窗口终点当作成功调节。

## 4. Cost 与 Reward

记 \([x]_+=\max(0,x)\)。对每个环路，六项归一化误差为：

\[
e_B=\frac{\ln(B/B^*)}{\ln(1.1)}
\]

\[
e_G=\left[\frac{G^*-G}{G^*}\right]_+,
\qquad
e_\phi=\left[\frac{\phi^*-\phi}{\phi^*}\right]_+
\]

\[
e_O=\left[\frac{O}{O^{\max}}-1\right]_+,
\quad
e_r=\left[\frac{t_r}{t_r^{\max}}-1\right]_+,
\quad
e_s=\left[\frac{t_s}{t_s^{\max}}-1\right]_+
\]

带宽误差是双向的；裕度、超调和时间误差是单边的。达到裕度下限或优于时域上限后不会继续获得额外 Cost 优势。

所有归一化误差使用 Huber 损失：

\[
\rho(x)=
\begin{cases}
\frac12x^2,& |x|\le1\\
|x|-\frac12,& |x|>1
\end{cases}
\]

单环频域、时域及综合 Cost 为：

\[
C_{\mathrm{freq},l}
=\frac{\rho(e_B)+\rho(e_G)+\rho(e_\phi)}{3}
\]

\[
C_{\mathrm{time},l}
=\frac{\rho(e_O)+\rho(e_r)+\rho(e_s)}{3}
\]

\[
C_l=0.5C_{\mathrm{freq},l}+0.5C_{\mathrm{time},l}
\]

在 40 个训练物理模型的完整审计中，每个环、每个评价域都按“平均表现 + 最坏模型”聚合。令 \(d\in\{\mathrm{freq},\mathrm{time}\}\)：

\[
C_{d,l}^{\mathrm{ens}}
=0.5\,\operatorname{mean}_{m}(C_{d,l,m})
+0.5\,\max_{m}(C_{d,l,m})
\]

随后使用 \(C_l=0.5C_{\mathrm{freq},l}^{\mathrm{ens}}+0.5C_{\mathrm{time},l}^{\mathrm{ens}}\)。16 个验证模型不进入 Reward、候选排序或参数更新，只使用相同六项指标独立报告泛化结果。

单环阶段使用对应的 \(C_l\)。联合阶段防止某一环被平均值掩盖：

\[
C_{\mathrm{joint}}
=\frac{C_i+C_\omega+C_\theta}{3}
+0.5\max(C_i,C_\omega,C_\theta)
\]

有效候选的单步 Reward 为：

\[
r_t=10\left(C_{t-1}-C_t\right)-0.02C_t
\]

若出现非有限数值、闭环发散或物理仿真提前终止，则：

\[
r_t=-100
\]

并结束当前 episode。目标尚未达到本身由六项 Cost 表达，不额外引入其他性能指标。

DOBC 与速度 PIDF 共用速度环的六项指标，没有独立 Cost。

## 5. 四阶段课程

| 阶段 | 可调整参数 | 默认步数 |
|---|---|---:|
| `current` | `kpcurr, kicurr, kdcurr` | 30,000 |
| `speed` | `kpspeed, kispeed, kdspeed, kgspeed, tauspeed` | 70,000 |
| `position` | `kppos, kipos, kdpos` | 40,000 |
| `joint` | 全部11个参数 | 120,000 |

四阶段合计260,000步。阶段之间连续使用同一个 SAC 模型与 Replay Buffer，只切换动作掩码和阶段状态。

## 6. Observation 与动作

| Observation 项 | 维数 | 含义 |
|---|---:|---|
| `sampled_frf` | 96 | 当前完整物理模型的三环基础对象频率响应上下文 |
| `friction_context` | 6 | 当前 LuGre/粘性摩擦不确定性上下文 |
| `parameter_state` | 11 | 当前11个控制参数的归一化状态 |
| `performance_metrics` | 18 | 三环各6个归一化误差，顺序为带宽、增益裕度、相位裕度、超调、上升时间、调节时间 |
| `action_mask` | 11 | 当前阶段允许调整的参数 |
| `stage` | 4 | 四阶段独热编码 |

总维数为：

\[
96+6+11+18+11+4=146
\]

动作是 \([-1,1]\) 内的11维连续增量，经参数变换、单步幅度限制与阶段掩码后映射到实际控制参数。

## 7. SAC 网络

环境使用 Stable-Baselines3 `MultiInputPolicy`。字典 observation 展平为146维特征。

Actor 主干：

```text
146 → 256 → ReLU → 256 → ReLU
                         ├→ μ      (11)
                         └→ log σ  (11)
```

Actor 使用重参数化高斯采样并经 `tanh` 映射为11维动作。

两个 Critic 完全独立，每个接收状态与动作拼接后的157维输入：

```text
Q1: 157 → 256 → ReLU → 256 → ReLU → 1
Q2: 157 → 256 → ReLU → 256 → ReLU → 1
```

目标 Critic 使用 \(\tau=0.005\) 软更新。配置还包括 `batch_size=256`、`gamma=0.98` 和每条 transition 一次梯度更新。

## 8. 模型集合与隔离

| 集合 | 数量 | 用途 | 是否进入在线 Reward/训练 Cost |
|---|---:|---|---:|
| 训练模型 | 40 | episode 抽样、Replay Buffer、训练 Cost | 是 |
| 验证模型 | 16 | 与训练 Cost 隔离的六指标泛化报告 | 否 |
| 最终测试模型 | 24 | 候选确定后的独立最终评价 | 否 |

每个 episode 只抽取一个完整物理电机实例；三环频域和时域评价始终使用同一组物理参数。训练模型聚合 Cost 与验证模型报告严格分离，验证数据不会回流到在线 Reward。

实测三环频响只用于离线模型误差诊断，不进入训练 observation、Reward、Replay Buffer 或运行时 Cost。

## 9. 本机训练前流程

当前使用 `D:\OtherSoftware\elc_RL\python.exe`。本轮只执行：

```text
editable安装
  ↓
依赖一致性检查
  ↓
完整pytest
  ↓
快速环境检查
  ↓
无SAC梯度更新的并行仿真基准
  ↓
停止，等待用户确认是否训练
```

命令见 [LOCAL_SETUP.md](LOCAL_SETUP.md)。

## 10. 新旧实验兼容边界

当前协议已改变以下内容：

- 性能目标与 Cost；
- Reward；
- 训练阶段与动作掩码；
- observation 键、维数和输入指纹；
- 参数搜索空间；
- 40 kHz/5 kHz 多采样率计划与物理模型 schema；
- 训练/验证聚合语义；
- 最终测试报告结构和依赖指纹。

因此历史模型、Replay Buffer、checkpoint、候选文件、排行榜和最终测试消费记录只能作为历史证据，不能续训或作为当前候选。正式训练必须写入新的输出目录；最终测试集合及清单也必须按当前协议重新构建、重新封存后才能使用。

## 11. 实机边界

物理模型中的 LuGre 摩擦参数仍包含未经实机辨识的临时仿真假设。当前输出只能说明候选在数学物理模型与声明不确定性范围内的表现，不能直接宣称为实机最优。进入 HIL 或实机前仍需确认驱动器限幅、编码器、采样与延迟、摩擦参数以及安全保护策略。
