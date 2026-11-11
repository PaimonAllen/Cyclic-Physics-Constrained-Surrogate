# 数据与预处理

数据使用 SI 单位。位置为 m，速度为 m/s，加速度为 m/s²，推力与阻力为 N，壳体应变能为 J。源文件夹的 `case_config.json` 提供控制、几何与材料上下文。每个 `rail_history.npz` 保留时间、十二导轨响应、参考命令、推力和能量。完整壳体自由度与求解器实现属于开发项目。

`scripts/prepare_data.py` 根据固定行选择读取原始记录，重建位置、速度、推力、能量、下一状态、区间加速度和 `R = u − 7.27a`。训练、验证与测试划分由 `data/splits/records.csv` 和 `cyclic_groups.json` 固定。标准化统计仅由训练记录计算。网络输入由参考命令、控制与静态上下文构成，当前真实响应用于监督标签。

已处理表中的 `X`（86 列）及额外能量梯度标签是历史辅助字段。发布的响应网络通过 `_build_features` 构建实际使用的 6 个局部和 29 个全局特征。48 层、17 通道的历史表示属于原始数据生成背景，当前网络输入以此处的实际特征列表为准。

名义表的训练、验证和测试记录数分别为 16,605、2,254、669，对应 118、20、6 个工况。名义正式结果使用其中 3 个典型工况的 347 个时间记录。分组表对应 9,428、3,652、6,448 个记录和 84、32、28 个工况，其中正式测试为 25 个工况的 6,126 个记录，另外 3 个工况是单独记录的历史 OOD。两个表使用相同的 19,528 个记录，划分与训练集统计不同。

## 数据划分索引

`0 = train`, `1 = validation`, `2 = test`.

`trajectory_id` 索引 `data/trajectories/case_XXX/`，`step_index`、`time`、`time_next` 与 `dt` 确定记录区间。`data/splits/records.csv` 提供每行在两种协议下的划分标识。

## 局部输入

| Field | 含义 |
|---|---|
| `p_ref` | 参考位置 |
| `pdot_ref` | 参考速度 |
| `p_ref_terminal` | 最终参考位移 |
| `arctan_u_scale` | 前馈载荷幅值 |
| `feedforward_force` | 当前时刻前馈载荷 |
| `pddot_ref_fd` | 参考速度区间差分 |

## 全局上下文

29 个全局特征依次为

```text
arctan_k, sim_time, dt, tau, arctan_ramp,
pid_kp, pid_kd, pid_ki, pid_error_sign,
log10_abs_pid_kp, log10_abs_pid_kd,
terminal_mean, terminal_std, terminal_rms,
u_scale_mean, u_scale_std, u_scale_rms,
geometry.Rb, geometry.Rd, geometry.Ru,
geometry.Hb, geometry.Hd, geometry.Hu, geometry.thickness,
material.E, material.nu, material.rho, material.rho_b, material.damping_c
```

归档响应数组保留 float64，标准化网络输入转换为 float32，误差汇总采用 float64。
