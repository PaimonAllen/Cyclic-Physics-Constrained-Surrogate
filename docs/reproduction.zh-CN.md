# 复现指南

按 README 安装依赖后，依次运行预处理与检查点评估。新生成结果写入 `outputs/`，发布的归档结果保留在 `models/` 和 `results/reported/`。

```bash
python scripts/prepare_data.py --protocol nominal
python scripts/prepare_data.py --protocol grouped
python scripts/evaluate.py --protocol all --model all --check-archived
python scripts/diagnostics.py --model full --run run_01
python scripts/evaluate_supplementary.py --model full --run run_01
python scripts/predict.py
python scripts/verify_release.py
```

| 模型标识 | 组件或骨干网络 |
|---|---|
| `full` | 完整循环模型 |
| `no_pd_projection` | 独立学习速度，取消硬 PD 投影 |
| `no_cyclic_coupling` | 参数量匹配的逐导轨残差块 |
| `no_invariant_energy_pool` | 有序可训练导轨汇聚 |
| `no_force_balance` | 独立阻力输出头 |
| `no_hard_closures` | 同时独立学习速度与阻力 |
| `soft_pinn` | SiLU 密集网络，软物理约束，运动学权重 2 |
| `soft_pinn_tanh` | 6 层 tanh 密集网络，宽度 924 |
| `soft_pinn_fourier` | 6 层 tanh 密集网络，宽度 915，96 个时间 Fourier 特征 |

Fourier 特征尺度为 0.5、2 和 8，每个尺度包含 16 个频率，分别计算正弦与余弦。共同种子实验均训练 3,000 轮，归档批量大小为 512。各模型的优化器和损失设置以检查点随附的配置为准。

`--check-archived` 将各检查点重新推理的结果与其归档预测进行核对。为容纳 float32 CPU/GPU 后端差异及 PD 反解的误差放大，每个预测字段的无量纲 RMS 差异应小于其归档预测 RMS 的 0.05%。程序记录每次实际差异，评估期间权重保持固定。

`diagnostics.py` 输出代数 PD 残差、导轨力平衡残差、原生时间网格上的时间导数诊断、导轨功和循环重标记误差。导数指标与查询网格和差分约定有关。`evaluate_supplementary.py` 使用发布的原生记录区间和固定权重，输出六项响应误差及导轨饱和记录比例。这里的通用诊断采用文档所述约定，归档论文表格保留其具体实验口径。

训练默认每 100 轮生成 Loss 和相对 RMSE 图。需要适配本机内存时，可通过 `--batch-size` 调整批量大小。`--dataset` 可指定其他具有相同字段接口的数据表。重新进行正式比较时应保持归档配置，短轮次和小样本运行用于检查程序执行。

## 文件完整性

`SHA256SUMS` 记录发布文件的 SHA-256。`release_manifest.json` 用开发项目中的相对路径追踪科学数据来源。检查点与归档预测均按字节复制，源轨迹导出保留原始导轨及能量数值，并将工况元数据中的路径改为可移植形式。来源路径使用相同的公开运行编号。代码修改集中于独立运行所需的导入、默认路径、命令入口和排版。
