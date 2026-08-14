# 实验索引

| 阶段 | 步骤 | 主要结论 | 报告 |
|---|---:|---|---|
| 数据预处理 | 01–07 | 时间轴审计、插值基准、全空列策略、监督特征 | `preprocessing/` |
| 预测建模 | 08–23 | 基线、GPU XGBoost、时间因素、多步预测与验证 | `forecasting/03_多步发电预测与时间因素实验报告.md` |
| 清洗再优化 | 24–37 | 清洗再审计、特征消融、动态融合，最佳 MAPE 5.4671% | `forecasting/04_数据清洗再审计与预测优化实验报告.md` |
| 发电优化 | 38–45 | 约束审计、历史回放、稳健性、可视化、14/14 验收 | `optimization/` |

逐脚本状态、输入输出和报告映射记录在 `results/registry/experiment_registry.csv`。版本化结论位于 `results/releases/<version>/`，不得覆盖旧版本。

