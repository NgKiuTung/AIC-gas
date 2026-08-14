# 实验索引

| 阶段 | 步骤 | 主要结论 | 报告 |
|---|---:|---|---|
| 数据预处理 | 01–07 | 时间轴审计、插值基准、全空列策略、监督特征 | `preprocessing/` |
| 预测建模 | 08–23 | 基线、GPU XGBoost、时间因素、多步预测与验证 | `forecasting/03_多步发电预测与时间因素实验报告.md` |
| 清洗再优化 | 24–37 | 清洗再审计、特征消融、动态融合，最佳 MAPE 5.4671% | `forecasting/04_数据清洗再审计与预测优化实验报告.md` |
| 评估协议整改 | Phase 1 / 46 | 5 个两天嵌套伪测试 MAPE 4.7149%，冻结 ModelSpec | `forecasting/05_嵌套时间评估与模型规范冻结实验报告.md` |
| 生产模型整改 | Phase 2 / 47 | d5/d6 GPU 重训，全量重载与 CPU parity 22/22 通过 | `forecasting/06_生产模型重训与重载一致性实验报告.md` |
| 发电优化 | 38–45 | 约束审计、历史回放、稳健性、可视化、14/14 验收 | `optimization/` |

逐脚本状态、输入输出和报告映射记录在 `results/registry/experiment_registry.csv`。版本化结论位于 `results/releases/<version>/`，不得覆盖旧版本。
