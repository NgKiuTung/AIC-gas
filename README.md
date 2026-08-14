# 煤气发电预测与发电优化

本仓库用于竞赛初赛阶段的训练数据治理、多步发电预测和约束发电优化。当前严格时间外推三折验证的预测 MAPE 为 **5.4671%**；平衡调度方案在历史回放中的收益提升为 **3.2159%**，Bootstrap 95% 置信区间为 **2.8985%–3.5400%**，最终优化验收 **14/14** 通过。

> 数据合规：比赛数据不随仓库发布。任何训练、实验或 CI 流程都不得读取 `dataset/初赛-评分所用测试集`，评分集也不得参与训练或模型选择。

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements\base.txt -r requirements\gpu.txt -r requirements\dev.txt
pip install -e .
pytest
python codefiles\pipelines\run_validation.py
```

关键入口：

- 项目导航与复现说明：`docs/index.md`
- 稳定流水线：`codefiles/pipelines/`
- 可复用模块：`codefiles/src/gas_power/`
- 历史实验脚本：`codefiles/legacy/`
- 实验登记：`results/registry/experiment_registry.csv`
- 版本化存档：本机 `results/releases/<version>/`（禁止提交或作为公开 Release 发布）

每个里程碑版本保留独立的本地快照；Git 只保存代码、配置、实验报告、聚合指标和合规证据，不保存赛事原始数据、逐行真实值、特征缓存、模型或可恢复真实时序的图。完整归档仅在本机受控保存，并用 SHA-256 校验。仓库必须保持 Private；当前未附开源许可证，默认保留全部权利。
