# 煤气发电预测与发电优化

本仓库用于竞赛初赛阶段的训练数据治理、多步发电预测和约束发电优化。历史三折结果 **5.4671% MAPE** 属于同一 OOF 上选参后的研究指标；预注册的 5 个两天嵌套时间伪测试给出 **4.7149% MAPE / 0.95285 分**，相对 persistence 降低 8.11%。该结果仍是训练期回放而非官方评分成绩。冻结的 d5/d6 生产组件已完成 RTX 4060 重训、全量重载及 CPU parity 验证；正式原始表推理与提交链仍待 Phase 3–4 完成。

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

训练期评估与生产模型构建：

```powershell
python codefiles\pipelines\run_nested_validation.py
python codefiles\pipelines\run_production_refit.py --device cuda
python codefiles\pipelines\validate_production_bundle.py
```

关键入口：

- 项目导航与复现说明：`docs/index.md`
- 稳定流水线：`codefiles/pipelines/`
- 可复用模块：`codefiles/src/gas_power/`
- 历史实验脚本：`codefiles/legacy/`
- 实验登记：`results/registry/experiment_registry.csv`
- 当前冻结模型规范：`configs/model_spec.yaml`
- Phase 1 评估图：`results/figures_safe/phase1/`
- 版本化存档：本机 `results/releases/<version>/`（禁止提交或作为公开 Release 发布）

每个里程碑版本保留独立的本地快照；Git 只保存代码、配置、实验报告、聚合指标和合规证据，不保存赛事原始数据、逐行真实值、特征缓存、模型或可恢复真实时序的图。完整归档仅在本机受控保存，并用 SHA-256 校验。仓库必须保持 Private；当前未附开源许可证，默认保留全部权利。
