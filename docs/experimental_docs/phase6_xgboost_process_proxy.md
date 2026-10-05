# Phase 6：XGBoost 过程代理研究快照

## 使用边界

该分支保存 2025 年 1～9 月初赛／复赛联合研究中的 XGBoost 过程代理代码、特征定义、聚合验证指标和本地文件校验清单。赛事原始表、逐行特征／标签、模型权重、逐行 OOF 与 10 月候选预测**未进入 Git**，只能在符合赛事规则的受控本地环境保存。`local_artifact_inventory.json` 仅记录相对路径、字节数和 SHA-256，不包含这些文件的数值。

这份快照**不是可仅从 GitHub 克隆后重现训练的完整数据包**。完整运行依赖本地受控文件；不得从清单的哈希值反推或替代原文件。仓库现有合规策略要求“Git 保存代码、配置、文档和聚合指标，本地受控存储保存训练数据与模型”，本分支沿用该边界。

## 模型与标签口径

过程代理为 XGBoost 3.4.1，两目标 `generator_1`、`generator_all` 分别拟合，目标是起点可用的同记录负荷代理；目标先减训练中位数，使用 `reg:absoluteerror`、`hist`、最多 15 叶、220 轮、学习率 0.045。初赛样本权重为 0.25，复赛样本权重为 1；复赛历史窗口为 180 天。XGBoost 代理与冻结的四周同星期／过去七日同刻模板融合，组成 B7；EXP-C 仅对短周期 `generator_all` 施加复赛高频 LightGBM 残差，因此**最终系统不是纯 XGBoost**。参数和训练规模见 [`base_model_fit.json`](../../results/experiments/phase6_xgboost_proxy/final_jan_sep_b7/base_model_fit.json) 与 [`training_manifest.json`](../../results/experiments/phase6_xgboost_proxy/final_jan_sep_b7/training_manifest.json)。

初赛每 15 分钟负荷记录的统计口径未经赛事方确认，作为过程代理标签，**不宣称它是正式 15 分钟区间均值**。复赛验证真值为完整 15 个一分钟负荷在未来 `[t,t+15)` 等区间的均值。预处理保留工业 0 值、不跨阶段插值；初赛过程观测延迟 15 分钟视为可用。模型输入不得使用起点 `t` 之后的真实过程量或负荷。列级定义与可用性见 [`phase6_xgboost_feature_manifest.json`](phase6_xgboost_feature_manifest.json)。

重要合规差异：这次历史研究把已下发的初赛 `Pre_test_` 负荷纳入过程代理拟合。仓库现行的初赛训练数据访问政策禁止评分目录进入训练。因此，该快照只记录已有研究结果，**不应直接作为仓库的正式合规训练或竞赛提交方案**；是否允许跨阶段复用这段数据须按复赛正式规则及项目负责人审查后另行决定。

## 9 月时间外验证

下表的 MAPE 是 `mean(abs(预测−真值)/真值)`，双目标平均为两个目标各自 MAPE 的算术平均。短周期各目标有效 23,004 个预测单元，长周期各目标 271,824 个；验证窗口及标签与旧 LightGBM 过程代理相同。原始聚合指标见 [XGBoost 指标](../../results/experiments/phase6_xgboost_proxy/oof_residual_b7/C_august_selected_on_B7_metrics.csv) 和 [LightGBM 对照](../../results/experiments/phase6_xgboost_proxy/lightgbm_reference/C_august_selected_on_B7_metrics.csv)。

| 周期 | 目标 | LightGBM B7+C MAPE | XGBoost B7+C MAPE |
| --- | --- | ---: | ---: |
| 2 小时 | `generator_1` | 11.3637% | 10.8422% |
| 2 小时 | `generator_all` | 5.6175% | 5.5471% |
| 2 小时 | 双目标平均 | 8.4906% | **8.1946%** |
| 24 小时 | `generator_1` | 12.5817% | 12.4436% |
| 24 小时 | `generator_all` | 8.9920% | 8.9694% |
| 24 小时 | 双目标平均 | 10.7869% | **10.7065%** |

6～8 月逐月 OOF 共 8,832 个起点；8 月选择关闭 `generator_1` 残差、保留 `generator_all` 残差。选择依据在 [`c_target_selection.json`](../../results/experiments/phase6_xgboost_proxy/oof_residual_b7/c_target_selection.json)，逐 horizon 指标和其余消融结果在同一聚合结果目录。9 月只作为时间外验证；面向 10 月的最终模型随后纳入 9 月 OOF，不能把 9 月离线成绩解释成最终权重的无偏实测。

按原项目的本地换算公式、假设 `anomaly=normal`，9 月离线换算为 **67.5904／100**。它不是官方分数，不能与 Stage2.2 的官方 58.1578 或其他版本官方分数直接相减。10 月候选文件未在此仓库发布；[推理聚合审计](../../results/experiments/phase6_xgboost_proxy/final_jan_sep_b7/inference_audit.json) 仅记录 960 个起点、数值范围、运行时间和区间均值爬坡边界。区间均值边界不等于单分钟或单台机组爬坡合规证明。

## 本地文件恢复与校验

本地受控环境需恢复 `features_experiments/exp_bc/data/` 中的 7 个特征／训练文件、`wjt/gas_stage22_rebuild/features/history_process_features.npz`、`results/exp_bc_xgboost/` 的模型和其余结果。所需相对路径、字节数和 SHA-256 见 [`local_artifact_inventory.json`](../../results/experiments/phase6_xgboost_proxy/local_artifact_inventory.json)。该清单由 [`build_phase6_artifact_inventory.py`](../../codefiles/tools/build_phase6_artifact_inventory.py) 在受控本地目录生成；它不读取赛事原始目录，也不输出训练样本值。

安装本实验补充依赖：

```powershell
python -m pip install -r requirements\phase6_xgboost_proxy.txt
```

在本地数据已恢复、赛事规则审查通过后，可运行以下针对性代码测试；它们使用合成数据，不读取真实赛事文件：

```powershell
python -m unittest discover -s tests -p 'test_exp_bc_*.py' -v
```

完整训练入口依次位于 `features_experiments.exp_bc.b_variants`、`oof_residual`、`finalize`，推理入口为 `infer_october`。运行前必须确认所有本地依赖的 SHA-256 匹配，并再次检查比赛期数据使用许可。不得把本地模型、特征 Parquet 或候选 CSV 复制到此 Git 仓库。
