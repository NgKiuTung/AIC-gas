# AIC-Gas 技术报告：Phase6 Short × Phase7.1 Long

> 图按证据链组织。图内不放大标题；结果级主张、统计定义和解释放在正文与图注中。

## 摘要

- 最终线上成绩：60.0903 / 100
- Short：Phase6 lineage，30.9525 / 50
- Long：Phase7.1，29.1378 / 50
- Phase7.1：276 项特征；270 项仅使用 origin 及以前信息，6 项为 origin 前已知信息，0 项目标泄漏特征。

## 1. 最终提交的双分支来源

核心主张：最终提交由两个相互独立的来源合并，而不是串行模型链。Phase7.1 使用处理后缓存与 276 特征研究 Long；Short 保持 Phase6 冻结版本；二者最终合并为 60.0903。

插入：`figureA_pipeline_and_score.pdf`

建议图注：
**Figure 1 | Provenance and composition of the final submission.** a, Parallel provenance of the selected Phase7.1 Long branch and frozen Phase6 Short branch before final combination. b, Scale of the processed cache and model feature space. c, Online Short and Long score contributions.

## 2. Phase7.1 特征架构

核心主张：276 项特征具有明确物理来源和算子结构，而不是平铺变量集合。过程信号按 gas production、air heaters、mixed-gas inflow、gas holder、industrial users 与 generator fuel use 组织；每类过程信号映射为状态、滚动统计、动态变化与 coverage/QC 特征，同时保留 6 项已知未来信息。

插入：`figureB_feature_architecture.pdf`

建议图注：
**Figure 2 | Physical and causal architecture of the Phase7.1 feature space.** a, Feature counts across physical signal domains and transformation families. b, Composition of the 276 features by transformation family. c, Feature-provenance map showing 270 causal-at-origin features, six known-before-origin features and no target-derived features.

## 3. Long 模型筛选证据

核心主张：候选模型不是全局替换 reference。Phase7.1 按 `2 targets × 4 horizon bands` 独立筛选；最终 `generator_1` 四个 band 保持 reference，`generator_all` 四个 band 均选择 XGB 180d t12。

插入：`figureC_model_selection_evidence.pdf`

建议图注：
**Figure 3 | Component-wise evidence for Phase7.1 Long selection.** a, Screening-stage mean gain relative to the frozen reference across candidates and components. b, Confirmation-stage mean gain together with worst- and last-fold diagnostics for accepted `generator_all` bands. c, Final component assignment parsed from the frozen selection metadata.

## 4. 线上性能演进

核心主张：线上分数从参考基线 58.1578 提升至 58.5600，并在与冻结 Short 组合后达到 60.0903。

插入：`figureD_online_performance.pdf`

建议图注：
**Figure 4 | Online performance progression and task contributions.** a, Discrete score progression from the reference baseline through the Phase7.1 Long probe to the final combination, with score deltas. b, Short and Long contributions to the final score. c, Online accuracies of the two forecast horizons.

## 5. 数据质量与审计

核心主张：处理后缓存不仅保留特征矩阵，也保留 feature-level availability、raw-field missingness 与异常隔离事件，可追溯数据质量问题发生在哪些字段和阶段。

插入：`figureE_data_quality.pdf`

建议图注：
**Figure 5 | Data-quality diagnostics for the Phase7.1 processed cache.** a, Train/test feature-availability distributions with median and interquartile intervals. b, Highest raw-field missingness percentages after normalizing archived missing counts by source row counts. c, Field-level concentration of isolated anomalous values on a logarithmic count scale.

## 6. 复现与归档

记录：
- Phase7.1 processed-data SHA256 清单
- Phase7.1 screening / confirmation / selection / summary
- 最终 Long artifact SHA256
- Git commit / tag
- 本图包版本与 QA 输出

## 7. 局限与后续工作

只写实际证据支持的局限。Phase6 历史 processed-data 数值文件尚未全部恢复时，应明确 provenance 边界，不得把重建版本表述为原始历史 artifact。
