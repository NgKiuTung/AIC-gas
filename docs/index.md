# 项目文档导航

本项目采用“稳定模块 + 稳定流水线 + 历史实验 + 版本快照”的组织方式。所有非入口文档均位于 `docs/`，根目录仅保留 GitHub 所需的 `README.md`。

## 文档结构

- `competition/`：赛题理解与官方材料；官方 PDF 仅本地保留，不进入 Git。
- `design/`：架构、数据契约、建模思路和优化数学约束。
- `experimental_docs/`：按预处理、预测、优化阶段归档实验报告。
- `guides/`：环境、复现、实验登记和版本回退流程。

当前结论与可视化索引见 [实验索引](experimental_docs/experiment_index.md)，代码执行入口见 [复现指南](guides/reproduction.md)，版本存档见 [版本回退指南](guides/versioning_and_rollback.md)，实验资产保护规则见 [实验归档策略](guides/experiment_archive_policy.md)。
