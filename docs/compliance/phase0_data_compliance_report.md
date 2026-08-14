# Phase 0 数据与仓库合规整改报告

## 1. 结论

截至 2026-08-14，AIC-gas 已完成本轮最高优先级的数据泄露面收敛和 Git 历史净化：GitHub 仓库保持为 Private，公开 Release 已删除，旧标签已删除，`main` 已重建为不含旧父提交的新根提交。对远端仓库进行独立浅克隆后，赛事敏感内容扫描结果为 0 项。

本轮未读取、枚举、哈希或训练使用 `dataset/初赛-评分所用测试集`。本报告只处理仓库合规与交付边界，不改变预测任务定义。

## 2. 与官方赛题的对应关系

官方题目明确要求：

- 预测只能使用参考时刻及之前的数据，不得使用未来信息或测试标签；
- 初赛训练期为 2025-01-01 至 2025-04-30，评分期为 2025-05-01 至 2025-05-02；
- 核心预测指标为 `1-MAPE`；
- 短周期预测覆盖 15 至 120 分钟，提交字段使用 `generator_*_t+XX_pred`；
- 禁止传播、转售、公开展示或泄露赛事数据；
- 初赛 ZIP 内只允许一个 UTF-8 编码的 `result.csv`，且不得缺行或重复。

完整映射见 `docs/compliance/official_requirements_matrix.md`。后续清洗、验证、训练和提交链必须持续满足上述约束，尤其不得把评分测试集纳入训练、调参或指标计算。

## 3. 已完成整改

| 项目 | 整改前 | 整改后 |
|---|---:|---:|
| Git 跟踪敏感内容扫描 | 119 项 | 0 项 |
| GitHub 仓库可见性 | 曾为 Public | Private |
| GitHub Release | 曾有 1 个含完整实验归档的 Release | 0 个 |
| 远端旧标签 | 2 个 | 0 个 |
| 远端分支 | `main` 指向旧历史 | 仅 `main`，指向净化根提交 |
| GitHub forks | 0 | 0 |

具体措施：

1. 将 `results/releases/`、`results/visualizations/`、`results/optimization/`、模型、特征、提交结果及逐行时序产物加入 Git 禁止范围。
2. 从 Git 索引移除已跟踪的图、优化明细、逐行真实值/预测值和版本归档；本机原文件未删除。
3. 新增跟踪内容扫描器、暂存区审计和 CI 阻断检查。
4. 删除 GitHub Release `v0.1.1-archive` 及两个旧远端标签。
5. 使用净化后的文件树建立新根提交，并通过 `force-with-lease` 更新远端 `main`。
6. 在独立临时克隆中复核远端文件树和历史。

## 4. 可复核证据

- 净化后远端 `main`：`0f2b5c8af5a874267a5f6266de92dd92650c8bb4`
- 远端可见性：`private=true`，`visibility=private`
- 远端 Release 数量：0
- 远端标签数量：0
- 远端分支：仅 `main`
- 首次净化后独立浅克隆只见新的安全根提交，旧历史不可达；最终报告提交只会建立在该安全根提交之上
- 独立浅克隆敏感内容扫描：0 项，PASS
- 仓库检查：compileall、Ruff、pytest、敏感内容审计、优化 dry-run、目录校验、归档校验全部通过
- 本地完整实验归档：49/49 文件通过校验
- 本地完整实验归档 SHA-256：`E4B258E0C3DB537C4FE39473CC2167AEA05EAF186BCA2ACA4C65473A66350880`

机器可读证据位于 `results/compliance/` 和 `results/repository_validation/`。

## 5. 私有回退与实验存档

旧 Git 历史已保存为仅本机可见的完整 bundle：

- 路径：`results/releases/private_history_pre_sanitize.bundle`
- 大小：2,122,395 bytes
- SHA-256：`60D0F1F97576E695DC0C0EDB163203777E78AAD42CD624A70D8E4A1A78F65193`
- `git bundle verify`：通过，包含整改前 `main` 和两个旧标签的完整历史

该 bundle 与全部版本化实验存档均受 `.gitignore` 保护，不得提交 Git、上传 GitHub Release 或放入公开存储。实验报告仍保存在 `docs/experimental_docs/`，完整结果和回退归档仍保存在本机 `results/releases/`。

## 6. 残余风险与协作要求

历史内容曾经公开，因此“远端引用已删除”不能证明第三方从未下载或复制；Release 下载计数为 0 也不能作为绝对证明。GitHub 还可能在一段时间内保留不可达对象或缓存。

必须执行以下协作动作：

1. `thesmartzzz` 与 `Felix-sun-ghub` 停止从旧本地仓库 push，并重新克隆 Private 仓库。
2. 协作者不得把旧标签、旧分支或旧 bundle 推回 GitHub。
3. 项目负责人应根据赛事保密条款判断是否需要向主办方报备此前的短暂公开暴露；如需彻底清除 GitHub 缓存对象，可联系 GitHub Support。
4. 后续版本继续采用“Git 保存代码/配置/文档/聚合指标，本机受控存储保存完整实验数据”的双层回退方式。

## 7. 下一阶段准入条件

Phase 1 可以继续训练数据的可复现清洗、因果特征、嵌套时间验证和短周期预测，但必须满足：

- 评分测试集不参与训练、调参、特征选择、模型选择或指标计算；
- 任何时间特征和滞后特征只使用参考时刻及之前的信息；
- 先用训练期内的时间外层验证及合成输入完成端到端提交链测试；
- 真实评分推理必须单独授权，只读执行，并只生成符合官方格式的 `result.csv`；
- 所有实验报告进入 `docs/experimental_docs/`，所有运行证据进入 `results/`，且提交前必须通过敏感内容扫描。
