# 版本存档与回退

版本采用 `v主版本.次版本.修订版本[-说明]`。每个里程碑必须同时具备：

- Git 提交与净化后的标签，用于代码、配置和文档回退。
- 本机 `results/releases/<version>/` 不可覆盖快照，包含指标、配置、图、详细产物和 SHA-256 清单。
- `results/releases/current/version.json` 指向当前本地推荐版本。

从已创建的 Git 标签生成本地私有快照：

```powershell
python codefiles\tools\create_private_tag_archive.py --version v0.4.0-model --set-current
python codefiles\tools\verify_release_snapshot.py --version v0.4.0-model
```

脚本遇到同名目录会拒绝执行，防止覆盖历史。每个快照包含对应标签的仓库 ZIP，以及该阶段明确列入白名单的本地私有实验产物；模型阶段额外包含生产模型和训练日志。整个 `results/releases/` 只在本地保留，不进入 Git、GitHub Release 或公开托管。若以后需要跨机器回退，必须先确认赛事授权，再使用访问受控的私有存储。原始训练数据不复制进快照；评分集永不进入快照。
