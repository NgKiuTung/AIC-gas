# 版本存档与回退

版本采用 `v主版本.次版本.修订版本[-说明]`。每个里程碑必须同时具备：

- Git 提交与净化后的标签，用于代码、配置和文档回退。
- 本机 `results/releases/<version>/` 不可覆盖快照，包含指标、配置、图、详细产物和 SHA-256 清单。
- `results/releases/current/version.json` 指向当前本地推荐版本。

创建快照：

```powershell
python codefiles\tools\create_release_snapshot.py --version v0.1.0-preliminary --set-current
```

脚本遇到同名目录会拒绝执行，防止覆盖历史。整个 `results/releases/` 只在本地保留，不进入 Git、GitHub Release 或公开托管。若以后需要跨机器回退，必须先确认赛事授权，再使用访问受控的私有存储。原始训练数据只保留一份，不复制进快照；评分集永不进入快照。
