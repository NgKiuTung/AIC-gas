# 复现指南

先将比赛方训练文件按原名放入本地 `dataset/初赛-数据集/`。不要移动或调用评分集目录。

```powershell
python codefiles\pipelines\run_preprocessing.py
python codefiles\pipelines\run_training.py
python codefiles\pipelines\run_optimization.py
python codefiles\pipelines\run_validation.py
```

完整训练成本较高，可先用 `--dry-run` 检查顺序。所有稳定流水线会把控制台输出整理到 `results/logs/`，并在 `results/pipeline_manifests/` 记录执行清单。

