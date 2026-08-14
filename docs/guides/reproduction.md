# 复现指南

先将比赛方训练文件按原名放入本地 `dataset/初赛-数据集/`。训练、验证、特征选择和调参流程不得移动、读取或调用评分集目录。

```powershell
python codefiles\pipelines\run_preprocessing.py
python codefiles\pipelines\run_training.py
python codefiles\pipelines\run_optimization.py
python codefiles\pipelines\run_validation.py
```

完整训练成本较高，可先用 `--dry-run` 检查顺序。所有稳定流水线会把控制台输出整理到 `results/logs/`，并在 `results/pipeline_manifests/` 记录执行清单。

只有在模型、特征和融合参数全部冻结，并取得项目负责人明确授权后，才允许运行最终评分只读推理：

```powershell
python codefiles\pipelines\run_final_scoring_inference.py `
  --scoring-dir "D:\26AIC\dataset\初赛-评分所用测试集" `
  --output-dir "D:\26AIC\results\submissions\final" `
  --team-name "AIC-gas" `
  --authorization-token AUTHORIZED_FINAL_SCORING_READ_ONLY
```

该入口不会训练、调参或计算标签指标；它直接读取四个预期文件，校验输入前后哈希，并将预测 ZIP 写入 Git 忽略的本地提交目录。队伍名必须与竞赛平台登记名称一致。
