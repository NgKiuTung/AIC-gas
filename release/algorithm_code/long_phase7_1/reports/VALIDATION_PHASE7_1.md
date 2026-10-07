# Phase7.1 发布验收

## 本次新增
- `lightgbm_direct_long`：LightGBM Long direct residual。
- `mlp_long_residual`：浅层 MLP Long residual，输入为 276 项因果静态特征、已知未来信息和冻结 reference58 anchor。
- MLP 最后一层零初始化，epoch0 逐值等于 reference58，可无损回退。
- 显式 `--long-only`：Phase6 Short 缺失时继续 Long 研究，不伪造 Short，不生成可提交双文件包。

## 已执行
- 205 项 pytest：全部通过。
- `python -m compileall -q gasauto gasbench tests run.py`：通过。
- CPU preflight：LightGBM、Shallow MLP、TCN、XGBoost 均通过。
- LightGBM direct 的 fit/save/load/replay 已由现有树后端自动测试覆盖。
- Shallow MLP 已覆盖 8/96 动态输出、epoch0 精确回退、实际训练、0轮 refit、TaskNeural 完整 fit/load/replay。
- Long-only 导出测试：`selected/l_result.csv` 存在；`s_result.csv` 与 `results_only.zip` 不生成；`long_only_results.zip` 生成；`submission_pair_ready=false`。

## 未宣称
- 当前容器无 NVIDIA A10/CUDA，因此没有把 CPU preflight 当作 A10 验收。
- 当前容器没有重新执行官方 dataset.zip 的 Phase7.1 真实数据 Smoke。
- “约66分 LightGBM + 浅层神经网络”没有源码、参数、提交文件和评分拆分，只作为模型族研究动机，不是复现结果。
- Phase6 官方 59.0888 对应 Short 文件当前不可得；`--long-only` 产物不是正式提交包。

## A10 下一门禁
用户实例应依次执行：包 SHA/manifest 校验 → 复用已验证 MultiModel V2 Python → `prepare` → `phase7_smoke_a10.json` preflight → `research --long-only` → `verify --deep`。Smoke 通过后才进入 `phase7_a10_3h.json`。
