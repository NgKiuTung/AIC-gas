# AIC-Gas Phase7.1 Long-Focused Autoresearch — LightGBM + Shallow MLP

本版本以两条**官方已知最好组件**为冻结基准：

- Short：Phase6 `xgboost-process-proxy`，官方 short accuracy 90.48%，short score 30.9525/50。
- Long：原 `reference_58`，官方 long accuracy 84.58%，long score 28.7356/50。

如果平台继续按两张表独立评分后相加，则“Phase6 Short + reference58 Long”的已知组件和为 **59.6881**。该数值是已知官方组件分数之和，不是本版本新的官方提交成绩。

## Phase7.1 新增

用户获得了一个不可访问、不可复核的“约 66 分 LightGBM + 浅层神经网络”线索。由于拿不到源码、参数、预测文件或评分拆分，本项目**不宣称复现该 66 分方案**，只把它作为新增模型族的研究动机。

新增两条 Long 候选：

- `lightgbm_direct_long`：LightGBM 对 `reference_58` 做 96 步直接残差学习；
- `mlp_long_residual`：浅层 MLP 使用 276 项因果静态特征、已知未来日历/电价和冻结锚点做共享 horizon 残差，最后一层零初始化，epoch 0 逐值等于 incumbent。

最终仍按 `2 targets × 4 horizon bands` 独立筛选，因此 LightGBM、浅层 MLP、XGBoost、CatBoost、TCN、TiDE 可以在不同组件上分别胜出；没有确认收益就回退 `reference_58`。

Phase7.1 还新增显式 `--long-only` 模式。拿不到官方 59.0888 对应的 `s_result.csv` 时，Long 研究可以立即开始；程序不会伪造 Short，也不会把 Long-only 产物标成可提交的双文件包。

Phase7 不再平均分配算力给 Short/Long。Short 在研究启动时从你已经提交过的 `s_result.csv` 或 `results_only.zip` 复制并冻结；整个研究会话只训练 Long。Long 再拆成 2 个目标 × 4 个跨度带：

- `g1__h01_08`
- `g1__h09_24`
- `g1__h25_48`
- `g1__h49_96`
- `gall__h01_08`
- `gall__h09_24`
- `gall__h25_48`
- `gall__h49_96`

每个组件独立筛选、确认和回退。任何组件没有足够证据时继续使用 reference58 Long，不为了“有新模型”强制替换。

## Phase6 信息如何吸收

已审阅公开分支 `NgKiuTung/AIC-gas@agent/phase6-xgboost-process-proxy`。Phase6 不是纯 XGBoost：

- XGBoost 3.4.1 过程代理；
- 初赛权重 0.25，复赛权重 1；
- 复赛历史 180 天；
- `reg:absoluteerror` / `hist` / `lossguide` / 15 leaves / 220 rounds / lr 0.045；
- 约 356 项因果过程特征；
- 50% week4 + 50% day7 模板；
- 当前代理偏差按 6 小时衰减；
- Short `generator_all` 额外保留 LightGBM residual，`generator_1` residual 被历史选择关闭。

本版本新增 `proxy_wide`，使用已经验收的 276 项 Stage1 因果过程特征，复现 Phase6 的主要 XGBoost 参数，并系统比较 180/120/90 天窗口、day/week 模板比例、6/9/12 小时衰减、15/31 leaves。

## 数据和合规

- 测试真实 `generator_1` / `generator_all` 全段不可见。
- 任一起点只使用该起点及以前的过程量。
- 未来日历、电价按已有正式规则使用。
- Long 仍评价全部 96 个 15 分钟区间。
- 初赛同记录负荷只用于过程代理训练，不冒充已确认的未来区间均值标签。
- 训练残差需要参考预测时，使用过去训练截止冻结的代理，禁止最终 9 月底模型回填更早训练样本。

## A10 使用

建议新目录：

```bash
mkdir -p /mnt/workspace/AIC-Gas-Phase7
cd /mnt/workspace/AIC-Gas-Phase7
```

上传：
- `gas_phase7_1_lgb_mlp_code.zip`
- `gas_phase7_1_lgb_mlp_SHA256.txt`

Phase6 官方 59.0888 对应 Short 当前不可得时无需阻塞：直接使用 `--long-only`。如果之后从固定 GitHub 版本与同一 dataset 重建出真实 Short，再用新的运行目录加 `--short-source` 生成组合提交对。

校验、解压：

```bash
sha256sum -c gas_phase7_1_lgb_mlp_SHA256.txt
unzip -nq gas_phase7_1_lgb_mlp_code.zip
cd gas_phase7_1_lgb_mlp
python3 tools/check_package.py
```

优先复用已经验证过的 MultiModel V2 Python：

```bash
PY=/mnt/workspace/AIC-Gas-Multimodel-V2/gas_multimodel_v2/.venv/bin/python
test -x "$PY" && echo PYTHON_READY
```

若该环境不存在，再执行：

```bash
bash setup.sh
PY=.venv/bin/python
```

测试与准备：

```bash
"$PY" -m pytest -q

"$PY" run.py prepare   --dataset /mnt/workspace/AIC-Gas-Stage1/dataset.zip   --cache cache/main
```

A10 Smoke：

```bash
"$PY" run.py preflight   --config configs/phase7_smoke_a10.json   --output logs/phase7_preflight.json

# 当前拿不到 Phase6 Short 时，显式 Long-only：
"$PY" run.py research \
  --cache cache/main \
  --config configs/phase7_smoke_a10.json \
  --run runs/phase7_1_smoke_01 \
  --hours 0.5 \
  --long-only

# 如果之后恢复了真实 Phase6 Short，则新建运行目录并改用：
# --short-source /mnt/workspace/AIC-Gas-Phase7/phase6_results_only.zip
```

Smoke 验收：

```bash
"$PY" run.py verify \
  --cache cache/main \
  --run runs/phase7_1_smoke_01 \
  --dataset /mnt/workspace/AIC-Gas-Stage1/dataset.zip \
  --device cuda --deep
```

正式 3 小时上限：

```bash
mkdir -p logs

nohup "$PY" -u run.py research \
  --cache cache/main \
  --config configs/phase7_a10_3h.json \
  --run runs/phase7_1_long_01 \
  --hours 3 \
  --long-only \
  > logs/phase7_1_long_01_console.log 2>&1 &

echo $! > logs/phase7_1_long_01.pid
```

监控：

```bash
"$PY" run.py status --run runs/phase7_1_long_01
tail -f logs/phase7_1_long_01_console.log
```

正式结束后：

```bash
"$PY" run.py verify \
  --cache cache/main \
  --run runs/phase7_1_long_01 \
  --dataset /mnt/workspace/AIC-Gas-Stage1/dataset.zip \
  --device cuda --deep

"$PY" run.py pack \
  --run runs/phase7_1_long_01 \
  --output runs/phase7_1_long_01_handoff.zip
```

把 `phase7_1_long_01_handoff.zip` 发回即可。

## 结果目录

```text
runs/phase7_1_long_01/
  frozen_short/source.json
  # s_result.csv 仅在提供真实 --short-source 时存在
  screening.json
  selection_frozen.json
  confirmation.json
  selection.json
  candidate_results/
    reference_58/
    selected/
      l_result.csv
      long_only_results.zip   # --long-only 时，不是正式提交包
    phase6short_reference58long/  # 仅真实 Short 已附加时生成
```

提供真实 Short 时，`phase6short_reference58long/results_only.zip` 是已知最佳组件组合的本地构造基线，`selected/results_only.zip` 是 Phase7.1 经历史筛选确认后的组合版本。

`--long-only` 时只生成 `selected/l_result.csv` 与 `selected/long_only_results.zip`；`validation.json` 明确写入 `submission_pair_ready=false`，不得直接提交。

## 本地验收

Phase7.1 发布包已完成：

- 205 项自动测试通过；
- `python -m compileall` 通过；
- CPU preflight 真实检查通过：LightGBM、Shallow MLP、TCN、XGBoost；
- LightGBM direct 已由既有树后端 fit/reload 测试覆盖；
- Shallow MLP 已覆盖动态 8/96 步、epoch0 精确回退、实际训练/0轮refit、完整 TaskNeural fit/reload；
- Long-only 导出测试确认：不会生成伪 `s_result.csv`，不会生成 `results_only.zip`，只产生 `long_only_results.zip` 且 `submission_pair_ready=false`；
- 原 reference58 归档身份逻辑保持不变。

当前容器没有用户 A10/CUDA，也没有重新执行 Phase7.1 真实数据 Smoke。因此 GPU 与真实 dataset 全链路必须在用户实例执行 `phase7_smoke_a10.json` 后再进入正式 3 小时研究。没有新官方成绩。
