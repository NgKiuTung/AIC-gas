# Phase7.1 Long-Focused Autoresearch Program

目标不是“找到一个整体模型”，而是在冻结 Phase6 Short 后，只对 Long 的 8 个组件做有界研究。

## 冻结事实
- Phase6 Short 官方：90.48%，score 30.9525/50。
- reference58 Long 官方：84.58%，score 28.7356/50。
- 两者组件分数和：59.6881。
- Short/Long 可使用不同模型。
- 测试未来过程量不可用，测试负荷全段不可见。

## 研究对象
Long = 2 targets × 4 horizon bands：
1-8, 9-24, 25-48, 49-96。

筛选时每个组件独立计算 MAPE。候选只在组件证据足够时进入确认；确认失败就回退 reference58。最终可由不同候选负责不同目标和跨度。

## Phase7.1 新证据边界
- 外部仅有“约66分 LightGBM + 浅层神经网络”的不可复核线索。
- 无源码、参数、提交文件、分项分数或验证记录，因此不做复现声明。
- 将该线索降级为模型族假设：新增 LightGBM direct 与浅层 MLP residual，由现有历史门禁独立决定是否保留。
- 如果 Phase6 Short 文件暂时不可得，`--long-only` 只研究 Long，禁止生成伪 Short。

## 初始候选
- Phase6-style wide XGBoost proxy：180d, day/week 50/50, 6h decay。
- day weight：0 / 0.25 / 0.5。
- decay：6 / 9 / 12h。
- lookback：90 / 120 / 180d。
- leaves：15 / 31。
- XGBoost direct residual。
- CatBoost direct residual。
- LightGBM direct residual。
- Shallow MLP residual（276 causal static + known future + frozen anchor）。
- TCN long residual。
- TiDE-style 72h long residual。

每个候选均以完整十日外层窗口评分，筛选 6/7/8 月，确认 9 月两个十日窗口、三个固定 seed。

## 接受规则
筛选组件：
- 平均 MAPE gain >= 0.0002；
- 任一折退化不超过 0.003；
- 最新筛选折退化不超过 0.0015；
- 至少 2/3 折改善。

确认组件：
- 平均 MAPE gain >= 0.00015；
- 任一确认折退化不超过 0.0025；
- 最新确认折退化不超过 0.001；
- 至少 1/2 折改善；
- seed MAPE 标准差均值 <= 0.003。

这些是研发门槛，不等于官方评分器。

## 禁止
- 排行榜作为逐轮奖励；
- 用 10 月隐藏标签反推；
- 逐行人工修正测试预测；
- 因某组件改善而强迫其他组件一起替换；
- 用最终代理回填历史训练样本；
- 修改评估日期或标签定义来获得更低分。
