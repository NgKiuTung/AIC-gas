# Phase7.1 变更
- Short incumbent 从 reference58 改为外部冻结的 Phase6 XGBoost process proxy。
- Long incumbent 继续使用 reference58。
- 新增 276-feature `proxy_wide`，吸收 Phase6 XGBoost 参数和 180d 训练策略。
- 新增 week/day template 比例和 decay_hours 的候选维度。
- Long 按 2 targets × 4 horizon bands 独立筛选、确认和拼接。
- 新增 Phase6 Short 文件冻结、哈希及恢复身份检查。
- selected 与 phase6short_reference58long 两套结果独立输出。
- 保留旧双任务研究代码与 reference58 归档；不覆盖历史运行。

- 新增 LightGBM direct Long 残差候选。
- 新增浅层 MLP Long 残差候选；零初始化保证 epoch0 精确回退 reference58。
- 新增显式 `--long-only`，Phase6 Short 缺失时允许继续 Long 研究，但不生成可提交双文件包。
- Long-only `validation.json` 固定 `submission_pair_ready=false`。
- A10 环境优先复用已验证 MultiModel V2 `.venv`，避免重复安装依赖。
- 不把不可访问的“约66分 LightGBM+浅层神经网络”线索描述为已复现方案。
