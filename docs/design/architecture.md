# 项目架构

```text
配置 configs/
    │
    ├── 稳定模块 codefiles/src/gas_power/
    │       ├── 数据访问边界
    │       ├── 特征与预测接口
    │       ├── 评估指标
    │       └── 优化约束
    │
    ├── 稳定流水线 codefiles/pipelines/
    │       └── 调用已验证的 codefiles/legacy/ 实验脚本
    │
    └── results/
            ├── registry/ 实验登记
            └── releases/<version>/ 不可覆盖版本快照
```

`legacy/` 保留完整的实验演进和输出路径兼容性；新公共逻辑逐步抽取到 `src/gas_power/`。这允许继续迭代，同时避免一次性重写已验证的 45 个实验步骤。

