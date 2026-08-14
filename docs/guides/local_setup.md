# 本地环境

推荐 Python 3.10–3.12、NVIDIA RTX 4060 和匹配的 CUDA 驱动。基础依赖、GPU 依赖和开发依赖分别位于 `requirements/`。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements\base.txt -r requirements\gpu.txt -r requirements\dev.txt
pip install -e .
python codefiles\legacy\06_environment_check.py
```

环境检查输出必须保存在 `results/`。GitHub Actions 不具备竞赛数据或 GPU，只执行代码质量与合成烟雾测试。

