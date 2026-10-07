"""私有AIC赛事研究：TCN和TiDE-style多步网络；均不接收历史真实负荷。

TiDE-style是针对过程量黑窗任务的改编结构，不宣称论文逐层复现或复现论文指标。
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class DenseResidual(nn.Module):
    """两层密集残差块；LayerNorm不混入其他样本统计。"""

    def __init__(self, dim: int, hidden: int, out: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out))
        self.skip = nn.Linear(dim, out) if dim != out else nn.Identity()
        self.norm = nn.LayerNorm(out)

    def forward(self, x):
        """最后一维为特征，保留batch和时间维度。"""
        return self.norm(self.net(x) + self.skip(x))

class CausalBlock(nn.Module):
    """膨胀卷积只左填充；时间步不能读取同序列右侧值。"""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """输入N×C×T；归一化只跨通道，因果性可逐前缀测试。"""
        a = self.conv1(F.pad(x, (self.pad, 0)))
        a = self.drop(F.gelu(self.norm1(a.transpose(1, 2)).transpose(1, 2)))
        a = self.conv2(F.pad(a, (self.pad, 0)))
        a = self.drop(F.gelu(self.norm2(a.transpose(1, 2)).transpose(1, 2)))
        return x + a

class TCNForecaster(nn.Module):
    """历史过程TCN编码＋未来已知信息解码，一次输出96×2相对周期修正。"""

    def __init__(self, seq_dim: int, static_dim: int, length: int, cfg: dict):
        """NOTE: 初始输出恰为周期基线，复杂模型只有学到有效残差才偏离它。"""
        super().__init__()
        width = cfg.get('hidden', 64)
        drop = cfg.get('dropout', 0.1)
        self.project = nn.Conv1d(seq_dim, width, 1)
        self.blocks = nn.Sequential(*[CausalBlock(width, 2 ** i, drop) for i in range(cfg.get('levels', 6))])
        self.static = DenseResidual(static_dim, width * 2, width, drop)
        self.decoder = nn.Sequential(nn.Linear(width * 3 + 8, width * 2), nn.GELU(), nn.Dropout(drop), nn.Linear(width * 2, 2))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, sequence, static, future, anchor):
        """anchor已按训练目标尺度归一化；只有冻结历史周期，没有未来真负荷。"""
        a = self.blocks(self.project(sequence.transpose(1, 2)))
        context = torch.cat([a[:, :, -1], a.mean(dim=-1), self.static(static)], dim=-1)
        context = context[:, None].expand(-1, 96, -1)
        return anchor + self.decoder(torch.cat([context, future, anchor], dim=-1))

class TiDEStyleForecaster(nn.Module):
    """历史/未来协变量投影、密集编码解码、逐区间时间解码和周期残差跳连。"""

    def __init__(self, seq_dim: int, static_dim: int, length: int, cfg: dict):
        super().__init__()
        width, projected = (cfg.get('hidden', 128), cfg.get('projection', 8))
        drop = cfg.get('dropout', 0.1)
        self.project_history = DenseResidual(seq_dim, 64, projected, drop)
        self.project_future = DenseResidual(6, 32, projected, drop)
        size = length * projected + 96 * projected + static_dim
        self.encoder = nn.Sequential(DenseResidual(size, width, width, drop), DenseResidual(width, width, width, drop))
        self.decode = nn.Linear(width, 96 * projected)
        self.temporal = nn.Sequential(nn.Linear(projected + 8, width), nn.GELU(), nn.Dropout(drop), nn.Linear(width, 2))
        self.skip = nn.Linear(static_dim, 192)
        nn.init.zeros_(self.temporal[-1].weight)
        nn.init.zeros_(self.temporal[-1].bias)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def forward(self, sequence, static, future, anchor):
        """改编点：原始历史负荷被完全移除，以过程序列和冻结周模板替代。"""
        encoded = self.encoder(torch.cat([self.project_history(sequence).flatten(1), self.project_future(future).flatten(1), static], dim=-1))
        decoded = self.decode(encoded).reshape(len(sequence), 96, -1)
        residual = self.temporal(torch.cat([decoded, future, anchor], dim=-1))
        return anchor + residual + self.skip(static).reshape(-1, 96, 2)

def build_network(family: str, seq_dim: int, static_dim: int, length: int, cfg: dict) -> nn.Module:
    """两个模型使用相同输入契约，不接受未实现的模型名。"""
    classes = {'tcn': TCNForecaster, 'tide_style': TiDEStyleForecaster}
    return classes[family](seq_dim, static_dim, length, cfg)

def masked_mape(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """旧等权MAPE，仅用于诊断和与历史结果对照。"""
    finite = torch.isfinite(truth)
    safe = torch.where(finite, truth, torch.ones_like(truth))
    error = torch.abs(pred.float() - safe.float()) / safe.float()
    total = torch.zeros((), device=pred.device)
    for target in range(2):
        for end in (8, 96):
            valid = finite[:, :end, target]
            total = total + (error[:, :end, target] * valid).sum() / valid.sum().clamp_min(1) / 4
    return total


def masked_competition_mape(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """按已观测平台斜率训练：每目标短任务1份、长任务2份。"""
    finite = torch.isfinite(truth)
    safe = torch.where(finite, truth, torch.ones_like(truth))
    error = torch.abs(pred.float() - safe.float()) / safe.float()
    total = torch.zeros((), device=pred.device)
    for target in range(2):
        short = finite[:, :8, target]
        long = finite[:, :, target]
        short_loss = (error[:, :8, target] * short).sum() / short.sum().clamp_min(1)
        long_loss = (error[:, :, target] * long).sum() / long.sum().clamp_min(1)
        total = total + short_loss / 6 + long_loss / 3
    return total
