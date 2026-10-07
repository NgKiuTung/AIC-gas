"""私有AIC赛事研究：TCN、TiDE-style 与浅层 MLP 的任务特定残差编码器。"""
import torch
from torch import nn
from gasbench.networks import DenseResidual, CausalBlock

class TCN(nn.Module):
    """零初始化残差保证epoch0是明确的reference_58预测。"""

    def __init__(self, seq_dim, static_dim, length, cfg):
        super().__init__()
        self.blocks_count = cfg['blocks']
        width = cfg.get('hidden', 64)
        drop = cfg.get('dropout', 0.1)
        self.project = nn.Conv1d(seq_dim, width, 1)
        self.blocks = nn.Sequential(*[CausalBlock(width, 2 ** i, drop) for i in range(cfg.get('levels', 6))])
        self.static = DenseResidual(static_dim, 2 * width, width, drop)
        self.head = nn.Sequential(nn.Linear(width * 3 + 8, width * 2), nn.GELU(), nn.Dropout(drop), nn.Linear(width * 2, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, sequence, static, future, anchor):
        """future含确定性时间电价；anchor来自过去冻结代理，不是真实未来。"""
        a = self.blocks(self.project(sequence.transpose(1, 2)))
        context = torch.cat([a[:, :, -1], a.mean(-1), self.static(static)], -1)
        context = context[:, None].expand(-1, self.blocks_count, -1)
        return anchor + self.head(torch.cat([context, future, anchor], -1))

class TiDE(nn.Module):
    """改编密集历史/未来编码器；没有测试真实负荷输入，非原论文指标复现。"""

    def __init__(self, seq_dim, static_dim, length, cfg):
        super().__init__()
        self.h = cfg['blocks']
        p = cfg.get('projection', 8)
        width = cfg.get('hidden', 128)
        drop = cfg.get('dropout', 0.1)
        self.hist = DenseResidual(seq_dim, 64, p, drop)
        self.future = DenseResidual(6, 32, p, drop)
        self.encoder = nn.Sequential(DenseResidual(length * p + self.h * p + static_dim, width, width, drop), DenseResidual(width, width, width, drop))
        self.decode = nn.Linear(width, self.h * p)
        self.head = nn.Sequential(nn.Linear(p + 8, width), nn.GELU(), nn.Dropout(drop), nn.Linear(width, 2))
        self.skip = nn.Linear(static_dim, self.h * 2)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def forward(self, sequence, static, future, anchor):
        """任务跨度在构建时固定，短模型不存在88个不使用的输出。"""
        code = self.encoder(torch.cat([self.hist(sequence).flatten(1), self.future(future).flatten(1), static], -1))
        code = self.decode(code).reshape(len(sequence), self.h, -1)
        return anchor + self.head(torch.cat([code, future, anchor], -1)) + self.skip(static).reshape(-1, self.h, 2)

class ShallowMLP(nn.Module):
    """浅层表格网络；只修正冻结锚点，不读取未来过程或历史真实负荷。

    NOTE: 276 项静态因果特征已经包含 lag/rolling/diff。该模型故意不再
    重编码历史序列，以作为树模型之外的低复杂度归纳偏置。
    """

    def __init__(self, seq_dim, static_dim, length, cfg):
        super().__init__()
        del seq_dim, length
        self.blocks_count = cfg['blocks']
        width = cfg.get('hidden', 128)
        drop = cfg.get('dropout', 0.1)
        self.static = nn.Sequential(
            nn.Linear(static_dim, width),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(width + 8, max(32, width // 2)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(max(32, width // 2), 2),
        )
        # NOTE: epoch0 必须逐值等于 reference_58，训练失败时可无损回退。
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, sequence, static, future, anchor):
        """按每个 horizon 共享浅层网络，输出 reference_58 的残差修正。"""
        del sequence
        context = self.static(static)[:, None].expand(-1, self.blocks_count, -1)
        return anchor + self.head(torch.cat([context, future, anchor], -1))


def build(cfg, seq_dim, static_dim, length):
    """任务绑定到网络形状，未知模型立即拒绝。"""
    return {'tcn': TCN, 'tide_style': TiDE, 'mlp': ShallowMLP}[cfg['family']](seq_dim, static_dim, length, cfg)

def task_loss(pred, truth):
    """纯任务MAPE：两个目标等权，NaN真值不插补也不参与分母。"""
    valid = torch.isfinite(truth)
    if torch.any(valid & (truth <= 0)):
        raise ValueError('nonpositive truth')
    safe = torch.where(valid, truth, torch.ones_like(truth))
    error = torch.where(valid, torch.abs(pred.float() - safe.float()) / safe.float(), torch.zeros_like(pred))
    count = valid.sum(dim=(0, 1))
    if torch.any(count == 0):
        raise ValueError('empty target')
    return (error.sum(dim=(0, 1)) / count).mean()
