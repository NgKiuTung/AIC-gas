"""私有AIC赛事研究：训练计算额度单独计时，硬墙钟由监督器强制终止。"""
from contextlib import contextmanager
from pathlib import Path
import time
from gasbench.common import write_json

class TrainingClock:
    """训练调用累计不含外部评价；检查点记录实际用量，不声称用满额度。"""

    def __init__(self, seconds: float, status: Path):
        self.limit, self.elapsed, self.status = (float(seconds), 0.0, status)
        self.current = None
        self.mark('preparing')

    def mark(self, phase: str):
        """供父进程只读监控，包含当前调用的开始时刻。"""
        write_json(self.status, {'phase': phase, 'training_elapsed': self.elapsed, 'active_since': self.current, 'training_limit': self.limit, 'time': time.time()})

    def remaining(self):
        """计算剩余额度。"""
        active = time.time() - self.current if self.current else 0
        return max(0, self.limit - self.elapsed - active)

    @contextmanager
    def compute(self):
        """仅包围实际优化/训练调用；超额返回不能被标记完成。"""
        if self.remaining() <= 0:
            raise TimeoutError('training allowance exhausted')
        self.current = time.time()
        self.mark('training')
        try:
            yield
        finally:
            self.elapsed += time.time() - self.current
            self.current = None
            self.mark('preparing')
        if self.elapsed > self.limit:
            raise TimeoutError('training allowance exceeded')
