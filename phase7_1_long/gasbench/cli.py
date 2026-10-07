"""私有AIC赛事研究：统一入口，异常留下原始栈并返回非零状态。"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import sys
import signal
from gasbench.common import logging_to, msg

def parser() -> argparse.ArgumentParser:
    """所有命令的参数由此定义，README和启动器均调用此入口。"""
    p = argparse.ArgumentParser(description='AIC跨模型训练、因果验证及提交结果导出')
    s = p.add_subparsers(dest='command', required=True)
    q = s.add_parser('prepare')
    q.add_argument('--dataset', type=Path, required=True, help=msg('help.dataset'))
    q.add_argument('--cache', type=Path, required=True, help=msg('help.cache'))
    q.add_argument('--stage1', type=Path, help=msg('help.stage1'))
    q = s.add_parser('preflight')
    q.add_argument('--config', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q = s.add_parser('train')
    q.add_argument('--cache', type=Path, required=True)
    q.add_argument('--config', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--hours', type=float, required=True, help=msg('help.hours'))
    q.add_argument('--resume', action='store_true', help=msg('help.resume'))
    q = s.add_parser('_worker')
    q.add_argument('spec', type=Path)
    q = s.add_parser('verify')
    q.add_argument('--cache', type=Path, required=True)
    q.add_argument('--run', type=Path, required=True)
    q.add_argument('--dataset', type=Path, required=True)
    q.add_argument('--deep', action='store_true')
    q.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    q = s.add_parser('predict')
    q.add_argument('--cache', type=Path, required=True)
    q.add_argument('--model', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    q = s.add_parser('pack')
    q.add_argument('--run', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    return p

def main(argv=None) -> int:
    """任务成功才输出成功状态，未完成实验不返回伪成功码。"""
    args = parser().parse_args(argv)
    logging_to()
    try:
        if args.command == 'prepare':
            from gasbench.data import prepare
            result = prepare(args.dataset, args.cache, args.stage1)
        elif args.command == 'preflight':
            from gasbench.preflight import preflight
            from gasbench.config import read_config
            result = preflight(read_config(args.config), args.output)
        elif args.command == 'train':
            from gasbench.supervisor import train

            def stop_requested(signum, frame):
                """让监督器清理本任务worker并写中断状态。"""
                raise KeyboardInterrupt(f'signal {signum}')
            signal.signal(signal.SIGTERM, stop_requested)
            result = train(args.cache, args.config, args.output, args.hours, args.resume)
        elif args.command == '_worker':
            from gasbench.worker import run_task
            result = run_task(args.spec)
        elif args.command == 'verify':
            from gasbench.verify import verify
            result = verify(args.cache, args.run, args.dataset, args.deep, args.device)
        elif args.command == 'predict':
            from gasbench.delivery import predict
            result = predict(args.cache, args.model, args.output, args.device)
        else:
            from gasbench.delivery import pack
            result = pack(args.run, args.output)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result.get('status') in ('completed', 'passed', 'baseline_output_not_optimized_final') else 2
    except KeyboardInterrupt:
        logging.warning(msg('run.interrupted'))
        return 130
    except TimeoutError:
        logging.warning(msg('job.timeout'))
        return 124
    except Exception as exc:
        logging.exception(msg('cli.error', detail=str(exc)))
        return 1
if __name__ == '__main__':
    sys.exit(main())
