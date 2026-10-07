"""私有AIC赛事研究：数据、设备检查、研究、验收和交接的独立入口。"""
from pathlib import Path
import argparse
import json
import signal
from gasauto.protocol import read_plan
from gasbench.common import ROOT, load_json, logging_to

def parser():
    """命令均要求明确输出或输入目录，不从旧会话猜路径。"""
    p = argparse.ArgumentParser(description='AIC-Gas短长独立有界研究')
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--dataset', type=Path, required=True)
    prep.add_argument('--cache', type=Path, required=True)
    prep.add_argument('--stage1', type=Path)
    pre = sub.add_parser('preflight')
    pre.add_argument('--config', type=Path, required=True)
    pre.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('research')
    run.add_argument('--cache', type=Path, required=True)
    run.add_argument('--config', type=Path, default=ROOT / 'configs/a10_3h.json')
    run.add_argument('--run', '--output', dest='run', type=Path, required=True)
    run.add_argument('--hours', type=float, default=3.0)
    run.add_argument('--resume', action='store_true')
    run.add_argument('--extend-hours', type=float, default=0.0)
    run.add_argument('--short-source', type=Path, help='Phase6 official-best s_result.csv or results_only.zip; omit only with --long-only')
    run.add_argument('--long-only', action='store_true', help='Run Phase7 Long research without fabricating or exporting a Short submission')
    check = sub.add_parser('verify')
    check.add_argument('--cache', type=Path, required=True)
    check.add_argument('--run', type=Path, required=True)
    check.add_argument('--dataset', type=Path, required=True)
    check.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    check.add_argument('--deep', action='store_true')
    status = sub.add_parser('status')
    status.add_argument('--run', type=Path, required=True)
    pack = sub.add_parser('pack')
    pack.add_argument('--run', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    worker = sub.add_parser('_worker')
    worker.add_argument('spec', type=Path)
    return p

def main(argv=None):
    """退出码0仅用于实际成功；异常保留完整堆栈供定位。"""
    args = parser().parse_args(argv)
    logging_to()
    if args.command == 'prepare':
        from gasbench.data import prepare
        result = prepare(args.dataset, args.cache, args.stage1)
    elif args.command == 'preflight':
        from gasbench.preflight import preflight
        result = preflight(read_plan(args.config), args.output)
    elif args.command == 'research':
        def stop(signum, frame):
            raise KeyboardInterrupt(f'收到停止信号{signum}，只清理本运行worker。')
        signal.signal(signal.SIGTERM, stop)
        cfg = read_plan(args.config)
        if cfg.get('mode') == 'phase7_long_component_focus':
            if args.short_source is None and not args.long_only:
                raise ValueError('Phase7 requires --short-source unless --long-only is explicit')
            from gasauto.phase7 import run_phase7
            result = run_phase7(args.cache, args.config, args.run, args.short_source, args.hours, args.resume, args.extend_hours, args.long_only)
        else:
            from gasauto.controller import run_research
            result = run_research(args.cache, args.config, args.run, args.hours, args.resume, args.extend_hours)
    elif args.command == 'verify':
        from gasauto.verify import verify
        result = verify(args.cache, args.run, args.dataset, args.device, args.deep)
    elif args.command == 'pack':
        from gasauto.delivery import pack
        result = pack(args.run, args.output)
    elif args.command == 'status':
        from gasbench.lifecycle import same_process
        result = load_json(args.run / 'run_status.json')
        result = {**result, 'supervisor_alive': same_process(result['identity'])}
        active = load_json(args.run / 'active_worker.json') if (args.run / 'active_worker.json').exists() else {}
        result['worker_alive'] = bool(active.get('identity') and same_process(active['identity']))
    else:
        from gasauto.worker import execute
        result = execute(args.spec)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0
