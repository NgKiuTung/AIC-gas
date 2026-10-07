"""私有赛事研究：直接读dataset.zip，16表原生粒度合并，不解压执行任何代码。"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
import hashlib
import csv
import zipfile
import logging

import numpy as np
import pandas as pd

from gasstage.common import message, ROOT, load_json, save_npz, sha256, write_json
from gasstage.quality import numeric_frame, collapse_duplicates

LOGGER = logging.getLogger(__name__)


@dataclass
class NativeDataset:
    """观测值与来源分开；process接口永不包含两个真实负荷目标。"""
    values: pd.DataFrame
    flags: pd.DataFrame
    sources: pd.DataFrame
    target_names: tuple[str, str]

    @property
    def process(self) -> pd.DataFrame:
        """返回只含过程量的视图，防止默认把负荷当作普通输入。"""
        return self.values.drop(columns=list(self.target_names))

    @property
    def targets(self) -> pd.DataFrame:
        """目标只供历史拟合和评分使用，验证预测接口不接收全量目标。"""
        return self.values.loc[:, list(self.target_names)]

    @property
    def coarse(self) -> np.ndarray:
        """按数据来源识别15分钟原始记录，不从数值变化反推采样频率。"""
        return ((self.sources.max(axis=1).to_numpy() & 3) != 0)

    def save(self, path: Path) -> None:
        """原子导出浮点64观测及uint8来源/质量掩码，不写pickle。"""
        save_npz(path, time=self.values.index.to_numpy(dtype='datetime64[ns]'),
                 values=self.values.to_numpy(), flags=self.flags.to_numpy(dtype='uint8'),
                 columns=self.values.columns.to_numpy(dtype=str), sources=self.sources.to_numpy(dtype='uint8'),
                 source_columns=self.sources.columns.to_numpy(dtype=str), targets=np.array(self.target_names))

    @classmethod
    def load(cls, path: Path) -> NativeDataset:
        """恢复自有无pickle缓存；schema与索引仍在主入口检查。"""
        with np.load(path, allow_pickle=False) as x:
            idx = pd.DatetimeIndex(x['time'], name='datetime')
            return cls(pd.DataFrame(x['values'], idx, x['columns']),
                       pd.DataFrame(x['flags'], idx, x['columns']),
                       pd.DataFrame(x['sources'], idx, x['source_columns']), tuple(x['targets']))


def _member_map(z: zipfile.ZipFile) -> dict[str, str]:
    result = {}
    for item in z.infolist():
        path = PurePosixPath(item.filename)
        if path.is_absolute() or '..' in path.parts or item.file_size > 200_000_000:
            raise ValueError(message('ingest.error.07'))
        if item.is_dir():
            continue
        if path.name in result:
            raise ValueError(f'ambiguous member basename: {path.name}')
        result[path.name] = item.filename
    return result


def _read_table(data: bytes) -> pd.DataFrame:
    header = next(csv.reader([data.decode('utf-8-sig').splitlines()[0]]))
    if len(header) != len(set(header)):
        raise ValueError(message('ingest.error.01'))
    raw = pd.read_csv(BytesIO(data), dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if 'datetime' not in raw or raw.columns.duplicated().any():
        raise ValueError(message('ingest.error.02'))
    idx = pd.to_datetime(raw.pop('datetime'), format='%Y-%m-%d %H:%M:%S', errors='raise')
    if idx.dt.tz is not None or (idx.dt.second != 0).any():
        raise ValueError(message('ingest.error.03'))
    raw.index = pd.DatetimeIndex(idx, name='datetime')
    return raw


def _combine(kind: str, parts: list, source_parts: list, flag_parts: list) -> tuple:
    joined = pd.concat(parts).sort_index(kind='stable')
    values, duplicate = collapse_duplicates(joined)
    flags = pd.concat(flag_parts).loc[lambda x: ~x.index.duplicated(keep='first')].sort_index()
    # NOTE: 重叠仅需按时间对位掩码求OR；向量化避免百万次Python组回调。
    source_rows = pd.concat(source_parts).sort_index(kind='stable')
    boundaries = np.flatnonzero(~source_rows.index.duplicated(keep='first'))
    masks = np.bitwise_or.reduceat(source_rows.to_numpy(dtype='uint8'), boundaries)
    sources = pd.Series(masks, index=source_rows.index[boundaries], name=source_rows.name)
    for d in duplicate:
        d['table'] = kind
    return values, flags, sources, duplicate


def read_dataset(path: Path, cfg: dict, audit_dir: Path) -> NativeDataset:
    """读取指定完整包；严格匹配16表、源日期、同类schema、工作簿哈希及隐藏目标。"""
    digest = sha256(path)
    if digest != cfg['expected_dataset_sha256']:
        raise ValueError(message('ingest.error.04'))
    audit_dir.mkdir(parents=True, exist_ok=True)
    inventory, profiles, changes, duplicates = [], [], [], []
    values_list, flags_list, source_list = [], [], []
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError(message('ingest.error.08'))
        members = _member_map(z)
        for resource in ['price', 'dictionary']:
            reference = load_json(ROOT / f'resources/{resource}.json')
            if hashlib.sha256(z.read(members[reference['source_file']])).hexdigest() != reference['source_sha256']:
                raise ValueError(f'changed {resource} workbook requires reviewed extraction')
        for kind in cfg['table_kinds']:
            parts, source_parts, flag_parts, raw_parts, schema = [], [], [], [], None
            for source in cfg['sources']:
                filename = source['prefix'] + kind + '.csv'
                payload = z.read(members[filename])
                raw = _read_table(payload)
                if schema is not None and list(raw.columns) != schema:
                    raise ValueError(f'schema mismatch: {filename}')
                schema = list(raw.columns)
                raw_parts.append(raw)
                if raw.index.min() != pd.Timestamp(source['start']) or raw.index.max() != pd.Timestamp(source['end']):
                    raise ValueError(f'source range mismatch: {filename}')
                if (raw.index.minute % source['frequency_minutes'] != 0).any():
                    raise ValueError(f'sampling grid mismatch: {filename}')
                clean, flags, changed, profile = numeric_frame(raw, cfg['quality'])
                changes.extend(dict(x, source=filename) for x in changed)
                profiles.extend(dict(x, source=filename) for x in profile)
                parts.append(clean); flag_parts.append(flags)
                source_parts.append(pd.Series(source['bit'], index=clean.index, name=kind, dtype='uint8'))
                inventory.append({'file': filename, 'member': members[filename], 'rows': len(raw),
                    'start': str(raw.index.min()), 'end': str(raw.index.max()),
                    'frequency_minutes': source['frequency_minutes'], 'sha256': hashlib.sha256(payload).hexdigest()})
            # NOTE: 清洗前也检查重复，防止两个不同的坏值都变NaN后被误认为一致。
            collapse_duplicates(pd.concat(raw_parts))
            values, flags, src, dup = _combine(kind, parts, source_parts, flag_parts)
            values_list.append(values); flags_list.append(flags); source_list.append(src); duplicates.extend(dup)
    value = pd.concat(values_list, axis=1).sort_index()
    flag = pd.concat(flags_list, axis=1).reindex(value.index).fillna(1).astype('uint8')
    source = pd.concat(source_list, axis=1).reindex(value.index).fillna(0).astype('uint8')
    result = NativeDataset(value, flag, source, tuple(cfg['targets']))
    test = result.targets.loc[cfg['protocol']['test_start']:]
    if len(test) != 14400 or test.notna().any().any():
        raise ValueError(message('ingest.error.05'))
    if sha256(path) != digest:
        raise ValueError(message('ingest.error.06'))
    pd.DataFrame(inventory).to_csv(audit_dir/'inventory.csv', index=False)
    pd.DataFrame(profiles).to_csv(audit_dir/'numeric_profile.csv', index=False)
    pd.DataFrame(changes).to_csv(audit_dir/'changes.csv', index=False)
    write_json(audit_dir/'deduplication.json', duplicates)
    write_json(audit_dir/'source.json', {'sha256':digest, 'raw_rows_after_union':len(value),
        'columns':len(value.columns), 'process_columns':len(result.process.columns), 'changes':len(changes),
        'duplicates':len(duplicates), 'test_target_observations':int(test.notna().sum().sum())})
    LOGGER.info(message('ingest.complete'), len(value), len(result.process.columns))
    return result
