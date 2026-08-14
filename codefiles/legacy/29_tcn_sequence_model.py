"""GPU temporal convolution model over the latest 24 hours of causal signals."""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
CAUSAL_PATH = ROOT / "results" / "preprocessing" / "processed" / "preprocessed_train_causal.csv"
SUPERVISED_PATH = ROOT / "results" / "features" / "train_supervised_features.pkl"
RESULT_DIR = ROOT / "results" / "experiments" / "tcn_sequence"
LOG_DIR = ROOT / "results" / "preprocessing" / "logs"
HORIZONS = tuple(range(1, 9))
SEQUENCE_LENGTH = 96
EPOCHS = 32
BATCH_SIZE = 256
FOLDS = (
    ("fold_1", "2025-03-01 00:00:00", "2025-03-15 23:45:00"),
    ("fold_2", "2025-04-01 00:00:00", "2025-04-15 23:45:00"),
    ("fold_3", "2025-04-16 00:00:00", "2025-04-30 21:45:00"),
)
BETA_GRID = np.round(np.arange(0.0, 1.5001, 0.05), 2)
SEED = 20260803


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tcn_sequence_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "29_tcn_sequence_model.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, values: torch.Tensor, end_indices: np.ndarray, labels: np.ndarray):
        self.values = values
        self.end_indices = end_indices.astype(np.int64)
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        stop = int(self.end_indices[index]) + 1
        start = stop - SEQUENCE_LENGTH
        return self.values[start:stop], self.labels[index]


class CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(nn.functional.pad(x, (self.pad, 0)))
        y = self.dropout(self.activation(self.norm1(y)))
        y = self.conv2(nn.functional.pad(y, (self.pad, 0)))
        y = self.dropout(self.activation(self.norm2(y)))
        return residual + y


class TCN(nn.Module):
    def __init__(self, input_features: int, channels: int = 64):
        super().__init__()
        self.project = nn.Conv1d(input_features, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[CausalBlock(channels, dilation, dropout=0.12) for dilation in (1, 2, 4, 8, 16)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, 96), nn.GELU(),
            nn.Dropout(0.12), nn.Linear(96, 16),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        x = self.project(values.transpose(1, 2))
        x = self.blocks(x)
        return self.head(x[:, :, -1])


def labels_and_currents(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y1 = frame[[f"label_p50_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    y120 = frame[[f"label_p120_h{h}" for h in HORIZONS]].to_numpy(dtype=np.float32)
    yall = y1 + y120
    current1 = frame["feat_p50_current"].to_numpy(dtype=np.float32)[:, None]
    currentall = (frame["feat_p50_current"] + frame["feat_p120_current"]).to_numpy(dtype=np.float32)[:, None]
    relative = np.concatenate(
        [(y1 - current1) / np.maximum(np.abs(current1), 1e-6),
         (yall - currentall) / np.maximum(np.abs(currentall), 1e-6)], axis=1
    )
    return relative, y1, yall, np.concatenate([current1, currentall], axis=1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for values, _ in loader:
            values = values.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs.append(model(values).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    seed_everything(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    causal = pd.read_csv(CAUSAL_PATH, encoding="utf-8-sig", low_memory=False)
    causal["datetime"] = pd.to_datetime(causal["datetime"], errors="raise")
    supervised = pd.read_pickle(SUPERVISED_PATH)
    supervised["datetime"] = pd.to_datetime(supervised["datetime"], errors="raise")
    excluded = {"datetime", "split", "generator_1", "generator_all"}
    sequence_features = [
        c for c in causal.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(causal[c])
    ]
    if causal[sequence_features].isna().any().any():
        missing = causal[sequence_features].isna().sum()
        raise ValueError(f"Sequence features contain missing: {missing[missing > 0].to_dict()}")
    timestamp_to_index = pd.Series(causal.index.to_numpy(), index=causal["datetime"]).to_dict()
    supervised["sequence_end_index"] = supervised["datetime"].map(timestamp_to_index)
    if supervised["sequence_end_index"].isna().any():
        raise ValueError("Supervised timestamps do not map to causal sequence")

    oof_parts: list[pd.DataFrame] = []
    history_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for fold_number, (fold_name, start_text, end_text) in enumerate(FOLDS):
        seed_everything(SEED + fold_number)
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        train = supervised[supervised["datetime"] < start - pd.Timedelta(minutes=120)].copy()
        valid = supervised[(supervised["datetime"] >= start) & (supervised["datetime"] <= end)].copy()
        raw_train_rows = causal["datetime"] <= train["datetime"].max()
        scaler_source = causal.loc[raw_train_rows, sequence_features].astype(float)
        median = scaler_source.median()
        q25, q75 = scaler_source.quantile(0.25), scaler_source.quantile(0.75)
        scale = (q75 - q25).where((q75 - q25).abs() > 1e-9, scaler_source.std().clip(lower=1e-6))
        normalized = ((causal[sequence_features].astype(float) - median) / scale).clip(-12, 12)
        values_tensor = torch.from_numpy(normalized.to_numpy(dtype=np.float32)).share_memory_()
        train_labels, _, _, _ = labels_and_currents(train)
        valid_labels, y1_valid, yall_valid, currents = labels_and_currents(valid)
        train_dataset = SequenceDataset(values_tensor, train["sequence_end_index"].to_numpy(), train_labels)
        valid_dataset = SequenceDataset(values_tensor, valid["sequence_end_index"].to_numpy(), valid_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)
        model = TCN(len(sequence_features)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=8e-5)
        loss_fn = nn.SmoothL1Loss(beta=0.03)
        scaler = torch.amp.GradScaler("cuda")
        begin = time.perf_counter()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            loss_sum, count = 0.0, 0
            for values, labels in train_loader:
                values = values.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(values)
                    loss = loss_fn(output, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * len(values)
                count += len(values)
            scheduler.step()
            history_rows.append(
                {"fold": fold_name, "epoch": epoch, "train_loss": loss_sum / count,
                 "learning_rate": optimizer.param_groups[0]["lr"]}
            )
            if epoch == 1 or epoch % 8 == 0:
                LOGGER.info("%s epoch=%d loss=%.6f", fold_name, epoch, loss_sum / count)
        relative_prediction = predict(model, valid_loader, device)
        runtime = time.perf_counter() - begin
        runtime_rows.append(
            {"fold": fold_name, "train_rows": len(train), "valid_rows": len(valid),
             "sequence_length": SEQUENCE_LENGTH, "sequence_features": len(sequence_features),
             "epochs": EPOCHS, "runtime_seconds": runtime, "device": torch.cuda.get_device_name(0)}
        )
        current1, currentall = currents[:, :1], currents[:, 1:2]
        for target, truth, current, correction in (
            ("generator_1", y1_valid, current1, current1 * relative_prediction[:, :8]),
            ("generator_all", yall_valid, currentall, currentall * relative_prediction[:, 8:]),
        ):
            for index, horizon in enumerate(HORIZONS):
                oof_parts.append(
                    pd.DataFrame(
                        {"datetime": valid["datetime"].to_numpy(), "fold": fold_name,
                         "target": target, "horizon_step": horizon, "horizon_minutes": horizon * 15,
                         "actual": truth[:, index], "current": current[:, 0],
                         "raw_correction": correction[:, index]}
                    )
                )
        LOGGER.info("Completed %s runtime=%.2fs", fold_name, runtime)

    oof = pd.concat(oof_parts, ignore_index=True)
    selected_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    for target, group in oof.groupby("target"):
        h = (group["horizon_step"].to_numpy(dtype=float) - 1.0) / 7.0
        truth = group["actual"].to_numpy(dtype=float)
        current = group["current"].to_numpy(dtype=float)
        correction = group["raw_correction"].to_numpy(dtype=float)
        best: tuple[float, float, float] | None = None
        for start in BETA_GRID:
            for end in BETA_GRID:
                beta = start + (end - start) * h
                pred = np.maximum(current + beta * correction, 0.0)
                ape = np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-6)
                detail = group[["fold", "horizon_step"]].copy()
                detail["ape"] = ape
                value = float(detail.groupby(["fold", "horizon_step"])["ape"].mean().mean())
                if best is None or value < best[2]:
                    best = (float(start), float(end), value)
        assert best is not None
        start, end, value = best
        selected_rows.append({"target": target, "beta_h15": start, "beta_h120": end, "mean_mape": value})
        output = group.copy()
        output["beta"] = start + (end - start) * h
        output["prediction"] = np.maximum(current + output["beta"].to_numpy() * correction, 0.0)
        output["ape_persistence"] = np.abs(current - truth) / np.maximum(np.abs(truth), 1e-6)
        output["ape_model"] = np.abs(output["prediction"].to_numpy() - truth) / np.maximum(np.abs(truth), 1e-6)
        prediction_parts.append(output)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = predictions.groupby(["fold", "target", "horizon_step", "horizon_minutes"], as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), model_mape=("ape_model", "mean")
    )
    fold = predictions.groupby("fold", as_index=False).agg(
        persistence_mape=("ape_persistence", "mean"), model_mape=("ape_model", "mean")
    )
    overall = float(metrics["model_mape"].mean())
    persistence = float(metrics["persistence_mape"].mean())
    oof.to_csv(RESULT_DIR / "raw_oof_predictions.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULT_DIR / "selected_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULT_DIR / "metrics_by_fold_target_horizon.csv", index=False, encoding="utf-8-sig")
    fold.to_csv(RESULT_DIR / "fold_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(RESULT_DIR / "selected_parameters.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history_rows).to_csv(RESULT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runtime_rows).to_csv(RESULT_DIR / "runtime.csv", index=False, encoding="utf-8-sig")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scope": "causal preprocessed training sequences only",
        "external_scoring_data_accessed": False,
        "model": "causal TCN", "sequence_length": SEQUENCE_LENGTH,
        "sequence_features": len(sequence_features), "epochs": EPOCHS,
        "selected_parameters": selected_rows, "persistence_mape": persistence,
        "model_mape": overall, "score": 1.0 - overall,
    }
    (RESULT_DIR / "experiment_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
