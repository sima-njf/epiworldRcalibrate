"""Generic BiLSTM calibration engine.

This file does not know what SIR, SEIR, or SIRH mean.  It receives:
  * an epidemic-curve matrix,
  * a matrix of known parameters,
  * a matrix of target parameters.

The R side decides which columns are known and which are targets.
"""

import copy
import json
import time
import warnings
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


OUTPUT_ACTIVATIONS = ("sigmoid", "softplus")
INCIDENCE_TRANSFORMS = ("none", "log1p")


class BiLSTMModel(nn.Module):
    def __init__(self, known_dim, target_dim, hidden_size=160, num_layers=3, dropout=0.5,
                 output_activations=None):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(2 * hidden_size + known_dim, 64)
        self.fc2 = nn.Linear(64, target_dim)
        # One activation per target, applied to the min-max scaled outputs.
        # The original pretrained SIR model used sigmoid, softplus, softplus.
        if output_activations is None:
            output_activations = ["sigmoid"] * target_dim
        self.output_activations: List[str] = list(output_activations)

    def forward(self, epicurve, known):
        _, (hidden, _) = self.lstm(epicurve)
        curve_features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        features = torch.cat((curve_features, known), dim=1)
        features = torch.relu(self.fc1(features))
        out = self.fc2(features)
        columns = []
        for j, activation in enumerate(self.output_activations):
            if activation == "softplus":
                columns.append(nn.functional.softplus(out[:, j]))
            else:
                columns.append(torch.sigmoid(out[:, j]))
        return torch.stack(columns, dim=1)


def _fixed_incidence_scaler(n_days, incidence_max=100000.0):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_ = np.zeros(n_days, dtype=np.float64)
    scaler.data_max_ = np.full(n_days, incidence_max, dtype=np.float64)
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.scale_ = 1.0 / scaler.data_range_
    scaler.min_ = -scaler.data_min_ * scaler.scale_
    scaler.n_features_in_ = n_days
    scaler.n_samples_seen_ = 1
    return scaler


def _transform_incidence(x, transform):
    # "log1p": log(1 + cases), so small and large counts are both readable.
    x = np.asarray(x, dtype=np.float32)
    if transform == "log1p":
        return np.log1p(np.maximum(x, 0))
    return x


REPORT_COLUMNS = ("report_rate", "overdispersion", "weekday_sd")


def _report_cases(true, rate, overdispersion, weekday_sd):
    # Reported cases from true daily cases (torch, one curve per row), as in
    # the R simulator: mean = rate * cases * weekday factor (7 factors with
    # mean 1), negative binomial with variance mean + overdispersion * mean^2,
    # drawn as a Gamma-Poisson mixture.
    n, n_days = true.shape
    w = torch.exp(weekday_sd[:, None] * torch.randn(n, 7, device=true.device))
    w = w / w.mean(dim=1, keepdim=True)
    mu = rate[:, None] * true * w[:, torch.arange(n_days, device=true.device) % 7]
    k = (1.0 / overdispersion)[:, None].expand_as(mu)
    lam = torch.distributions.Gamma(k, torch.ones_like(k)).sample() * mu / k
    return torch.poisson(lam)


def _report_cases_np(true, rate, overdispersion, weekday_sd, rng):
    # The same with numpy, for a fixed validation draw.
    n, n_days = true.shape
    w = np.exp(weekday_sd[:, None] * rng.standard_normal((n, 7)))
    w = w / w.mean(axis=1, keepdims=True)
    mu = rate[:, None] * true * w[:, np.arange(n_days) % 7]
    k = np.broadcast_to((1.0 / overdispersion)[:, None], mu.shape)
    return rng.poisson(rng.gamma(k, mu / k)).astype(np.float32)


def _scale_incidence(x, scaler):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[1] == scaler.n_features_in_:
        return scaler.transform(x).astype(np.float32)
    raise ValueError(
        f"Epicurve has {x.shape[1]} days but the saved model expects "
        f"{scaler.n_features_in_}."
    )


def _optional_physics_loss(pred_scaled, known_scaled, target_scaler, known_scaler,
                           target_names, known_names, weight, mse):
    if weight <= 0:
        return torch.tensor(0.0, dtype=pred_scaled.dtype, device=pred_scaled.device)

    required_targets = {"ptran", "crate", "R0"}
    if not required_targets.issubset(target_names) or "recov" not in known_names:
        return torch.tensor(0.0, dtype=pred_scaled.dtype, device=pred_scaled.device)

    target_min = torch.tensor(target_scaler.data_min_, dtype=pred_scaled.dtype, device=pred_scaled.device)
    target_range = torch.tensor(target_scaler.data_range_, dtype=pred_scaled.dtype, device=pred_scaled.device)
    known_min = torch.tensor(known_scaler.data_min_, dtype=pred_scaled.dtype, device=pred_scaled.device)
    known_range = torch.tensor(known_scaler.data_range_, dtype=pred_scaled.dtype, device=pred_scaled.device)

    pred = pred_scaled * target_range + target_min
    known = known_scaled * known_range + known_min

    p = target_names.index("ptran")
    c = target_names.index("crate")
    r = target_names.index("R0")
    g = known_names.index("recov")

    return weight * mse(pred[:, r] * known[:, g], pred[:, p] * pred[:, c])


def _predict_in_chunks(model, x, known, chunk_size=1024):
    # Whole validation sets can be too large for one pass on a GPU. On a GPU
    # with little free memory, retry with smaller chunks (same predictions).
    while True:
        try:
            return torch.cat([
                model(x[i:i + chunk_size], known[i:i + chunk_size])
                for i in range(0, len(x), chunk_size)
            ])
        except torch.OutOfMemoryError:
            if chunk_size <= 16:
                raise
            torch.cuda.empty_cache()
            chunk_size //= 4


def _window_batches(n, batch_size, window_min, window_max, n_days, rng, device):
    # One epoch of random windows: every batch gets its own length, uniform in
    # [window_min, window_max], and every curve its own start, uniform over the
    # positions where the window fits.
    order = rng.permutation(n)
    for b in range(0, n, batch_size):
        idx = order[b:b + batch_size]
        length = int(rng.integers(window_min, window_max + 1))
        starts = rng.integers(0, n_days - length + 1, size=len(idx))
        yield torch.as_tensor(idx, device=device), length, starts


def _informative_days(curves, stop_zeros):
    # Number of days before the first run of `stop_zeros` consecutive zero days
    # (the whole curve if there is none): after a week without cases the
    # epidemic is taken to be over and the rest is ignored.
    curves = np.atleast_2d(curves)
    n, n_days = curves.shape
    ends = np.full(n, n_days)
    run = np.zeros(n, dtype=int)
    found = np.zeros(n, dtype=bool)
    for t in range(n_days):
        run = np.where(curves[:, t] == 0, run + 1, 0)
        hit = (run >= stop_zeros) & ~found
        ends[hit] = t + 1 - stop_zeros
        found |= hit
    return ends


def _informative_window_batches(n, batch_size, window_min, window_max, active, rng, device):
    # Like _window_batches, but windows stay inside each curve's informative
    # days: a batch's length is uniform in [window_min, window_max] (up to the
    # longest informative period), its curves are drawn from those whose
    # informative period is at least that long, and each start is uniform over
    # the positions where the window fits inside the informative days.
    longest = int(active.max())
    for _ in range(int(np.ceil(n / batch_size))):
        length = int(rng.integers(window_min, min(window_max, longest) + 1))
        pool = np.flatnonzero(active >= length)
        idx = rng.choice(pool, size=min(batch_size, len(pool)), replace=False)
        starts = (rng.random(len(idx)) * (active[idx] - length + 1)).astype(int)
        yield torch.as_tensor(idx, device=device), length, starts


def _take_windows(x, idx, length, starts):
    # x: (curves, days). Returns (len(idx), length, 1).
    pos = torch.as_tensor(starts, device=x.device)[:, None] + torch.arange(length, device=x.device)
    return x[idx[:, None], pos].unsqueeze(-1)


def train_model(theta_csv, incidence_csv, output_dir, known_names, target_names,
                model_type="generic", epochs=100, batch_size=64,
                learning_rate=0.001, seed=122, hidden_size=160,
                num_layers=3, dropout=0.5, physics_weight=0.0,
                validation_fraction=0.2, incidence_max=100000.0,
                output_activations=None, device="auto", epoch_callback=None,
                window_min=None, window_max=None, grad_clip=None,
                window_stop_zeros=None, state_csvs=None, incidence_transform="none",
                observation_noise=False, lr_schedule="none", resume=False):
    # resume: save a checkpoint after every epoch in output_dir and, if one
    # from the same settings is there, continue from it (for long runs on
    # GPUs that may fail). The checkpoint is removed when training ends.
    # observation_noise: the curves are true daily cases and the theta CSV has
    # report_rate, overdispersion and weekday_sd; every training batch gets
    # newly drawn reported cases (validation: one fixed draw), so the network
    # never sees the same noise twice. With windows, the noise is drawn for
    # each window.
    # Optional state targets: one CSV per state (curves x days), the value of
    # that state at the start of each day. With windows, the target is the
    # state at the start of the window (for example the share infectious).
    state_names = list(state_csvs) if state_csvs else []
    if state_names and window_min is None:
        raise ValueError("State targets need windows (window_min).")
    n_targets = len(target_names) + len(state_names)
    if output_activations is None:
        output_activations = ["sigmoid"] * n_targets
    output_activations = list(output_activations)
    if len(output_activations) != n_targets:
        raise ValueError(
            f"Got {len(output_activations)} output activations for "
            f"{n_targets} targets."
        )
    unknown = [a for a in output_activations if a not in OUTPUT_ACTIVATIONS]
    if unknown:
        raise ValueError(f"Unknown output activation(s): {unknown}")
    if lr_schedule not in ("none", "cosine"):
        raise ValueError(f"Unknown learning-rate schedule: {lr_schedule}")
    if incidence_transform not in INCIDENCE_TRANSFORMS:
        raise ValueError(f"Unknown incidence transform: {incidence_transform}")

    device = _resolve_device(device)
    print(f"Training on: {device}", flush=True)
    if str(device).startswith("cuda"):
        # Start CUDA before loading the data, so that a GPU that cannot be used
        # fails the job at once instead of after the data are read.
        torch.zeros(1, device=device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    theta = pd.read_csv(theta_csv)
    incidence = pd.read_csv(incidence_csv).values.astype(np.float32)
    if observation_noise:
        missing = [c for c in REPORT_COLUMNS if c not in theta.columns]
        if missing:
            raise ValueError(f"observation_noise needs theta columns: {missing}")
        report = theta[list(REPORT_COLUMNS)].values.astype(np.float32)
        true_cases = incidence.copy()
    incidence = _transform_incidence(incidence, incidence_transform)

    missing = [x for x in known_names + target_names if x not in theta.columns]
    if missing:
        raise ValueError(f"Columns not found in theta CSV: {missing}")

    known_raw = theta[known_names].values.astype(np.float32)
    targets_raw = theta[target_names].values.astype(np.float32)

    idx = np.arange(len(theta))
    train_idx, val_idx = train_test_split(
        idx, test_size=validation_fraction, random_state=42, shuffle=True
    )

    incidence_scaler = _fixed_incidence_scaler(
        incidence.shape[1], float(_transform_incidence(incidence_max, incidence_transform))
    )
    known_scaler = MinMaxScaler().fit(known_raw[train_idx])
    target_scaler = MinMaxScaler().fit(targets_raw[train_idx])
    if state_names:
        states_raw = np.stack(
            [pd.read_csv(state_csvs[nm]).values.astype(np.float32) for nm in state_names], axis=2
        )
        if states_raw.shape[:2] != incidence.shape:
            raise ValueError(f"State files must be {incidence.shape} like the curves.")
        # One scaler for all targets: the parameters' ranges and the states' ranges.
        lo = np.concatenate([targets_raw[train_idx].min(0), states_raw[train_idx].min((0, 1))])
        hi = np.concatenate([targets_raw[train_idx].max(0), states_raw[train_idx].max((0, 1))])
        target_scaler = MinMaxScaler().fit(np.vstack([lo, hi]))
        k = len(target_names)
        state_lo, state_range = target_scaler.data_min_[k:], target_scaler.data_range_[k:]
        states_scaled = ((states_raw - state_lo) / state_range).astype(np.float32)

    if observation_noise:
        val_cases = _report_cases_np(true_cases[val_idx], *report[val_idx].T,
                                     rng=np.random.default_rng(seed + 2))
        incidence[val_idx] = _transform_incidence(val_cases, incidence_transform)
    x_train = _scale_incidence(incidence[train_idx], incidence_scaler)
    x_val = _scale_incidence(incidence[val_idx], incidence_scaler)
    k_train = known_scaler.transform(known_raw[train_idx]).astype(np.float32)
    k_val = known_scaler.transform(known_raw[val_idx]).astype(np.float32)
    k = len(target_names)
    lo, rng_ = target_scaler.data_min_[:k], target_scaler.data_range_[:k]
    y_train = ((targets_raw[train_idx] - lo) / rng_).astype(np.float32)
    y_val = ((targets_raw[val_idx] - lo) / rng_).astype(np.float32)

    # Random windows of each curve instead of whole curves.
    n_days = incidence.shape[1]
    windowed = window_min is not None
    if windowed:
        window_min = int(window_min)
        window_max = int(window_max) if window_max is not None else n_days
        if not 1 <= window_min <= window_max <= n_days:
            raise ValueError(
                f"Need 1 <= window_min <= window_max <= {n_days} (the curve length)."
            )
        rng = np.random.default_rng(seed)
        # With observation noise the windows are cut from the true cases instead.
        x_train_2d = None if observation_noise else torch.tensor(x_train, dtype=torch.float32, device=device)
        k_train_t = torch.tensor(k_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        x_val_2d = torch.tensor(x_val, dtype=torch.float32, device=device)
        # Fixed validation windows: 10 lengths spread over the range, random starts.
        val_rng = np.random.default_rng(seed + 1)
        grid = np.unique(np.linspace(window_min, window_max, 10).round().astype(int))
        val_lengths = grid[np.arange(len(val_idx)) % len(grid)]
        val_starts = np.array([val_rng.integers(0, n_days - L + 1) for L in val_lengths])
        if window_stop_zeros:
            # With observation noise: training windows end where the expected
            # number of reported cases is zero for a week; validation uses its
            # fixed draw, as new curves are cut.
            train_active = _informative_days(
                np.round(report[train_idx, :1] * true_cases[train_idx]) if observation_noise
                else incidence[train_idx], window_stop_zeros)
            val_active = _informative_days(incidence[val_idx], window_stop_zeros)
            print(
                f"Informative days: median {np.median(train_active):.0f}; "
                f"{np.mean(train_active < window_min):.1%} of training curves are shorter "
                f"than {window_min} days and are not used",
                flush=True,
            )
            # Validation, like training, only uses curves with at least
            # window_min informative days (it picks the epoch that is kept).
            usable = val_active >= window_min
            if not usable.all():
                print(
                    f"Validation: {np.mean(~usable):.1%} of curves have fewer than "
                    f"{window_min} informative days and are left out",
                    flush=True,
                )
                val_idx, val_active = val_idx[usable], val_active[usable]
                x_val, k_val, y_val = x_val[usable], k_val[usable], y_val[usable]
                val_lengths = val_lengths[usable]
                x_val_2d = torch.tensor(x_val, dtype=torch.float32, device=device)
            # Validation windows inside the informative days too: the longest
            # grid length that fits, and a start where it fits.
            fits = [grid[grid <= max(a, window_min)].max() for a in val_active]
            val_lengths = np.minimum(val_lengths, fits)
            val_starts = np.array([
                val_rng.integers(0, max(a, L) - L + 1) for a, L in zip(val_active, val_lengths)
            ])
        states_train_t = None
        if state_names:
            states_train_t = torch.tensor(states_scaled[train_idx], device=device)
            # Validation targets: the states at the start of each validation window.
            y_val = np.concatenate(
                [y_val, states_scaled[val_idx][np.arange(len(val_idx)), val_starts]], axis=1
            )
    static_names = list(target_names)
    target_names = static_names + state_names

    if observation_noise:
        true_train_t = torch.tensor(true_cases[train_idx], device=device)
        report_train_t = torch.tensor(report[train_idx], device=device)
        k_train_t = torch.tensor(k_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        scale_max = float(_transform_incidence(incidence_max, incidence_transform))

        def noisy_batches():
            order = torch.randperm(len(train_idx), device=device)
            for b in range(0, len(train_idx), batch_size):
                idx = order[b:b + batch_size]
                cases = _report_cases(true_train_t[idx], *report_train_t[idx].T)
                if incidence_transform == "log1p":
                    cases = torch.log1p(cases)
                yield (cases / scale_max).unsqueeze(-1), k_train_t[idx], y_train_t[idx]

    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train[:, :, None], dtype=torch.float32),
            torch.tensor(k_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    x_val_t = torch.tensor(x_val[:, :, None], dtype=torch.float32, device=device)
    k_val_t = torch.tensor(k_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    model = BiLSTMModel(
        known_dim=len(known_names),
        target_dim=len(target_names),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_activations=output_activations,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # "cosine": the learning rate falls smoothly from its start value to 1% of
    # it over the epochs, so the last epochs make small, steady steps.
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)
        if lr_schedule == "cosine" else None
    )
    mse = nn.MSELoss()

    def predict_validation():
        if not windowed:
            return _predict_in_chunks(model, x_val_t, k_val_t)
        out = torch.empty(len(val_idx), len(target_names), device=device)
        for length in np.unique(val_lengths):
            sel = torch.as_tensor(np.where(val_lengths == length)[0], device=device)
            windows = _take_windows(x_val_2d, sel, int(length), val_starts[sel.cpu().numpy()])
            out[sel] = _predict_in_chunks(model, windows, k_val_t[sel])
        return out

    train_history = []
    validation_history = []
    validation_target_history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    checkpoint = Path(output_dir) / "checkpoint.pt"
    checkpoint_key = {
        "epochs": int(epochs), "batch_size": int(batch_size), "learning_rate": float(learning_rate),
        "seed": int(seed), "hidden_size": int(hidden_size), "num_layers": int(num_layers),
        "dropout": float(dropout), "lr_schedule": lr_schedule, "targets": list(target_names),
        "known": list(known_names), "n_train": int(len(train_idx)), "window_min": window_min,
        "window_max": window_max,
    }
    start_epoch = 0
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["key"] == checkpoint_key:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(saved["scheduler"])
            best_state = {k: v.to(device) for k, v in saved["best_state"].items()}
            best_loss, best_epoch = saved["best_loss"], saved["best_epoch"]
            train_history = saved["train_history"]
            validation_history = saved["validation_history"]
            validation_target_history = saved["validation_target_history"]
            torch.set_rng_state(saved["torch_rng"])
            if device.type == "cuda" and saved["cuda_rng"] is not None:
                torch.cuda.set_rng_state(saved["cuda_rng"])
            np.random.set_state(saved["numpy_rng"])
            if windowed:
                rng.bit_generator.state = saved["window_rng"]
            start_epoch = saved["epoch"]
            print(f"Resuming from the checkpoint after epoch {start_epoch}", flush=True)
        else:
            print("Checkpoint from other settings: starting again", flush=True)

    for epoch in range(start_epoch, epochs):
        start = time.time()
        model.train()
        total = 0.0
        seen = 0
        if windowed:
            sampler = (
                _informative_window_batches(len(train_idx), batch_size, window_min, window_max,
                                            train_active, rng, device)
                if window_stop_zeros else
                _window_batches(len(train_idx), batch_size, window_min, window_max, n_days, rng, device)
            )
            def window_targets(idx, starts):
                if states_train_t is None:
                    return y_train_t[idx]
                at_start = states_train_t[idx, torch.as_tensor(starts, device=device)]
                return torch.cat([y_train_t[idx], at_start], dim=1)
            def window_inputs(idx, length, starts):
                if not observation_noise:
                    return _take_windows(x_train_2d, idx, length, starts)
                cases = _take_windows(true_train_t, idx, length, starts).squeeze(-1)
                cases = _report_cases(cases, *report_train_t[idx].T)
                if incidence_transform == "log1p":
                    cases = torch.log1p(cases)
                return (cases / scale_max).unsqueeze(-1)
            batches = (
                (window_inputs(idx, length, starts), k_train_t[idx], window_targets(idx, starts))
                for idx, length, starts in sampler
            )
        elif observation_noise:
            batches = noisy_batches()
        else:
            batches = ((xb.to(device), kb.to(device), yb.to(device)) for xb, kb, yb in loader)
        skipped = 0
        for xb, kb, yb in batches:
            optimizer.zero_grad()
            pred = model(xb, kb)
            loss = mse(pred, yb)
            loss = loss + _optional_physics_loss(
                pred, kb, target_scaler, known_scaler,
                target_names, known_names, physics_weight, mse
            )
            if not torch.isfinite(loss):
                # A non-finite loss (a numerical fault, for example on an old GPU)
                # would turn every weight into NaN; skip the batch instead.
                skipped += 1
                continue
            loss.backward()
            # Limits the size of each update (steadies LSTMs on long sequences);
            # with no limit it only measures the gradient.
            norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip if grad_clip else float("inf"))
            if not torch.isfinite(norm):
                skipped += 1
                continue
            optimizer.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        if skipped:
            print(f"epoch {epoch + 1}: skipped {skipped} batches with a non-finite loss or gradient", flush=True)
        train_history.append(total / max(seen, 1))

        model.eval()
        with torch.no_grad():
            pred = predict_validation()
            validation_target_history.append(
                ((pred - y_val_t) ** 2).mean(dim=0).cpu().numpy()
            )
            val_loss = mse(pred, y_val_t)
            val_loss = val_loss + _optional_physics_loss(
                pred, k_val_t, target_scaler, known_scaler,
                target_names, known_names, physics_weight, mse
            )
            val_loss = float(val_loss.item())

        validation_history.append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler.step()

        print(
            f"epoch {epoch + 1}/{epochs}  train {train_history[-1]:.5f}  "
            f"validation {val_loss:.5f}  best epoch {best_epoch}  "
            f"{time.time() - start:.1f}s",
            flush=True,
        )

        if resume:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            tmp = checkpoint.with_suffix(".tmp")
            torch.save({
                "key": checkpoint_key, "epoch": epoch + 1,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_state": best_state, "best_loss": best_loss, "best_epoch": best_epoch,
                "train_history": train_history, "validation_history": validation_history,
                "validation_target_history": validation_target_history,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                "numpy_rng": np.random.get_state(),
                "window_rng": rng.bit_generator.state if windowed else None,
            }, tmp)
            tmp.replace(checkpoint)

        # Optional hook, e.g. for hyperparameter search: gets the epoch, the
        # validation MSE of each target and the total validation loss. It may
        # raise to stop training.
        if epoch_callback is not None:
            epoch_callback(
                epoch + 1,
                dict(zip(target_names, (float(v) for v in validation_target_history[-1]))),
                val_loss,
            )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_scaled = predict_validation().cpu().numpy()
    pred = target_scaler.inverse_transform(pred_scaled)
    truth = targets_raw[val_idx]
    if state_names:
        truth = np.concatenate(
            [truth, states_raw[val_idx][np.arange(len(val_idx)), val_starts]], axis=1
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save for CPU, so the model can be used without a GPU.
    model = model.to("cpu")
    scripted = torch.jit.script(model)
    config = {
        "model_type": model_type,
        "calibrator": "BiLSTM",
        "known_names": list(known_names),
        "target_names": list(target_names),
        "sequence_length": int(incidence.shape[1]),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "epochs": int(epochs),
        "best_epoch": int(best_epoch),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "physics_weight": float(physics_weight),
        "incidence_max": float(incidence_max),
        "incidence_transform": incidence_transform,
        "observation_noise": bool(observation_noise),
        "output_activations": output_activations,
        "device": str(device),
        "grad_clip": float(grad_clip) if grad_clip else None,
        "lr_schedule": lr_schedule,
        "variable_length": bool(windowed),
        "window_stop_zeros": int(window_stop_zeros) if windowed and window_stop_zeros else None,
        "window_min": int(window_min) if windowed else None,
        "window_max": int(window_max) if windowed else None,
        "state_targets": state_names,
    }

    torch.jit.save(
        scripted,
        str(output_dir / "model.pt"),
        _extra_files={"metadata.json": json.dumps(config)},
    )
    torch.save(model.state_dict(), output_dir / "weights.pt")
    joblib.dump(incidence_scaler, output_dir / "scaler_incidence.pkl")
    joblib.dump(known_scaler, output_dir / "scaler_additional.pkl")
    joblib.dump(target_scaler, output_dir / "scaler_targets.pkl")
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    metrics = []
    pred_table = {}
    if windowed:
        pred_table["window_length"] = val_lengths
        pred_table["window_start"] = val_starts
    for j, name in enumerate(target_names):
        metrics.append({
            "parameter": name,
            "MAE": float(mean_absolute_error(truth[:, j], pred[:, j])),
            "RMSE": float(np.sqrt(mean_squared_error(truth[:, j], pred[:, j]))),
            "R2": float(r2_score(truth[:, j], pred[:, j])),
        })
        pred_table[f"true_{name}"] = truth[:, j]
        pred_table[f"pred_{name}"] = pred[:, j]

    pd.DataFrame(metrics).to_csv(output_dir / "validation_metrics.csv", index=False)
    pd.DataFrame(pred_table).to_csv(output_dir / "validation_predictions.csv", index=False)
    history = pd.DataFrame({
        "epoch": np.arange(1, epochs + 1),
        "train_loss": train_history,
        "validation_loss": validation_history,
    })
    # Validation MSE of each target on the scaled 0-1 scale, without the
    # physics penalty.
    for j, name in enumerate(target_names):
        history[f"validation_mse_{name}"] = [h[j] for h in validation_target_history]
    history.to_csv(output_dir / "training_history.csv", index=False)

    if checkpoint.exists():
        checkpoint.unlink()
    print(f"Saved trained calibrator to: {output_dir}")
    print(f"Best epoch: {best_epoch}")
    print(pd.DataFrame(metrics).to_string(index=False))


def _resolve_device(device):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


# Saved models already loaded in this session, so calibrating many curves
# reads each model from disk only once. A retrained model (newer model.pt)
# is loaded again.
_LOADED_MODELS = {}


def _load_saved_model(model_dir, device):
    model_path = model_dir / "model.pt"
    key = (str(model_path.resolve()), model_path.stat().st_mtime_ns, str(device))
    if key not in _LOADED_MODELS:
        metadata = {"metadata.json": ""}
        model = torch.jit.load(
            str(model_path),
            map_location=device,
            _extra_files=metadata,
        )
        _LOADED_MODELS[key] = (
            model,
            json.loads(metadata["metadata.json"]),
            joblib.load(model_dir / "scaler_incidence.pkl"),
            joblib.load(model_dir / "scaler_additional.pkl"),
            joblib.load(model_dir / "scaler_targets.pkl"),
        )
    return _LOADED_MODELS[key]


def _prepare_variable_length(epicurve, config):
    # A curve for a model trained on windows of any length.
    curve = np.asarray(epicurve, dtype=np.float32).reshape(-1)
    if config.get("window_stop_zeros"):
        # Cut at the first week without cases, as in training (keeping at
        # least the shortest training window).
        keep = _informative_days(curve, config["window_stop_zeros"])[0]
        curve = curve[:max(keep, min(config["window_min"], len(curve)))]
    # The same fixed divisor for every day.
    transform = config.get("incidence_transform", "none")
    return _transform_incidence(curve, transform) / \
        float(_transform_incidence(config["incidence_max"], transform))


def predict_saved_model(epicurve, known, model_dir, device="auto"):
    device = _resolve_device(device)
    model, config, incidence_scaler, known_scaler, target_scaler = \
        _load_saved_model(Path(model_dir), device)
    transform = config.get("incidence_transform", "none")

    if config.get("variable_length"):
        x = _prepare_variable_length(epicurve, config).reshape(1, -1)
    else:
        x = _scale_incidence(_transform_incidence(epicurve, transform), incidence_scaler)
    known = np.asarray(known, dtype=np.float32)
    if known.ndim == 1:
        known = known[None, :]
    known = known_scaler.transform(known).astype(np.float32)

    # A loaded TorchScript LSTM keeps its weights in separate GPU tensors, so
    # cuDNN warns on every call; for one curve the extra copy is negligible.
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="RNN module weights are not part")
        pred_scaled = model(
            torch.tensor(x[:, :, None], dtype=torch.float32, device=device),
            torch.tensor(known, dtype=torch.float32, device=device),
        ).cpu().numpy()

    pred = target_scaler.inverse_transform(pred_scaled)[0]
    return dict(zip(config["target_names"], pred.tolist()))


def predict_saved_model_batch(epicurves, known, model_dir, device="auto", chunk_size=4096):
    # Many curves at once (one row of `known` each). Fixed-length models: a
    # matrix of curves. Models trained on windows of any length: curves of any
    # lengths (a list), run in groups of equal length. Returns one array per
    # target.
    device = _resolve_device(device)
    model, config, incidence_scaler, known_scaler, target_scaler = \
        _load_saved_model(Path(model_dir), device)
    known = known_scaler.transform(np.atleast_2d(np.asarray(known, dtype=np.float32)))
    known = known.astype(np.float32)

    def run(x, k):
        return model(
            torch.tensor(x[:, :, None], dtype=torch.float32, device=device),
            torch.tensor(k, dtype=torch.float32, device=device),
        ).cpu().numpy()

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="RNN module weights are not part")
        if config.get("variable_length"):
            xs = [_prepare_variable_length(c, config) for c in epicurves]
            if len(known) != len(xs):
                raise ValueError(f"Got {len(xs)} curves but {len(known)} rows of known values.")
            lengths = np.array([len(x) for x in xs])
            pred_scaled = np.empty((len(xs), len(config["target_names"])), dtype=np.float32)
            for length in np.unique(lengths):
                sel = np.flatnonzero(lengths == length)
                for i in range(0, len(sel), chunk_size):
                    s = sel[i:i + chunk_size]
                    pred_scaled[s] = run(np.stack([xs[j] for j in s]), known[s])
        else:
            x = _scale_incidence(
                _transform_incidence(epicurves, config.get("incidence_transform", "none")),
                incidence_scaler,
            )
            if len(known) != len(x):
                raise ValueError(f"Got {len(x)} curves but {len(known)} rows of known values.")
            pred_scaled = np.concatenate([
                run(x[i:i + chunk_size], known[i:i + chunk_size]) for i in range(0, len(x), chunk_size)
            ])
    pred = target_scaler.inverse_transform(pred_scaled)
    return {name: pred[:, j] for j, name in enumerate(config["target_names"])}


