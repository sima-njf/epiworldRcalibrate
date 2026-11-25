"""Hyperparameter search for the generic BiLSTM calibrator with Optuna.

Each trial trains with bilstm.train_model() for a few epochs and is scored on
the validation MSE (0-1 scale) of the chosen targets, taking the best epoch.
Weak trials are stopped early. Several workers can share one study: run this
script once per GPU with the same --study-dir.

--export-only writes trials.csv and best_params.json from an existing study
without running new trials.
"""

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from bilstm import train_model


def _parse_names(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def _export(study, study_dir):
    study.trials_dataframe().to_csv(study_dir / "trials.csv", index=False)
    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not done:
        return
    best = study.best_trial
    with open(study_dir / "best_params.json", "w") as f:
        json.dump(
            {"trial": best.number, "score": best.value, **best.params},
            f, indent=2,
        )
        f.write("\n")
    print(f"Best trial {best.number}: score {best.value:.6f} {best.params}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", required=True)
    parser.add_argument("--curves", required=True)
    parser.add_argument("--known", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--study-name", default="bilstm")
    parser.add_argument("--score-targets", default="R0,ptran")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=122)
    parser.add_argument("--incidence-max", type=float, default=100000.0)
    parser.add_argument("--output-activation", default="sigmoid")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr-max", type=float, default=3e-3)
    parser.add_argument("--window-min", type=int, default=None)
    parser.add_argument("--window-max", type=int, default=None)
    parser.add_argument("--window-stop-zeros", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--state-files", default=None,
                        help="name=path,... state targets at the window start (as in train_bilstm.py)")
    parser.add_argument("--incidence-transform", default="none")
    parser.add_argument("--physics-weight", type=float, default=None,
                        help="fix the physics weight instead of searching it")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(study_dir / "journal.log")))
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    if args.export_only:
        _export(study, study_dir)
        return

    known_names = _parse_names(args.known)
    target_names = _parse_names(args.targets)
    score_targets = _parse_names(args.score_targets)
    state_csvs = dict(x.split("=", 1) for x in _parse_names(args.state_files)) if args.state_files else None
    output_activations = _parse_names(args.output_activation)
    if len(output_activations) == 1:
        output_activations = output_activations * (len(target_names) + len(state_csvs or {}))

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, args.lr_max, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 160, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        }
        params["physics_weight"] = (
            args.physics_weight if args.physics_weight is not None
            else trial.suggest_float("physics_weight", 1e-5, 1e-1, log=True)
        )
        best = [math.inf]

        def report(epoch, target_mse, val_loss):
            score = sum(target_mse[t] for t in score_targets) / len(score_targets)
            best[0] = min(best[0], score)
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        out_dir = tempfile.mkdtemp(prefix="trial_")
        try:
            train_model(
                theta_csv=args.theta,
                incidence_csv=args.curves,
                output_dir=out_dir,
                known_names=known_names,
                target_names=target_names,
                epochs=args.epochs,
                seed=args.seed,
                incidence_max=args.incidence_max,
                output_activations=output_activations,
                device=args.device,
                epoch_callback=report,
                window_min=args.window_min,
                window_max=args.window_max,
                window_stop_zeros=args.window_stop_zeros,
                grad_clip=args.grad_clip,
                state_csvs=state_csvs,
                incidence_transform=args.incidence_transform,
                **params,
            )
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
        return best[0]

    # A failed trial (for example out of GPU memory) is recorded and skipped.
    study.optimize(
        objective,
        timeout=args.timeout,
        catch=(RuntimeError, ValueError),
        callbacks=[MaxTrialsCallback(args.n_trials, states=(TrialState.COMPLETE, TrialState.PRUNED))],
    )
    _export(study, study_dir)


if __name__ == "__main__":
    main()
