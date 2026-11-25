"""Command-line wrapper for the generic BiLSTM trainer.

The reusable model/training/prediction functions live in bilstm.py.
This file only parses command-line arguments, so reticulate can safely
source bilstm.py during prediction without triggering argparse.
"""

import argparse
from bilstm import train_model


def _parse_names(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", required=True)
    curve_group = parser.add_mutually_exclusive_group(required=True)
    curve_group.add_argument("--curves")
    curve_group.add_argument("--incidence")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--known", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--type", default="generic")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=122)
    parser.add_argument("--hidden-size", type=int, default=160)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--physics-weight", type=float, default=0.0)
    parser.add_argument("--incidence-max", type=float, default=100000.0)
    parser.add_argument("--incidence-transform", default="none", choices=["none", "log1p"])
    parser.add_argument("--observation-noise", action="store_true",
                        help="curves are true cases; draw reported cases every batch")
    parser.add_argument("--output-activation", default="sigmoid")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-min", type=int, default=None)
    parser.add_argument("--window-max", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--lr-schedule", default="none", choices=["none", "cosine"])
    parser.add_argument("--resume", action="store_true",
                        help="checkpoint every epoch and continue from a checkpoint")
    parser.add_argument("--window-stop-zeros", type=int, default=None)
    parser.add_argument("--state-files", default=None,
                        help="name=path pairs separated by commas")
    args = parser.parse_args()

    target_names = _parse_names(args.targets)
    output_activations = _parse_names(args.output_activation)
    if len(output_activations) == 1:
        output_activations = output_activations * len(target_names)

    train_model(
        theta_csv=args.theta,
        incidence_csv=args.curves if args.curves is not None else args.incidence,
        output_dir=args.output_dir,
        known_names=_parse_names(args.known),
        target_names=target_names,
        model_type=args.type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        physics_weight=args.physics_weight,
        incidence_max=args.incidence_max,
        incidence_transform=args.incidence_transform,
        observation_noise=args.observation_noise,
        output_activations=output_activations,
        device=args.device,
        window_min=args.window_min,
        window_max=args.window_max,
        grad_clip=args.grad_clip,
        lr_schedule=args.lr_schedule,
        resume=args.resume,
        window_stop_zeros=args.window_stop_zeros,
        state_csvs=dict(x.split("=", 1) for x in _parse_names(args.state_files))
        if args.state_files else None,
    )


if __name__ == "__main__":
    main()
