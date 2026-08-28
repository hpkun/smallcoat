from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting import plot_training_metrics_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a training-metrics summary plot from a JSON log file.",
    )
    parser.add_argument(
        "metrics_path",
        nargs="?",
        default="outputs/metrics/train_metrics.json",
        help="Path to the input JSON metrics file. Defaults to outputs/metrics/train_metrics.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output image. Defaults next to the JSON file.",
    )
    parser.add_argument(
        "--reward-window",
        type=int,
        default=10,
        help="Moving-average window for episode rewards.",
    )
    parser.add_argument(
        "--critic-window",
        type=int,
        default=100,
        help="Moving-average window for critic loss.",
    )
    parser.add_argument(
        "--critic-upper-quantile",
        type=float,
        default=0.95,
        help="Upper quantile used to clip critic loss spikes before plotting.",
    )
    return parser


def default_output_path(metrics_path: Path) -> Path:
    return Path("outputs") / "figures" / f"{metrics_path.stem}_summary.png"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    output_path = Path(args.output) if args.output else default_output_path(metrics_path)

    saved_path = plot_training_metrics_summary(
        metrics_path,
        output_path=output_path,
        reward_window=args.reward_window,
        critic_window=args.critic_window,
        critic_upper_quantile=args.critic_upper_quantile,
    )
    print(saved_path)


if __name__ == "__main__":
    main()
