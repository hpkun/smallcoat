from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting import plot_layer_arrival_rates_summary
from src.plotting import plot_layer_diagnostics_summary
from src.plotting import plot_service_metrics_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate service-metrics plots from a training JSON log file, "
            "including completion, deadline failure, capacity drop, and delay."
        ),
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
        "--layer-output",
        help="Optional path to a layer-arrival-rate plot for UAV, BS, and LEO.",
    )
    parser.add_argument(
        "--layer-diagnostics-output",
        help="Optional path to a per-layer failure/delay diagnostics plot.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Moving-average smoothing window.",
    )
    return parser


def default_output_path(metrics_path: Path) -> Path:
    return Path("outputs") / "figures" / f"{metrics_path.stem}_service_metrics.png"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    output_path = Path(args.output) if args.output else default_output_path(metrics_path)

    saved_path = plot_service_metrics_summary(
        metrics_path,
        output_path=output_path,
        smoothing_window=args.window,
    )
    print(saved_path)

    if args.layer_output:
        layer_saved_path = plot_layer_arrival_rates_summary(
            metrics_path,
            output_path=Path(args.layer_output),
            smoothing_window=args.window,
        )
        print(layer_saved_path)

    if args.layer_diagnostics_output:
        layer_diagnostics_saved_path = plot_layer_diagnostics_summary(
            metrics_path,
            output_path=Path(args.layer_diagnostics_output),
            smoothing_window=args.window,
        )
        print(layer_diagnostics_saved_path)


if __name__ == "__main__":
    main()
