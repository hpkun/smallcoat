from __future__ import annotations

import argparse
import json
from math import isclose
from pathlib import Path
from typing import Any

import torch

from audit_replica import load_checkpoint
from drl_ra.trace import input_sequence_digest, load_trace
from drl_ra.violation_audit import ALL_CATEGORIES, REQUESTED_CATEGORIES
from replay_trace import METRICS_ONLY_FIELDS, file_digest, run_replay


def report_markdown(result: dict[str, Any]) -> str:
    summary = result["violation_breakdown"]
    lines = [
        "# V2 reliability violation breakdown", "",
        f"{summary['violation_tasks']}/{summary['tasks']} tasks violated reliability (CVR={summary['cvr_pct']:.2f}%).",
        f"Original comparison verified: {result['reference_comparison_verified']}.",
        "Reward oracle was not run. Feasibility uses the decision snapshot before task execution.", "",
        "| Category | Tasks | % of all tasks | % of violations |",
        "|---|---:|---:|---:|",
    ]
    for category in ALL_CATEGORIES:
        if category not in REQUESTED_CATEGORIES and summary["counts"][category] == 0:
            continue
        lines.append(f"| {category} | {summary['counts'][category]} | {summary['pct_all_tasks'][category]:.2f} | {summary['pct_violations'][category]:.2f} |")
    lines += ["", f"Requested three categories cover all violations: {summary['requested_three_cover_all']}."]
    if not summary["requested_three_cover_all"]:
        lines.append("max3_primary_bottleneck means a global feasible set can meet the requirement, but no set of at most three nodes containing the current primary can do so.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose V2 violations on a saved trace without a reward oracle.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--comparison", required=True, help="Original metrics_comparison.json to verify the exact run.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    output = Path(args.output_dir)
    for name in ("violation_breakdown.json", "violation_breakdown.md"):
        if (output / name).exists():
            raise FileExistsError(f"{output / name} exists; choose a new output directory")
    torch.set_num_threads(args.torch_threads)
    trace = load_trace(args.trace)
    _, metadata = load_checkpoint(args.checkpoint, args.device)
    if metadata.get("method") != "drl-ra-learned-replica":
        raise ValueError("expected a V2 learned replica checkpoint")
    reference: dict[str, Any] = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    original = reference["algorithms"]["drl-ra-learned-replica"]
    expected = {
        "checkpoint_sha256": file_digest(args.checkpoint),
        "trace_sha256": trace["sha256"], "input_sequence_sha256": input_sequence_digest(trace["entries"]),
        "steps": trace["steps"], "seed": trace["seed"],
    }
    if any(original.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint or trace differs from the original comparison")
    result = run_replay(trace, args.checkpoint, args.device, expected_method="drl-ra-learned-replica",
                        progress=True, metrics_only=True, violation_breakdown=True)
    if any(not isclose(result["summary"][field], original["summary"][field], rel_tol=0.0, abs_tol=1e-10)
           for field in METRICS_ONLY_FIELDS):
        raise RuntimeError("replayed metrics differ from the original comparison; refusing to classify a different run")
    result["reference_comparison"] = args.comparison
    result["reference_comparison_verified"] = True
    result["reward_oracle_run"] = False
    result["records"] = [row for row in result["records"] if row["violation"]]
    result["category_task_ids"] = {
        category: [row["task_id"] for row in result["records"] if row["violation_category"] == category]
        for category in ALL_CATEGORIES
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "violation_breakdown.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / "violation_breakdown.md").open("x", encoding="utf-8") as stream:
        stream.write(report_markdown(result))
    print(report_markdown(result))
    print(f"Saved to {output.resolve()}")


if __name__ == "__main__":
    main()
