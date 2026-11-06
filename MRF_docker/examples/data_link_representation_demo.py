"""Adapt an independent MGEN-style workload for task representation only.

The traffic scenario never imports task_repr. This adapter makes the mapping
explicit and does not infer a radio waveform, QoS guarantee, or component CID.
"""

import argparse
import json
from pathlib import Path

from examples.data_link_traffic_sim import DEFAULT_SCENARIO, parse_scenario
from task_repr import represent_task


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "data_link_task_representation.json"


def adapt_scenario(source: str) -> dict:
    """Map explicit flow facts to a v1 task; missing estimates stay zero."""
    flows, events = parse_scenario(source)
    if not flows:
        raise ValueError("scenario must contain at least one flow")
    horizon_ms = int(max(event["time_s"] for event in events) * 1000)
    if horizon_ms <= 0:
        raise ValueError("scenario must have a positive time horizon")
    nodes = []
    edges = []
    for flow in flows:
        source_id = f"packet_source_{flow.flow_id}"
        transmit_id = f"udp_transmit_{flow.flow_id}"
        flow_facts = {
            "flow_id": flow.flow_id,
            "start_ms": int(flow.start_s * 1000),
            "off_event_ms": int(flow.stop_s * 1000),
            "packets_per_s": float(flow.packets_per_s),
            "packet_bytes": flow.packet_bytes,
        }
        if flow.count is not None:
            flow_facts["count_limit"] = flow.count
        nodes.extend((
            {
                "id": source_id, "capability": "periodic_packet_generation",
                "inputs": {}, "outputs": {"datagram": "net/udp_datagram"},
                "estimated_runtime_ms": 0, "parameters": flow_facts,
            },
            {
                "id": transmit_id, "capability": "udp_datagram_transmission",
                "inputs": {"datagram": "net/udp_datagram"}, "outputs": {},
                "estimated_runtime_ms": 0,
                "parameters": {
                    "source_port": flow.source_port,
                    "destination_host": flow.destination_host,
                    "destination_port": flow.destination_port,
                },
            },
        ))
        edges.append({
            "source": source_id, "source_port": "datagram",
            "target": transmit_id, "target_port": "datagram",
        })
    return {
        "schema_version": 1,
        "task_id": "mgen-data-link-2s",
        "semantics": {
            "family": "CNI", "task_type": "udp_data_link_traffic",
            "scenario": "mgen_style_data_link_workload", "direction": "tx",
            "base_priority": 0,
        },
        "qos": {
            "arrival_ms": 0, "deadline_ms": horizon_ms,
            "max_latency_ms": None, "min_throughput_mbps": None,
            "min_reliability": None, "quality_constraints": [],
        },
        "resources": {
            "cpu_millicores": 0, "memory_mib": 0, "dsp_units": 0,
            "fpga_units": 0, "bandwidth_mbps": 0,
        },
        "graph": {"nodes": nodes, "edges": edges},
        "state": {
            "status": "pending", "observed_at_ms": 0,
            "progress": 0, "current_priority": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="用任务表征模块处理 MGEN 风格业务流")
    parser.add_argument("scenario", nargs="?", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw = adapt_scenario(args.scenario.read_text(encoding="utf-8"))
    represented = represent_task(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(represented.to_dict(), ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Task: {represented.task_id}")
    print(f"Horizon: {represented.Q.deadline_ms:g} ms")
    print(f"Functional DAG: {len(represented.G.nodes)} nodes, {len(represented.G.edges)} edges")
    print(f"Representation: {args.output}")


if __name__ == "__main__":
    main()
