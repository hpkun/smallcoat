"""Turn a small MGEN event script into a deterministic packet schedule.

This is an independent traffic workload generator. It does not open sockets or
depend on task representation, matching, scheduling, or runtime components.
"""

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = Path(__file__).parent / "TrafficScenarios" / "data_link_2s.mgn"
DEFAULT_OUTPUT = ROOT / "outputs" / "data_link_2s_packets.json"
ON = re.compile(
    r"(?P<time>\S+) ON (?P<flow>\d+) UDP SRC (?P<src>\d+) "
    r"DST (?P<host>[^/\s]+)/(?P<dst>\d+) "
    r"PERIODIC \[(?P<rate>\S+) (?P<size>\d+)\]"
    r"(?: COUNT (?P<count>\d+))?\Z"
)
OFF = re.compile(r"(?P<time>\S+) OFF (?P<flow>\d+)\Z")
LISTEN = re.compile(r"(?P<time>\S+) LISTEN UDP (?P<port>\d+)\Z")
IGNORE = re.compile(r"(?P<time>\S+) IGNORE UDP (?P<port>\d+)\Z")


@dataclass(frozen=True)
class Flow:
    flow_id: int
    start_s: Decimal
    stop_s: Decimal
    source_port: int
    destination_host: str
    destination_port: int
    packets_per_s: Decimal
    packet_bytes: int
    count: int | None


def _seconds(value: str, line_number: int) -> Decimal:
    try:
        seconds = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"line {line_number}: invalid event time") from exc
    if not seconds.is_finite() or seconds < 0:
        raise ValueError(f"line {line_number}: event time must be finite and >= 0")
    return seconds


def _port(value: str, line_number: int) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError(f"line {line_number}: UDP port must be 1..65535")
    return port


def parse_scenario(text: str) -> tuple[tuple[Flow, ...], list[dict]]:
    """Parse the ON/OFF/LISTEN/IGNORE subset used by this example."""
    active: dict[int, dict] = {}
    flows: list[Flow] = []
    events: list[dict] = []
    listening: set[int] = set()
    last_time = Decimal(0)
    for line_number, source in enumerate(text.splitlines(), start=1):
        line = source.split("#", 1)[0].strip()
        if not line:
            continue
        match = ON.fullmatch(line)
        kind = "ON"
        if match is None:
            match = OFF.fullmatch(line)
            kind = "OFF"
        if match is None:
            match = LISTEN.fullmatch(line)
            kind = "LISTEN"
        if match is None:
            match = IGNORE.fullmatch(line)
            kind = "IGNORE"
        if match is None:
            raise ValueError(f"line {line_number}: unsupported MGEN event")
        fields = match.groupdict()
        when = _seconds(fields["time"], line_number)
        if when < last_time:
            raise ValueError(f"line {line_number}: events must be in time order")
        last_time = when
        if kind == "LISTEN":
            port = _port(fields["port"], line_number)
            if port in listening:
                raise ValueError(f"line {line_number}: port already listening")
            listening.add(port)
            events.append({"time_s": float(when), "event": kind, "port": port})
        elif kind == "IGNORE":
            port = _port(fields["port"], line_number)
            if port not in listening:
                raise ValueError(f"line {line_number}: port is not listening")
            listening.remove(port)
            events.append({"time_s": float(when), "event": kind, "port": port})
        elif kind == "ON":
            flow_id = int(fields["flow"])
            if flow_id in active:
                raise ValueError(f"line {line_number}: flow is already active")
            rate = _seconds(fields["rate"], line_number)
            size = int(fields["size"])
            count = int(fields["count"]) if fields["count"] is not None else None
            if rate == 0 or size == 0 or count == 0:
                raise ValueError(f"line {line_number}: rate, size, and count must be > 0")
            active[flow_id] = {
                "flow_id": flow_id, "start_s": when,
                "source_port": _port(fields["src"], line_number),
                "destination_host": fields["host"],
                "destination_port": _port(fields["dst"], line_number),
                "packets_per_s": rate, "packet_bytes": size, "count": count,
            }
            events.append({"time_s": float(when), "event": kind, "flow_id": flow_id})
        else:
            flow_id = int(fields["flow"])
            if flow_id not in active:
                raise ValueError(f"line {line_number}: OFF references an inactive flow")
            started = active.pop(flow_id)
            if when <= started["start_s"]:
                raise ValueError(f"line {line_number}: OFF must follow ON")
            flows.append(Flow(stop_s=when, **started))
            events.append({"time_s": float(when), "event": kind, "flow_id": flow_id})
    if active or listening:
        raise ValueError("scenario must close every flow and listening port")
    return tuple(sorted(flows, key=lambda flow: flow.flow_id)), events


def simulate(text: str) -> dict:
    flows, events = parse_scenario(text)
    packets = []
    summaries = []
    for flow in flows:
        interval = Decimal(1) / flow.packets_per_s
        index = 0
        while flow.start_s + interval * index < flow.stop_s:
            if flow.count is not None and index >= flow.count:
                break
            when = flow.start_s + interval * index
            packets.append({
                "time_s": float(when), "flow_id": flow.flow_id,
                "sequence": index + 1, "protocol": "UDP",
                "source_port": flow.source_port,
                "destination_host": flow.destination_host,
                "destination_port": flow.destination_port,
                "packet_bytes": flow.packet_bytes,
            })
            index += 1
        summaries.append({
            "flow_id": flow.flow_id, "start_s": float(flow.start_s),
            "off_event_s": float(flow.stop_s),
            "last_packet_s": float(flow.start_s + interval * (index - 1)) if index else None,
            "end_reason": "count_limit" if flow.count is not None and index == flow.count
            else "off_event",
            "packet_rate_per_s": float(flow.packets_per_s),
            "packet_bytes": flow.packet_bytes, "count_limit": flow.count,
            "packets_planned": index, "bytes_planned": index * flow.packet_bytes,
        })
    packets.sort(key=lambda packet: (packet["time_s"], packet["flow_id"], packet["sequence"]))
    return {
        "simulation": "packet_schedule_only",
        "time_unit": "seconds",
        "flow_events": events,
        "flows": summaries,
        "totals": {
            "packets_planned": len(packets),
            "bytes_planned": sum(packet["packet_bytes"] for packet in packets),
        },
        "packets": packets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate an MGEN-style data-link traffic workload")
    parser.add_argument("scenario", nargs="?", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = simulate(args.scenario.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Scenario: {args.scenario}")
    for flow in report["flows"]:
        print(f"Flow {flow['flow_id']}: first {flow['start_s']:g} s, "
              f"last {flow['last_packet_s']:g} s, "
              f"{flow['packets_planned']} packets, {flow['bytes_planned']} bytes "
              f"({flow['end_reason']})")
    print(f"Total: {report['totals']['packets_planned']} packets, "
          f"{report['totals']['bytes_planned']} bytes")
    print(f"Packet schedule: {args.output}")


if __name__ == "__main__":
    main()
