"""加载六类业务的明确处理配置并展示细化及静态匹配结果。"""

import argparse
import json
import math
from pathlib import Path

from matching import match_task


INPUTS = Path(__file__).resolve().parent / "BusinessRequests" / "External"


def lfm_reference(sample_rate_mhz, bandwidth_mhz, pulse_width_us):
    count = round(sample_rate_mhz * pulse_width_us)
    duration = pulse_width_us * 1e-6
    bandwidth = bandwidth_mhz * 1e6
    samples = []
    for index in range(count):
        time = index / (sample_rate_mhz * 1e6)
        phase = 2 * math.pi * (-bandwidth * time / 2 + bandwidth * time * time / (2 * duration))
        samples.append(str(complex(round(math.cos(phase), 8), round(math.sin(phase), 8))))
    return samples


def radio_processing(profile):
    return {"profile": profile, "waveform": "ofdm", "modulation": "qpsk",
            "channel_code": "convolutional", "frame_format": "demo_crc_frame_v1",
            "crc": "crc16", "fft_length": 256, "cp_length": 32,
            "subcarrier_allocation": list(range(1, 215)), "pilot_pattern": [],
            "frame_payload_bits": 128, "frame_header_bits": 64, "crc_bits": 16,
            "fragment_payload_bytes": 16, "block_length": 208,
            "constraint_length": 7, "generators": [121, 91]}


def configured_request(name):
    raw = json.loads((INPUTS / name).read_text(encoding="utf-8"))
    if name == "01_voice_communication.json":
        raw["data_input"]["format"] = "audio/pcm"
        processing = radio_processing("codec2_ofdm_qpsk_conv")
        processing.update(codec="codec2", codec2_mode="3200", codec2_frame_ms=20,
                          rx_input_format="iq/complex", fft_length=128, cp_length=16,
                          subcarrier_allocation=list(range(1, 103)), block_length=96,
                          frame_payload_bits=64, frame_header_bits=16)
        raw["request"]["traffic"]["encoded_payload_bytes_per_frame"] = 8
    elif name == "02_data_link_transfer.json":
        raw["data_input"]["format"] = "data/message"
        processing = radio_processing("ofdm_qpsk_conv_tx")
    elif name == "03_target_search.json":
        raw["data_input"]["format"] = "iq/complex"
        processing = {"profile": "lfm_pulse_doppler_ca_cfar", "waveform": "lfm",
                      "detection_method": "ca_cfar", "input_sample_rate_mhz": 40,
                      "baseband_sample_rate_mhz": 20, "decimation": 2,
                      "nco_frequency_mhz": 0.1, "input_block_length": 16384,
                      "baseband_block_length": 8192,
                      "reference_waveform": lfm_reference(20, 10, 2),
                      "pulse_bandwidth_mhz": 10, "pulse_width_us": 2,
                      "carrier_frequency_mhz": 900, "pri_us": 500,
                      "cpi_pulses": 16, "doppler_fft_length": 16,
                      "guard_cells": 2, "reference_cells": 8, "pfa": 1e-6}
    elif name == "04_target_tracking.json":
        raw["data_input"]["format"] = "iq/complex"
        processing = {"profile": "radar_echo_gnn_ekf", "coordinate_frame": "local_enu",
                      "measurement_covariance": [[100, 0], [0, 25]],
                      "measurement_timestamp_ms": 2000, "motion_model": "constant_velocity",
                      "association_method": "gnn", "filter_type": "ekf"}
    elif name == "05_signal_interception.json":
        raw["data_input"]["format"] = "iq/wideband"
        processing = {"profile": "wideband_channelized_psd_intercept",
                      "architecture": "channelized_psd", "input_sample_rate_mhz": 4,
                      "input_block_length": 32, "channel_count": 4, "decimation": 2,
                      "selected_channel": 0, "channel_sample_rate_mhz": 2,
                      "channel_block_length": 16, "fft_length": 8, "overlap": 0.5,
                      "measurement_method": "psd_peak_occupancy"}
    elif name == "06_signal_identification.json":
        raw["data_input"]["format"] = "iq/complex"
        processing = {"profile": "iq_features_classifier", "input_sample_rate_mhz": 4,
                      "input_block_length": 1024, "filter_cutoff": 0.25,
                      "feature_set": ["spectrum", "cyclostationary"],
                      "classifier_id": "reference_classifier", "model_version": "1.0",
                      "unknown_class_policy": "reject_below_threshold"}
    else:
        raise ValueError(name)
    raw["request"]["processing"] = processing
    return raw


def configured_data_link_receive():
    raw = configured_request("02_data_link_transfer.json")
    raw["direction"] = "rx"
    raw["data_input"].update(kind="aligned_iq_stream", format="iq/complex")
    raw["request"]["processing"].update(
        profile="ofdm_qpsk_conv_rx", frame_alignment="aligned",
        channel_equalization="ideal_channel", input_block_length=288)
    return raw


def main():
    parser = argparse.ArgumentParser(description="演示业务任务细化与静态组件匹配")
    parser.add_argument("task", choices=[str(i) for i in range(1, 7)] + ["data-rx", "all"],
                        nargs="?", default="all")
    parser.add_argument("--detail", action="store_true", help="输出完整规格图和诊断 JSON")
    args = parser.parse_args()
    names = sorted(INPUTS.glob("0[1-6]_*.json"))
    jobs = [(str(index), path.stem, configured_request(path.name))
            for index, path in enumerate(names, 1)]
    jobs.append(("data-rx", "02_data_link_receive", configured_data_link_receive()))
    for key, label, raw in jobs:
        if args.task not in ("all", key):
            continue
        result = match_task(raw)
        data = result.to_dict()
        if args.detail:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            graph = data.get("refined_graph", {})
            print(f"{label}: {result.status}; profile={graph.get('profile_id')}; "
                  f"nodes={len(graph.get('nodes', []))}; bindings={len(result.bindings)}")
            print(f"  CIDs={sorted({binding.cid for binding in result.bindings.values()})}")
            print(f"  uncovered_outputs={graph.get('uncovered_outputs', [])}")
            print(f"  unresolved_codes={sorted({issue.code for issue in result.unresolved})}")


if __name__ == "__main__":
    main()
