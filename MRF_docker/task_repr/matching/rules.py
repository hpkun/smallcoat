"""能力专用参数规则。说明文字留在能力表，判断使用这里的确定性代码。"""

import inspect
import json
import math

from ..component_models.runtime_registry import RUNTIME_COMPONENT_REGISTRY


class MatchRuleError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


def _require(parameters, *names):
    missing = [name for name in names if name not in parameters]
    if missing:
        raise MatchRuleError("NEEDS_PARAMETERS", f"missing required parameters: {', '.join(missing)}")


def _number(value, name, *, minimum=None, maximum=None, integer=False):
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or not math.isfinite(value):
        raise MatchRuleError("INVALID_PARAMETERS", f"{name} must be a finite {'integer' if integer else 'number'}")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise MatchRuleError("INVALID_PARAMETERS", f"{name} must be in [{minimum}, {maximum}]")
    return value


def _array(value, name, *, allow_empty=False, complex_values=False):
    if not isinstance(value, str):
        raise MatchRuleError("INVALID_PARAMETERS", f"{name} must be a JSON array encoded as a string")
    try:
        items = json.loads(value)
    except json.JSONDecodeError as exc:
        raise MatchRuleError("INVALID_PARAMETERS", f"{name} must contain valid JSON") from exc
    if not isinstance(items, list) or not (items or allow_empty):
        raise MatchRuleError("INVALID_PARAMETERS", f"{name} must be a {'possibly empty' if allow_empty else 'nonempty'} JSON array")
    for item in items:
        if type(item) in (int, float) and math.isfinite(item):
            continue
        if complex_values and isinstance(item, str):
            try:
                number = complex(item)
            except ValueError:
                pass
            else:
                if math.isfinite(number.real) and math.isfinite(number.imag):
                    continue
        raise MatchRuleError("INVALID_PARAMETERS", f"{name} must contain finite {'real or complex' if complex_values else 'real'} numbers")
    return items


def _alias(p, old, new):
    if old in p:
        if new in p and p[new] != p[old]:
            raise MatchRuleError("INVALID_PARAMETERS", f"conflicting {old} and {new}")
        p[new] = p.pop(old)


def _mhz_to_hz(p, old, new):
    if old in p:
        mhz = _number(p.pop(old), old)
        hz = mhz * 1_000_000
        if not math.isfinite(hz):
            raise MatchRuleError("INVALID_PARAMETERS", f"{old} exceeds numeric range")
        if new in p and p[new] != hz:
            raise MatchRuleError("INVALID_PARAMETERS", f"conflicting {old} and {new}")
        p[new] = hz


def adapt_parameters(task, node, cid: str) -> dict:
    """将任务标量配置转换为组件参数；不猜测缺失的功能参数。"""
    p = dict(node.parameters)
    capability = node.capability
    if cid not in RUNTIME_COMPONENT_REGISTRY:
        raise MatchRuleError("UNSUPPORTED", f"no runtime implementation for {cid}")

    # These fields describe the signal contract and are checked by the matcher.
    p.pop("sample_rate_mhz", None)
    p.pop("input_block_length", None)
    p.pop("output_block_length", None)

    if cid == "C03":
        _mhz_to_hz(p, "nco_frequency_mhz", "nco_frequency")
        _mhz_to_hz(p, "bandwidth_mhz", "bandwidth")
        p.setdefault("mode", "ddc")
        if p["mode"] != "ddc":
            raise MatchRuleError("UNSUPPORTED", "this capability requires DDC mode")
        _require(p, "decimation", "nco_frequency")
        _number(p["decimation"], "decimation", minimum=1, integer=True)
        _number(p["nco_frequency"], "nco_frequency")
    elif cid == "C05":
        expected = "fir" if capability == "fir_filter" else "iir"
        filter_type = p.get("filter_type", expected)
        if not isinstance(filter_type, str) or filter_type.lower() != expected:
            raise MatchRuleError("UNSUPPORTED", f"{capability} requires {expected} filter_type")
        p["filter_type"] = expected
        if expected == "fir":
            if "coefficients" not in p:
                _require(p, "cutoff")
        else:
            _require(p, "coefficients", "denominator")
        for key in ("coefficients", "denominator", "band_edges"):
            if key in p:
                p[key] = _array(p[key], key, complex_values=key != "band_edges")
    elif cid == "C07":
        _require(p, "preamble")
        p["preamble"] = _array(p["preamble"], "preamble", complex_values=True)
        if p.get("sync_algorithm", "data_aided") != "data_aided":
            raise MatchRuleError("UNSUPPORTED", "only known-preamble data_aided sync is supported")
    elif cid == "C08":
        _require(p, "fft_size")
        _number(p["fft_size"], "fft_size", minimum=2, integer=True)
        mode = "ifft" if capability == "ifft" else "fft"
        selected_mode = p.get("mode", mode)
        if not isinstance(selected_mode, str) or selected_mode.lower() != mode:
            raise MatchRuleError("UNSUPPORTED", f"{capability} requires {mode} mode")
        p["mode"] = mode
    elif cid == "C09":
        _require(p, "channel_count", "decimation")
        _number(p["channel_count"], "channel_count", minimum=2, integer=True)
        _number(p["decimation"], "decimation", minimum=1, integer=True)
        for key in ("channel_map", "prototype_filter"):
            if key in p:
                p[key] = _array(p[key], key, complex_values=key == "prototype_filter")
    elif cid == "C14":
        expected = "qpsk" if capability == "qpsk_demodulation" else "qam"
        modulation = p.pop("modulation", expected)
        if not isinstance(modulation, str) or modulation.lower() != expected:
            raise MatchRuleError("UNSUPPORTED", f"{capability} requires {expected} modulation")
        p["modulation"] = expected
        if p.get("mode", "demodulate") != "demodulate":
            raise MatchRuleError("UNSUPPORTED", "demodulation capability requires demodulate mode")
        p["mode"] = "demodulate"
        _alias(p, "modulation_order", "order")
        if expected == "qam":
            _require(p, "order")
        output = next(iter(node.outputs.values()), "")
        decision = "soft" if output == "data/soft_bits" else "hard" if output in ("data/bits", "data/decoded_bits") else None
        if decision is None:
            raise MatchRuleError("INTERFACE_MISMATCH", "cannot infer soft/hard decision from task output format")
        if p.get("decision_mode", decision) != decision:
            raise MatchRuleError("INVALID_PARAMETERS", "decision_mode conflicts with task output")
        p["decision_mode"] = decision
    elif cid == "C15":
        mode = "tx" if capability == "ofdm_tx" else "rx"
        if p.get("mode", mode) != mode:
            raise MatchRuleError("UNSUPPORTED", f"{capability} requires {mode} mode")
        p["mode"] = mode
        _require(p, "fft_length", "cp_length", "subcarrier_allocation", "pilot_pattern")
        for key in ("subcarrier_allocation", "pilot_pattern", "pilot_values", "guard_carriers"):
            if key in p:
                p[key] = _array(p[key], key, allow_empty=key in ("pilot_pattern", "guard_carriers"),
                                complex_values=key == "pilot_values")
    elif cid == "C17":
        if p.get("mode", "decode") != "decode" or p.get("codec_type", "convolutional") != "convolutional":
            raise MatchRuleError("UNSUPPORTED", "only convolutional decoding is supported")
        p["mode"] = "decode"
        p["codec_type"] = "convolutional"
        _require(p, "block_length", "constraint_length", "generators")
        p["generators"] = _array(p["generators"], "generators")
        input_format = next(iter(node.inputs.values()), "")
        decision = "soft" if input_format == "data/soft_bits" else "hard" if input_format in ("data/bits", "data/decoded_bits") else None
        if decision is None:
            raise MatchRuleError("INTERFACE_MISMATCH", "cannot infer FEC decision mode from task input")
        if p.get("hard_or_soft", decision) != decision:
            raise MatchRuleError("INVALID_PARAMETERS", "hard_or_soft conflicts with task input")
        p["hard_or_soft"] = decision
        if "code_rate" in p:
            if p.pop("code_rate") != "1/2" or len(p["generators"]) != 2:
                raise MatchRuleError("UNSUPPORTED", "code_rate does not match two-generator convolutional code")
    elif cid == "C21":
        waveform = p.pop("waveform", "LFM")
        if not isinstance(waveform, str) or waveform.upper() != "LFM":
            raise MatchRuleError("UNSUPPORTED", "only LFM pulse compression is mapped")
        bandwidth_mhz = p.pop("pulse_bandwidth_mhz", None)
        if bandwidth_mhz is not None:
            _number(bandwidth_mhz, "pulse_bandwidth_mhz")
            if bandwidth_mhz <= 0:
                raise MatchRuleError("INVALID_PARAMETERS", "pulse_bandwidth_mhz must be positive")
        # Bandwidth alone cannot synthesize a matched-filter reference.
        _require(p, "reference_waveform")
        p["reference_waveform"] = _array(p["reference_waveform"], "reference_waveform", complex_values=True)
        if "range_gate" in p:
            p["range_gate"] = _array(p["range_gate"], "range_gate")
    elif cid == "C23":
        method = p.pop("method", "CA-CFAR")
        if not isinstance(method, str) or method.upper() != "CA-CFAR":
            raise MatchRuleError("UNSUPPORTED", "only CA-CFAR is implemented")
        _alias(p, "false_alarm_probability", "pfa")
        _require(p, "pfa")
        _number(p["pfa"], "pfa")
        if not 0 < p["pfa"] < 1:
            raise MatchRuleError("INVALID_PARAMETERS", "pfa must lie in (0,1)")
        p["cfar_type"] = "ca"
    elif cid == "C27":
        _alias(p, "fft_size", "fft_length")
        _mhz_to_hz(p, "scan_bandwidth_mhz", "scan_bandwidth")
        _require(p, "fft_length", "overlap")
        if p.pop("scan_mode", "serial") != "serial":
            raise MatchRuleError("UNSUPPORTED", "parallel channel scan needs explicit channel selection")
    elif cid == "C32":
        _require(p, "delay_samples")

    signature = inspect.signature(RUNTIME_COMPONENT_REGISTRY[cid])
    unknown = set(p) - set(signature.parameters)
    if unknown:
        raise MatchRuleError("INVALID_PARAMETERS", f"unsupported {cid} parameters: {sorted(unknown)}")
    return p
