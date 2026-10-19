"""C05：窗化 sinc FIR 设计与显式系数 IIR，支持跨块状态连续。"""

import numpy as np

from ...runtime import RuntimePort
from ...runtime.blocks import BlockComponent
from .._signal import fir, integer, lowpass, real, vector


class DigitalFilterComponent(BlockComponent):
    cid = "C05"
    metadata_key = "filter"

    def __init__(self, filter_type="fir", coefficients=None, denominator=None,
                 response="lowpass", cutoff=0.25, band_edges=None, order=32):
        super().__init__(filter_type=filter_type, coefficients=coefficients, denominator=denominator,
                         response=response, cutoff=cutoff, band_edges=band_edges, order=order)

    def _validate_config(self, filter_type="fir", coefficients=None, denominator=None,
                         response="lowpass", cutoff=0.25, band_edges=None, order=32):
        if filter_type not in ("fir", "iir"):
            raise ValueError("filter_type must be fir or iir")
        if response not in ("lowpass", "highpass", "bandpass", "bandstop"):
            raise ValueError("unsupported response")
        integer(order, "order", 2)
        if order % 2:
            raise ValueError("designed FIR order must be even")
        cutoff = real(cutoff, "cutoff")
        if not 0 < cutoff < 1:
            raise ValueError("cutoff uses Nyquist units and must lie in (0,1)")
        if band_edges is not None:
            if len(band_edges) != 2:
                raise ValueError("band_edges needs two normalized frequencies")
            band_edges = tuple(real(x, "band_edge") for x in band_edges)
            if not 0 < band_edges[0] < band_edges[1] < 1:
                raise ValueError("band_edges must satisfy 0 < low < high < 1")
        if coefficients is not None:
            coefficients = vector(coefficients, "coefficients")
        if filter_type == "iir":
            if coefficients is None or denominator is None:
                raise ValueError("IIR requires numerator coefficients and denominator")
            denominator = vector(denominator, "denominator")
            if denominator[0] == 0:
                raise ValueError("denominator[0] must not be zero")
            if len(denominator) > 1 and np.any(np.abs(np.roots(denominator)) >= 1):
                raise ValueError("IIR poles must lie strictly inside the unit circle")
        elif denominator is not None:
            raise ValueError("denominator applies only to IIR")
        if coefficients is None and response.startswith("band") and band_edges is None:
            raise ValueError("bandpass/bandstop design requires band_edges")
        return dict(filter_type=filter_type, coefficients=coefficients, denominator=denominator,
                    response=response, cutoff=cutoff, band_edges=band_edges, order=order)

    def _setup(self):
        p = self._parameters
        if p["coefficients"] is None:
            center = np.zeros(p["order"] + 1)
            center[p["order"] // 2] = 1
            if p["response"] == "lowpass":
                h = lowpass(len(center), p["cutoff"])
            elif p["response"] == "highpass":
                h = center - lowpass(len(center), p["cutoff"])
            else:
                lo, hi = p["band_edges"]
                band = lowpass(len(center), hi) - lowpass(len(center), lo)
                h = band if p["response"] == "bandpass" else center - band
        else:
            h = p["coefficients"]
        a = np.array([1.0]) if p["denominator"] is None else p["denominator"]
        self._b, self._a = np.asarray(h) / a[0], a / a[0]
        self.input_port = RuntimePort("in", "input", ("time",), dimensions=(1,))
        self.output_port = RuntimePort("out", "output", ("time",), dimensions=(1,))

    def _execute(self, samples):
        if self._parameters["filter_type"] == "fir":
            output, state = fir(samples, self._b, self._state.get("filter"))
        else:
            # Direct Form II Transposed；状态长度 max(len(a),len(b))-1。
            n = max(len(self._a), len(self._b))
            b, a = np.pad(self._b, (0, n - len(self._b))), np.pad(self._a, (0, n - len(self._a)))
            state = self._state.get("filter", np.zeros(n - 1, dtype=complex)).copy()
            output = np.empty(len(samples), dtype=complex)
            for i, value in enumerate(samples):
                y = b[0] * value + (state[0] if n > 1 else 0)
                if n > 1:
                    state[:-1] = state[1:] + b[1:-1] * value - a[1:-1] * y
                    state[-1] = b[-1] * value - a[-1] * y
                output[i] = y
        self._state["filter"] = state
        self._business = dict(filter_type=self._parameters["filter_type"],
                              state_bytes=state.nbytes, numerator_taps=len(self._b),
                              denominator_taps=len(self._a))
        return output

    # configure / process / push 复用公共基类，分别提供配置、独立块和连续状态接口。
