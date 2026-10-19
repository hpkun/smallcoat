"""C23：一维/二维 CA-CFAR，输出等形状检测掩码和可序列化检测点列表。"""

import numpy as np

from ...runtime import RuntimePort
from ...runtime.blocks import BlockComponent
from .._signal import integer, real


class CFARComponent(BlockComponent):
    cid = "C23"
    metadata_key = "cfar"
    requires_stream_frame = True

    def __init__(self, guard_cells=1, reference_cells=4, pfa=1e-3,
                 threshold_scale=None, edge_policy="skip", cfar_type="ca", frame_length=None):
        super().__init__(guard_cells=guard_cells, reference_cells=reference_cells, pfa=pfa,
                         threshold_scale=threshold_scale, edge_policy=edge_policy,
                         cfar_type=cfar_type, frame_length=frame_length)

    def _validate_config(self, guard_cells=1, reference_cells=4, pfa=1e-3,
                         threshold_scale=None, edge_policy="skip", cfar_type="ca", frame_length=None):
        for value, name, minimum in ((guard_cells, "guard_cells", 0), (reference_cells, "reference_cells", 1)):
            if isinstance(value, (tuple, list)):
                if len(value) != 2:
                    raise ValueError(f"{name} must be a scalar or 2-axis pair")
                for x in value:
                    integer(x, name, minimum)
            else:
                integer(value, name, minimum)
        pfa = real(pfa, "pfa")
        if not 0 < pfa < 1:
            raise ValueError("pfa must lie in (0,1)")
        if threshold_scale is not None:
            threshold_scale = real(threshold_scale, "threshold_scale", 0)
            if threshold_scale == 0:
                raise ValueError("threshold_scale must be positive")
        if edge_policy not in ("skip", "shrink") or cfar_type != "ca":
            raise ValueError("only CA-CFAR with skip/shrink edges is implemented")
        if frame_length is not None:
            integer(frame_length, "frame_length")
        return dict(guard_cells=guard_cells, reference_cells=reference_cells, pfa=pfa,
                    threshold_scale=threshold_scale, edge_policy=edge_policy,
                    cfar_type=cfar_type, frame_length=frame_length)

    def _setup(self):
        self.input_port = RuntimePort("in", "input", ("power",), ("f", "i", "u"), (1, 2), "cells")
        self.output_port = RuntimePort("out", "output", ("detections",), ("b",), (1, 2), "cells")

    def _frame_size(self):
        return self._parameters["frame_length"]

    def _validate_samples(self, samples):
        # 二维图由 process 接受，流式模式只接收 frame_length 的一维距离图。
        if not samples.size or np.any(samples < 0):
            raise ValueError("CFAR needs a nonnegative power map")
        if samples.ndim == 1:
            super()._validate_samples(samples)

    def _execute(self, samples):
        p = self._parameters
        def axes(value):
            if isinstance(value, (tuple, list)):
                if samples.ndim != 2:
                    raise ValueError("axis pairs require a 2D power map")
                return tuple(value)
            return (value,) * samples.ndim
        guards, refs = axes(p["guard_cells"]), axes(p["reference_cells"])
        radius = tuple(g + r for g, r in zip(guards, refs))
        mask = np.zeros(samples.shape, dtype=bool)
        thresholds = np.zeros(samples.shape)
        detections, tested = [], 0
        for index in np.ndindex(samples.shape):
            if p["edge_policy"] == "skip" and any(i < r or i + r >= n for i, r, n in zip(index, radius, samples.shape)):
                continue
            slices = tuple(slice(max(0, i - r), min(n, i + r + 1)) for i, r, n in zip(index, radius, samples.shape))
            patch = samples[slices]
            selector = np.ones(patch.shape, dtype=bool)
            inner = tuple(slice(max(0, i - g) - s.start, min(n, i + g + 1) - s.start)
                          for i, g, n, s in zip(index, guards, samples.shape, slices))
            selector[inner] = False
            training = patch[selector]
            if not training.size:
                continue
            noise = float(np.mean(training))
            alpha = p["threshold_scale"]
            if alpha is None:
                alpha = len(training) * np.expm1(-np.log(p["pfa"]) / len(training))
            threshold = float(alpha * noise)
            thresholds[index] = threshold
            tested += 1
            if samples[index] > threshold:
                mask[index] = True
                power = float(samples[index])
                detections.append(dict(index=list(index), power=power, threshold=threshold,
                                       noise_power=noise, snr_linear=power / noise if noise > 0 else None))
        self._business = dict(cfar_type="ca", map_shape=list(samples.shape), tested_cells=tested,
                              detection_count=len(detections), detections=detections,
                              pfa=p["pfa"], edge_policy=p["edge_policy"],
                              threshold_workspace_bytes=thresholds.nbytes)
        return mask

    # 检测点坐标使用原图索引；物理坐标由上游距离/多普勒轴解释。


CFARDetectorComponent = CFARComponent
