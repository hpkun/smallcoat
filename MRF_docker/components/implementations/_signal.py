"""NumPy 参考算子及参数校验；频率、滤波状态的约定集中在此处。"""

import numpy as np


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def real(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isrealobj(value):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(value) or minimum is not None and value < minimum:
        raise ValueError(f"invalid {name}")
    return value


def vector(values, name):
    data = np.asarray(values, dtype=complex)
    if data.ndim != 1 or not data.size or not np.all(np.isfinite(data)) or not np.any(data):
        raise ValueError(f"{name} must be a nonzero finite 1D vector")
    return data.copy()


def window_values(name, size):
    if name in (None, "rect"):
        return np.ones(size)
    functions = {"hann": np.hanning, "hamming": np.hamming, "blackman": np.blackman}
    if name not in functions:
        raise ValueError("unsupported window")
    result = functions[name](size)
    if not np.any(result):
        raise ValueError("window has zero energy at this length")
    return result


def lowpass(taps, cutoff):
    """cutoff 以 Nyquist 频率归一化，返回直流增益为 1 的窗化 sinc FIR。"""
    integer(taps, "taps", 3)
    if taps % 2 != 1 or not 0 < cutoff < 1:
        raise ValueError("FIR needs odd taps and 0 < cutoff < 1")
    t = np.arange(taps) - (taps - 1) / 2
    h = cutoff * np.sinc(cutoff * t) * np.hamming(taps)
    return h / np.sum(h)


def fir(data, coefficients, state):
    """因果 FIR，返回等长输出和下一块需要的历史输入。"""
    h = np.asarray(coefficients)
    history = np.zeros(len(h) - 1, dtype=complex) if state is None else state
    joined = np.concatenate((history, data))
    output = np.convolve(joined, h, mode="valid")
    retained = joined[-(len(h) - 1):].copy() if len(h) > 1 else np.empty(0, dtype=complex)
    return output, retained


def bits(data):
    if not np.all((data == 0) | (data == 1)):
        raise ValueError("bits must contain only 0 and 1")
    return data.astype(np.uint8)
