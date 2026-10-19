"""运行组件工厂。独立于只估算资源的 registry.py，避免混淆两类对象。"""

from .common.ddc_component import DDCComponent
from .common.digital_filter_component import DigitalFilterComponent
from .common.synchronizer_component import SynchronizerComponent
from .common.fft_component import FFTComponent
from .common.channelizer_component import ChannelizerComponent
from .cni.psk_qam_component import PSKQAMComponent
from .cni.ofdm_component import OFDMComponent
from .cni.fec_component import FECComponent
from .radar.pulse_compression_component import PulseCompressionComponent
from .radar.cfar_component import CFARComponent
from .ew.spectrum_sensing_component import SpectrumSensingComponent
from .ew.drfm_component import DRFMComponent


RUNTIME_COMPONENT_REGISTRY = {
    cls.cid: cls for cls in (
        DDCComponent, DigitalFilterComponent, SynchronizerComponent, FFTComponent,
        ChannelizerComponent, PSKQAMComponent, OFDMComponent, FECComponent,
        PulseCompressionComponent, CFARComponent, SpectrumSensingComponent, DRFMComponent,
    )
}


def create_runtime_component(cid, **parameters):
    """按 CID 构建真实运行组件；未实现的 CID 明确报错，不做透明占位。"""
    try:
        component = RUNTIME_COMPONENT_REGISTRY[cid]
    except KeyError as exc:
        raise ValueError(f"no runtime implementation for {cid}") from exc
    return component(**parameters)
