# Waveform components

This package owns the component library and the existing OPS/OCS contracts.
The original C01-C32 definitions remain in `data/`, with C33-C34 added for IQ
front-end simulation. There are 15 CPU reference implementations, including
the newly runnable C04 digital AGC. See [RF front-end simulation](RF_FRONTEND_SIMULATION.md)
for research sources, behavior, and hardware boundaries.

| Directory | Responsibility |
| --- | --- |
| `catalog/` | Load and validate component definitions and capability mappings |
| `models/` | Component definitions and task-specific instance records |
| `specs/` | Load and validate OPS/OCS protocol and interface contracts |
| `runtime/` | Data packets, ports, metrics, and runtime base classes |
| `implementations/` | CPU reference algorithms and model registries |

See [OPS/OCS specifications](SPECIFICATIONS.md) and
[runtime usage](runtime/README.md). Static resource levels represent relative
simulation pressure and accelerator suitability, not measured device use.
