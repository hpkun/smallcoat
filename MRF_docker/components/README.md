# Waveform components

This package owns the component library and the existing OPS/OCS contracts.
The original 32 static definitions and capability mappings are preserved in
`data/`. The 12 CPU reference implementations remain in `implementations/`.

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
