# OPS and OCS specifications

This project uses OpenCPI's separation of protocol specifications (OPS) and
component specifications (OCS) as an interface design reference. The files
here are project JSON specifications, not OpenCPI XML or OpenCPI wire format.

## OPS

[`data/ops.json`](data/ops.json) is the source of truth for the 15 task data
format IDs supported by the static matcher. Each protocol has an ID, version,
data domain, NumPy dtype kinds, allowed array dimensions and item unit. The
existing task graph format strings are protocol IDs, so v1/v2 task JSON files
remain valid. A change to a protocol's meaning should use a new version or ID;
`schema_version` only versions the JSON document structure.

The current runtime carries one data operation per `DataPacket`: `samples` is
the payload; `sample_rate` (samples/s), `center_frequency` (Hz), `timestamp`
(seconds), `domain` and JSON `metadata` are its common envelope. OPS currently
specifies the payload contract and item unit. It does not specify binary
serialization, opcodes, endianness, transport framing or arbitrary OpenCPI
message operations.

## OCS

[`data/ocs.json`](data/ocs.json) defines interfaces for the 12 runnable
components. Each component has a CID, version, named input/output ports that
refer to OPS IDs, typed configuration properties and mandatory properties. The
property names match the runtime constructors. The 20 catalog entries without runnable interfaces
remain descriptive entries in [`data/components.json`](data/components.json);
their ports and parameters have not been assigned invented protocol semantics.

The matcher checks a task's OPS ID against the OCS port declaration, then
against the configured runtime port. The second check matters for components
whose mode changes their domain or output type, such as FFT/IFFT and hard/soft
demodulation. Existing parameter adaptation and component validation still
check value ranges and relationships; OCS property types alone do not replace
those rules.

```python
from task_repr import load_component_specs, load_protocol_specs

protocol = load_protocol_specs()["frequency/complex"]
interface = load_component_specs()["C08"]
assert protocol.domain == "frequency"
assert "frequency/complex" in interface.ports["out"].protocols
assert interface.properties["fft_size"] == "integer"
```

To add a runnable format or component, define its OPS entry first, reference
that ID from its OCS ports, keep OCS properties in sync with the constructor,
then add matching and runtime tests for its configured modes.
