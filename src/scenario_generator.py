from __future__ import annotations

from dataclasses import dataclass, replace

from .task_model import TaskModelConfig
from .task_model import UniformRange
from .task_model import MBIT
from .task_model import MCYCLE
from .task_model import MILLISECOND


@dataclass(frozen=True)
class ScenarioDefinition:
    """任务场景定义。"""

    name: str
    task_config: TaskModelConfig
    delay_sensitivity_lambda: float


TASK_SCENARIO_NAMES = (
    "delay-sensitive",
    "computation-intensive",
    "balanced",
)


def build_balanced_scenario(base_config: TaskModelConfig | None = None) -> ScenarioDefinition:
    """构建论文中的 balanced 场景。"""

    task_config = base_config or TaskModelConfig()
    lambda_value = task_config.delay_sensitivity_lambda or 8.0
    return ScenarioDefinition(
        name="balanced",
        task_config=replace(task_config, delay_sensitivity_lambda=lambda_value),
        delay_sensitivity_lambda=lambda_value,
    )


def build_delay_sensitive_scenario(
    base_config: TaskModelConfig | None = None,
) -> ScenarioDefinition:
    """构建 delay-sensitive 场景。"""

    task_config = base_config or TaskModelConfig()
    lambda_value = max(task_config.delay_sensitivity_lambda or 8.0, 12.0)
    delay_config = replace(
        task_config,
        tolerable_latency_s=UniformRange(0.0 * MILLISECOND, 20.0 * MILLISECOND),
        delay_sensitivity_lambda=lambda_value,
    )
    return ScenarioDefinition(
        name="delay-sensitive",
        task_config=delay_config,
        delay_sensitivity_lambda=lambda_value,
    )


def build_computation_intensive_scenario(
    base_config: TaskModelConfig | None = None,
) -> ScenarioDefinition:
    """构建 computation-intensive 场景。"""

    task_config = base_config or TaskModelConfig()
    lambda_value = min(task_config.delay_sensitivity_lambda or 8.0, 5.0)
    intensive_config = replace(
        task_config,
        input_size_bits=UniformRange(40 * MBIT, 90 * MBIT),
        total_compute_cycles=UniformRange(2_000 * MCYCLE, 3_000 * MCYCLE),
        delay_sensitivity_lambda=lambda_value,
    )
    return ScenarioDefinition(
        name="computation-intensive",
        task_config=intensive_config,
        delay_sensitivity_lambda=lambda_value,
    )


def build_task_scenario(
    name: str,
    base_config: TaskModelConfig | None = None,
) -> ScenarioDefinition:
    """Build one of the three baseline-compatible task distributions."""

    builders = {
        "delay-sensitive": build_delay_sensitive_scenario,
        "computation-intensive": build_computation_intensive_scenario,
        "balanced": build_balanced_scenario,
    }
    try:
        builder = builders[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported task scenario {name!r}; expected one of {TASK_SCENARIO_NAMES}."
        ) from exc
    return builder(base_config)
