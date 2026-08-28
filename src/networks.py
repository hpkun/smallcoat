from __future__ import annotations

import torch
from torch import nn

from .observation_builder import LINK_FEATURE_DIM
from .observation_builder import MAX_NEIGHBOR_LINKS
from .observation_builder import NODE_LOAD_DIM
from .observation_builder import OBSERVATION_INPUT_DIM
from .observation_builder import TASK_FEATURE_DIM


class MLP(nn.Module):
    """Generic multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ActorNetwork(nn.Module):
    """Actor network with a task-resource self-attention encoder before the MLP head."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_task_slots: int = 1,
        hidden_dims: list[int] | None = None,
        use_self_attention: bool = True,
        attention_embed_dim: int = 64,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if num_task_slots <= 0 or state_dim % num_task_slots != 0:
            raise ValueError(
                "ActorNetwork received incompatible state_dim/num_task_slots: "
                f"{state_dim}, {num_task_slots}"
            )

        self.num_task_slots = num_task_slots
        self.per_slot_state_dim = state_dim // num_task_slots
        self.use_self_attention = use_self_attention
        self.backbone_input_dim = state_dim

        if self.use_self_attention:
            if self.per_slot_state_dim != OBSERVATION_INPUT_DIM:
                raise ValueError(
                    "Self-attention actor requires per-slot state dim to match "
                    f"OBSERVATION_INPUT_DIM={OBSERVATION_INPUT_DIM}, got {self.per_slot_state_dim}."
                )
            if attention_embed_dim % attention_heads != 0:
                raise ValueError("attention_embed_dim must be divisible by attention_heads.")

            self.node_projection = nn.Linear(NODE_LOAD_DIM, attention_embed_dim)
            self.task_projection = nn.Linear(TASK_FEATURE_DIM, attention_embed_dim)
            self.link_projection = nn.Linear(LINK_FEATURE_DIM, attention_embed_dim)
            self.token_type_embedding = nn.Parameter(torch.zeros(1, 2 + MAX_NEIGHBOR_LINKS, attention_embed_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=attention_embed_dim,
                nhead=attention_heads,
                dim_feedforward=attention_embed_dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.attention_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.slot_norm = nn.LayerNorm(attention_embed_dim)
            self.backbone_input_dim = num_task_slots * attention_embed_dim

        self.backbone = MLP(self.backbone_input_dim, hidden_dims or [128, 128], action_dim)

    def _encode_slot_tokens(self, slot_state: torch.Tensor) -> torch.Tensor:
        node_end = NODE_LOAD_DIM
        task_end = node_end + TASK_FEATURE_DIM
        node_token = self.node_projection(slot_state[..., :node_end]).unsqueeze(1)
        task_token = self.task_projection(slot_state[..., node_end:task_end]).unsqueeze(1)
        link_features = slot_state[..., task_end:].view(
            slot_state.shape[0],
            MAX_NEIGHBOR_LINKS,
            LINK_FEATURE_DIM,
        )
        link_tokens = self.link_projection(link_features)
        tokens = torch.cat([node_token, task_token, link_tokens], dim=1)
        tokens = tokens + self.token_type_embedding
        encoded_tokens = self.attention_encoder(tokens)
        task_conditioned_summary = encoded_tokens[:, 1, :] + encoded_tokens.mean(dim=1)
        return self.slot_norm(task_conditioned_summary)

    def _encode_with_attention(self, state: torch.Tensor) -> torch.Tensor:
        original_shape = state.shape
        flat_state = state.reshape(-1, original_shape[-1])
        slot_states = flat_state.view(-1, self.num_task_slots, self.per_slot_state_dim)
        flat_slot_states = slot_states.reshape(-1, self.per_slot_state_dim)
        encoded_slots = self._encode_slot_tokens(flat_slot_states)
        encoded = encoded_slots.reshape(flat_state.shape[0], self.num_task_slots, -1)
        encoded = encoded.reshape(flat_state.shape[0], self.backbone_input_dim)
        return encoded.reshape(*original_shape[:-1], self.backbone_input_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        actor_input = self._encode_with_attention(state) if self.use_self_attention else state
        return self.backbone(actor_input)


class CriticNetwork(nn.Module):
    """Critic network."""

    def __init__(
        self,
        joint_state_dim: int,
        joint_action_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        input_dim = joint_state_dim + joint_action_dim
        self.backbone = MLP(input_dim, hidden_dims or [256, 128], 1)

    def forward(self, joint_state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat([joint_state, joint_action], dim=-1)
        return self.backbone(critic_input)


class VariableTaskActorNetwork(nn.Module):
    """Shared per-task policy that accepts a variable number of task tokens."""

    def __init__(
        self,
        per_task_state_dim: int,
        per_task_action_dim: int,
        *,
        use_self_attention: bool = False,
        use_resource_awareness: bool = False,
        attention_embed_dim: int = 64,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if per_task_state_dim != OBSERVATION_INPUT_DIM:
            raise ValueError(
                "VariableTaskActorNetwork requires one observation block per task: "
                f"expected {OBSERVATION_INPUT_DIM}, got {per_task_state_dim}."
            )
        self.per_task_state_dim = int(per_task_state_dim)
        self.per_task_action_dim = int(per_task_action_dim)
        self.use_self_attention = bool(use_self_attention)
        self.use_resource_awareness = bool(use_resource_awareness)
        if self.use_resource_awareness:
            self.local_projection = nn.Linear(NODE_LOAD_DIM, attention_embed_dim)
            self.task_projection = nn.Linear(TASK_FEATURE_DIM, attention_embed_dim)
            self.candidate_projection = nn.Linear(
                LINK_FEATURE_DIM, attention_embed_dim
            )
            self.resource_attention = nn.MultiheadAttention(
                embed_dim=attention_embed_dim,
                num_heads=attention_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.resource_norm = nn.LayerNorm(attention_embed_dim)
        else:
            self.task_encoder = MLP(per_task_state_dim, [128], attention_embed_dim)
        if self.use_self_attention:
            layer = nn.TransformerEncoderLayer(
                d_model=attention_embed_dim,
                nhead=attention_heads,
                dim_feedforward=attention_embed_dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.task_attention = nn.TransformerEncoder(layer, num_layers=1)
            self.task_norm = nn.LayerNorm(attention_embed_dim)
        self.action_head = MLP(attention_embed_dim, [128], per_task_action_dim)

    def _encode_resource_aware_tasks(self, task_states: torch.Tensor) -> torch.Tensor:
        leading_shape = task_states.shape[:-1]
        flat_states = task_states.reshape(-1, self.per_task_state_dim)
        node_end = NODE_LOAD_DIM
        task_end = node_end + TASK_FEATURE_DIM
        local = self.local_projection(flat_states[..., :node_end])
        task_query = self.task_projection(flat_states[..., node_end:task_end])
        candidates = flat_states[..., task_end:].view(
            flat_states.shape[0],
            MAX_NEIGHBOR_LINKS,
            LINK_FEATURE_DIM,
        )
        padding_mask = candidates.abs().sum(dim=-1).eq(0.0)
        candidate_tokens = self.candidate_projection(candidates)
        attended, _ = self.resource_attention(
            query=task_query.unsqueeze(-2),
            key=candidate_tokens,
            value=candidate_tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        encoded = self.resource_norm(local + task_query + attended.squeeze(-2))
        return encoded.reshape(*leading_shape, encoded.shape[-1])

    def forward(
        self,
        task_states: torch.Tensor,
        task_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if task_states.ndim == 2:
            task_states = task_states.unsqueeze(0)
            squeeze_batch = True
        elif task_states.ndim == 3:
            squeeze_batch = False
        else:
            raise ValueError("task_states must have shape [tasks, state] or [batch, tasks, state]")
        if self.use_resource_awareness:
            encoded = self._encode_resource_aware_tasks(task_states)
        else:
            encoded = self.task_encoder(task_states)
        if self.use_self_attention:
            padding_mask = None if task_mask is None else ~task_mask.bool()
            encoded = self.task_attention(encoded, src_key_padding_mask=padding_mask)
            encoded = self.task_norm(encoded)
        actions = self.action_head(encoded)
        if task_mask is not None:
            actions = actions * task_mask.unsqueeze(-1).to(actions.dtype)
        return actions.squeeze(0) if squeeze_batch else actions


class VariableTaskCriticNetwork(nn.Module):
    """Permutation-invariant centralized critic over variable task sets."""

    def __init__(self, per_task_state_dim: int, per_task_action_dim: int) -> None:
        super().__init__()
        self.per_task_state_dim = int(per_task_state_dim)
        self.per_task_action_dim = int(per_task_action_dim)
        self.token_encoder = MLP(
            self.per_task_state_dim + self.per_task_action_dim,
            [256, 128],
            128,
        )
        self.value_head = MLP(128 * 2 + 1, [256, 128], 1)

    def forward(
        self,
        task_states: torch.Tensor,
        task_actions: torch.Tensor,
        task_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.token_encoder(torch.cat([task_states, task_actions], dim=-1))
        mask = task_mask.unsqueeze(-1).to(encoded.dtype)
        count = mask.sum(dim=1).clamp(min=1.0)
        summed = (encoded * mask).sum(dim=1)
        mean = summed / count
        pooled = torch.cat([mean, summed / count.sqrt(), count / 100.0], dim=-1)
        return self.value_head(pooled)


class VariableTaskValueNetwork(nn.Module):
    """Permutation-invariant centralized state-value network for CMPPO."""

    def __init__(self, per_task_state_dim: int) -> None:
        super().__init__()
        self.per_task_state_dim = int(per_task_state_dim)
        self.token_encoder = MLP(per_task_state_dim, [256, 128], 128)
        self.value_head = MLP(128 * 2 + 1, [256, 128], 1)

    def forward(self, task_states: torch.Tensor, task_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.token_encoder(task_states)
        mask = task_mask.unsqueeze(-1).to(encoded.dtype)
        count = mask.sum(dim=1).clamp(min=1.0)
        summed = (encoded * mask).sum(dim=1)
        mean = summed / count
        pooled = torch.cat([mean, summed / count.sqrt(), count / 100.0], dim=-1)
        return self.value_head(pooled).squeeze(-1)
