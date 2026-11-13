import math

import torch
import torch.nn as nn

from src.rl.constants import TOKEN_KINDS_OF_ENTITY_KIND
from src.rl.models.core import Core
from src.rl.models.heads import HeadValue
from src.rl.models.map_encoder import message_passing
from src.rl.models.policy_heads import PolicyHeads
from src.rl.reward import RewardStream
from src.rl.types import PolicyOutput
from src.rl.types import StateBatch
from src.rl.types import StateCounts


class ActorCritic(nn.Module):
    """The policy and the value function over a batch of game states.

    Each entry point runs the core once, then the heads its caller needs:
    - `act`: the Actor's step and the eval's. Selects each state's action, without values.
    - `evaluate_actions`: the Learner's update. Scores recorded actions and computes the values.
    - `get_values`: the Learner's bootstrap. Computes the values alone.
    """

    def __init__(
        self,
        dim_entity: int,
        dim_global: int,
        transformer_dim_ff: int,
        transformer_num_heads: int,
        transformer_num_blocks: int,
        gnn_num_layers: int,
        map_encoder_dim: int,
        dim_ff_primary: int,
        dim_ff_value: int,
        dim_op: int,
        dim_key: int,
    ) -> None:
        super().__init__()

        # Core
        self.core = Core(
            dim_entity=dim_entity,
            dim_global=dim_global,
            transformer_dim_ff=transformer_dim_ff,
            transformer_num_heads=transformer_num_heads,
            transformer_num_blocks=transformer_num_blocks,
            gnn_num_layers=gnn_num_layers,
            map_encoder_dim=map_encoder_dim,
        )

        # Heads
        self._policy_heads = PolicyHeads(dim_entity, dim_global, dim_ff_primary, dim_op, dim_key)
        self._head_value = HeadValue(dim_global, dim_ff_value, len(RewardStream))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply the PPO init.

        Orthogonal (sqrt(2)), near-zero logit layers (policy ≈ uniform), unit-gain value.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.core.init_token_weights()
        self._policy_heads.init_logit_weights()
        nn.init.orthogonal_(self._head_value._network[-1].weight, gain=1.0)

    def act(
        self,
        state_batch: StateBatch,
        state_counts_cpu: StateCounts,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> PolicyOutput:
        """Select each state's action.

        Without values to compute, the core runs the draw-order layer only on the states the
        policy reads.

        Parameters
        ----------
        state_batch : StateBatch
            The game states.
        state_counts_cpu : StateCounts
            The states' counts on the CPU, `state_batch.state_counts`.
        greedy : bool, default False
            Whether each level takes its argmax rather than a sample.
        generator : torch.Generator | None, default None
            The generator samples draw from, or None for the global one.

        Returns
        -------
        PolicyOutput
            The selected actions and each level's statistics. Actions use the argmax under
            `greedy`, otherwise samples from `generator`.
        """
        core_output = self.core(state_batch, state_counts_cpu, with_value=False)
        entity_counts = _count_tokens_of_entity_kind(state_counts_cpu)

        # Sampling actions runs eager: `forward` itself, not the call `compile_hot_paths` compiles
        return self._policy_heads.forward(
            state_batch, core_output, entity_counts, greedy=greedy, generator=generator
        )

    def evaluate_actions(
        self,
        state_batch: StateBatch,
        state_counts_cpu: StateCounts,
        x_recorded_actions: torch.Tensor,
    ) -> tuple[PolicyOutput, torch.Tensor]:
        """Score the recorded actions and compute the values.

        Parameters
        ----------
        state_batch : StateBatch
            The game states.
        state_counts_cpu : StateCounts
            The states' counts on the CPU, `state_batch.state_counts`.
        x_recorded_actions : torch.Tensor
            (B, 3) long. Each state's recorded action: action type, token offset within that
            state, Monster target rank; -1 where a selection or target didn't apply.

        Returns
        -------
        policy_output : PolicyOutput
            The recorded actions and each level's statistics.
        x_values : torch.Tensor
            (B, len(RewardStream)). The values.
        """
        core_output = self.core(state_batch, state_counts_cpu)
        x_values = self._head_value(core_output.x_value_context)
        entity_counts = _count_tokens_of_entity_kind(state_counts_cpu)
        policy_output = self._policy_heads(
            state_batch, core_output, entity_counts, x_recorded_actions
        )
        return policy_output, x_values

    def get_values(self, state_batch: StateBatch, state_counts_cpu: StateCounts) -> torch.Tensor:
        """Compute the values.

        Parameters
        ----------
        state_batch : StateBatch
            The game states.
        state_counts_cpu : StateCounts
            The states' counts on the CPU, `state_batch.state_counts`.

        Returns
        -------
        torch.Tensor
            (B, len(RewardStream)). The values.
        """
        return self._head_value(self.core(state_batch, state_counts_cpu).x_value_context)


def compile_hot_paths(model: ActorCritic) -> None:
    """Fuse the update's hot paths.

    They are each transformer block, the draw-order layer's block, the Map GNN's message passing,
    and the policy heads on recorded actions. All but the heads are autotuned, so the first update
    compiles for a minute or two.

    Parameters
    ----------
    model : ActorCritic
        The model, compiled in place.
    """
    mode = "max-autotune-no-cudagraphs"
    for block in model.core._entity_transformer._entity_transformer_blocks:
        block.compile(dynamic=True, mode=mode)
    model.core._draw_order_layer._draw_order_block.compile(dynamic=True, mode=mode)
    model.core._map_gnn.message_passing = torch.compile(message_passing, dynamic=True, mode=mode)

    # Not autotuned: an empty block's key-net matmul fails to lower under autotuning
    model._policy_heads.compile(dynamic=True)


def _count_tokens_of_entity_kind(state_counts_cpu: StateCounts) -> list[int]:
    """Count each Entity kind's tokens over the batch.

    Parameters
    ----------
    state_counts_cpu : StateCounts
        The states' counts on the CPU.

    Returns
    -------
    list[int]
        Each Entity kind's number of tokens, indexed by `EntityKind`.
    """
    # Each Entity kind's tokens: its token kinds' summed on the host, in `EntityKind` order
    token_kind_counts_total = state_counts_cpu.x_token_kind_count.sum(0).tolist()
    return [
        sum(token_kind_counts_total[token_kind] for token_kind in token_kinds)
        for _, token_kinds in sorted(TOKEN_KINDS_OF_ENTITY_KIND.items())
    ]
