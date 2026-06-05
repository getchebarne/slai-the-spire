from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HeadOutput:
    logits: torch.Tensor  # (B, num_options) raw scores after masking
    indices: torch.Tensor | None  # (B,) sampled indices, None if not sampling
    log_probs: torch.Tensor | None  # (B,) log probs of sampled indices, None if not sampling


# =============================================================================
# Grouped Sampling
# =============================================================================
#
# When two slots hold semantically interchangeable entities (e.g., two
# Strikes in hand), the policy gradient should treat them as a single
# action choice — not two distinct slot picks. Otherwise raw per-slot
# softmax inflates `P(type) ∝ N_type * exp(L_type)` (the multiplicity
# enters the softmax denominator), spuriously biasing the policy toward
# whichever type happens to have more copies.
#
# Identity is supplied externally via `group_ids: (B, N) int` (same id =
# interchangeable; -1 = invalid/padding). The mask layer computes group
# ids per state from card attributes (see `card.card_identity_ids`).
# Heads that don't need dedup (binary choice, monster select, map select,
# relic select) pass `group_ids=None` to fall back to per-slot sampling.
#
# Sampling procedure: softmax over one representative per unique id →
# sample a group → uniformly pick an instance within the chosen group.
# log_prob = log(group_probability); the within-group uniform pick
# carries no policy gradient.
# =============================================================================


def _singleton_group_ids(masked_logits: torch.Tensor) -> torch.Tensor:
    """One-id-per-slot fallback for heads that don't need dedup."""
    B, N = masked_logits.shape
    return torch.arange(N, device=masked_logits.device).expand(B, -1)


def _first_occurrence_mask(group_ids: torch.Tensor) -> torch.Tensor:
    """For each row, mark the first occurrence of each non-(-1) group id.

    group_ids: (B, N) int (-1 at invalid slots)
    returns:   (B, N) bool — True at slot s iff no earlier slot in the
               same row has the same group_id, and the slot is valid.
    """
    B, N = group_ids.shape
    device = group_ids.device
    valid = group_ids >= 0
    # same[b, i, j] = True if positions i and j share a group id
    same = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)
    # earlier[i, j] = True if j < i (strict lower triangular)
    earlier = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=-1)
    # has_earlier_dup[b, i] = exists valid j < i with same group id
    has_earlier_dup = (same & earlier.unsqueeze(0) & valid.unsqueeze(1)).any(dim=2)
    return valid & ~has_earlier_dup


def sample_grouped(
    logits: torch.Tensor,
    mask: torch.Tensor,
    sample: bool,
    group_ids: torch.Tensor | None = None,
) -> HeadOutput:
    """Apply mask, then sample under the group distribution implied by
    `group_ids`. If `group_ids` is None, every slot is its own group
    (per-slot Categorical, behavior identical to a non-grouped head)."""
    masked_logits = logits.masked_fill(~mask, float("-inf"))

    if not sample:
        return HeadOutput(logits=masked_logits, indices=None, log_probs=None)

    if group_ids is None:
        group_ids = _singleton_group_ids(masked_logits)
    # Treat invalid slots as id=-1 even if the caller provided real ids
    # (defensive — keeps the rep mask honest).
    group_ids = torch.where(mask, group_ids, torch.full_like(group_ids, -1))

    is_rep = _first_occurrence_mask(group_ids)
    rep_logits = masked_logits.masked_fill(~is_rep, float("-inf"))

    group_dist = torch.distributions.Categorical(logits=rep_logits)
    rep_idx = group_dist.sample()
    log_probs = group_dist.log_prob(rep_idx)

    # Uniform within-group instance pick
    chosen_group = group_ids.gather(1, rep_idx.unsqueeze(1)).squeeze(1)  # (B,)
    in_group = (group_ids == chosen_group.unsqueeze(1)) & (group_ids >= 0)
    uniform = in_group.float()
    uniform = uniform / uniform.sum(dim=1, keepdim=True)
    indices = torch.distributions.Categorical(probs=uniform).sample()

    return HeadOutput(logits=masked_logits, indices=indices, log_probs=log_probs)


def _grouped_dist_and_rep_indices(
    masked_logits: torch.Tensor,
    indices: torch.Tensor,
    group_ids: torch.Tensor | None,
) -> tuple[torch.distributions.Categorical, torch.Tensor]:
    if group_ids is None:
        group_ids = _singleton_group_ids(masked_logits)
    valid = masked_logits != float("-inf")
    group_ids = torch.where(valid, group_ids, torch.full_like(group_ids, -1))

    is_rep = _first_occurrence_mask(group_ids)
    rep_logits = masked_logits.masked_fill(~is_rep, float("-inf"))
    group_dist = torch.distributions.Categorical(logits=rep_logits)

    chosen_group = group_ids.gather(1, indices.unsqueeze(1)).squeeze(1)
    is_chosen_group = is_rep & (group_ids == chosen_group.unsqueeze(1))
    rep_indices = is_chosen_group.int().argmax(dim=1)
    return group_dist, rep_indices


def recompute_grouped_log_prob_and_entropy(
    masked_logits: torch.Tensor,
    indices: torch.Tensor,
    group_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For PPO: replay the recorded action under current logits.

    `indices` is the recorded slot pick. We recover the group it belongs
    to, then compute that group's log_prob and the distribution's
    entropy. If `group_ids` is None, falls back to per-slot.
    """
    group_dist, rep_indices = _grouped_dist_and_rep_indices(masked_logits, indices, group_ids)
    return group_dist.log_prob(rep_indices), group_dist.entropy()


def _grouped_log_prob_only(
    masked_logits: torch.Tensor,
    indices: torch.Tensor,
    group_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same as `recompute_grouped_log_prob_and_entropy` but skips entropy.
    Used per-step inside multi-pick recompute, where the per-step entropy
    is discarded in favor of an initial-distribution entropy proxy."""
    group_dist, rep_indices = _grouped_dist_and_rep_indices(masked_logits, indices, group_ids)
    return group_dist.log_prob(rep_indices)


def _init_cur_group_ids(
    mask: torch.Tensor,
    group_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Per-step state for sequential multi-pick: mutable group ids with
    -1 at masked-out slots. Singleton groups when caller passes None.
    `torch.where` returns a fresh tensor, safe to mutate."""
    B, N = mask.shape
    if group_ids is None:
        group_ids = torch.arange(N, device=mask.device).expand(B, -1)
    return torch.where(mask, group_ids, torch.full_like(group_ids, -1))


def _initial_group_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
    group_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Entropy of the pre-pick group distribution. Used as the multi-pick
    head's entropy proxy — exact joint entropy of without-replacement
    sampling is intractable; this captures "how indecisive was the
    model's first pick"."""
    init_group_ids = _init_cur_group_ids(mask, group_ids)
    is_rep_init = _first_occurrence_mask(init_group_ids)
    rep_logits_init = logits.masked_fill(~mask | ~is_rep_init, float("-inf"))
    return torch.distributions.Categorical(logits=rep_logits_init).entropy()


def get_grouped_probs(
    masked_logits: torch.Tensor,
    group_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-slot probability under the grouped distribution. All members
    of a group share the same probability. Used for visualization
    (test_agent.py renders this)."""
    if group_ids is None:
        group_ids = _singleton_group_ids(masked_logits)
    valid = masked_logits != float("-inf")
    group_ids = torch.where(valid, group_ids, torch.full_like(group_ids, -1))

    is_rep = _first_occurrence_mask(group_ids)
    rep_logits = masked_logits.masked_fill(~is_rep, float("-inf"))
    rep_probs = torch.softmax(rep_logits, dim=-1)

    # Propagate each representative's probability to all members of its group
    same = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)  # (B, N, N)
    return torch.sum(same.float() * rep_probs.unsqueeze(1), dim=2)


# =============================================================================
# Binary Choice Head (for decision primaries)
# =============================================================================


class HeadBinaryChoice(nn.Module):
    """
    Small MLP for binary decisions (e.g., end_turn vs play_card).

    One instance per decision primary (COMBAT_DEFAULT, CARD_REWARD, REST_SITE).
    """

    def __init__(self, dim_global: int, dim_ff: int, num_choices: int = 2):
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_choices),
        )

    def forward(
        self,
        x_global: torch.Tensor,
        mask: torch.Tensor,
        sample: bool = True,
    ) -> HeadOutput:
        """
        Score and optionally sample from binary choices.

        Args:
            x_global: Global context vector (B, dim_global)
            mask: Valid choice mask (B, num_choices), True = valid
            sample: Whether to sample an action

        Returns:
            HeadOutput with scores and optionally sampled indices
        """
        logits = self._scorer(x_global)
        return sample_grouped(logits, mask, sample, group_ids=None)


# =============================================================================
# Entity Selection Heads (Secondary)
# =============================================================================


class HeadEntitySelection(nn.Module):
    """
    Base class for heads that select one entity from a sequence.
    Used for cards, monsters, etc.
    """

    def __init__(self, dim_entity: int, dim_global: int, dim_ff: int):
        """
        Args:
            dim_entity: Dimension of each entity embedding
            dim_global: Dimension of the global context vector
            dim_ff: Hidden dimension of the feedforward network
        """
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_entity + dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, 1),
        )

    def forward(
        self,
        x_entities: torch.Tensor,
        x_global: torch.Tensor,
        mask: torch.Tensor,
        sample: bool = True,
        group_ids: torch.Tensor | None = None,
    ) -> HeadOutput:
        """Score entities; sample under the grouped distribution if
        `group_ids` provided (interchangeable cards collapsed to a single
        action choice). Pass `group_ids=None` to disable dedup."""
        _, num_entities, _ = x_entities.shape

        # Broadcast global context to each entity
        x_global_exp = torch.unsqueeze(x_global, 1).expand(-1, num_entities, -1)
        x_input = torch.cat([x_entities, x_global_exp], dim=-1)

        # Score each entity
        logits = torch.squeeze(self._scorer(x_input), -1)  # (B, N)

        return sample_grouped(logits, mask, sample, group_ids=group_ids)


class HeadCardPlay(HeadEntitySelection):
    """Head for selecting a card from hand to play."""

    pass


class HeadCardRewardSelect(HeadEntitySelection):
    """Head for selecting a card from combat rewards."""

    pass


class HeadCardUpgrade(HeadEntitySelection):
    """Head for selecting a card from deck to upgrade at rest sites."""

    pass


class HeadCardSetup(HeadEntitySelection):
    """Head for selecting a hand card to mark `free_to_play_once` (Setup card).

    Separate parameters — Setup optimizes a different value function
    (which card most benefits from a discount next play).
    """

    pass


class HeadCardNightmare(HeadEntitySelection):
    """Head for selecting a hand card to copy into next turn's draw (Nightmare).

    Separate parameters — Nightmare is a "which card do I want to play
    three of next turn" decision.
    """

    pass


class HeadMonsterSelect(nn.Module):
    """
    Head for selecting a monster to target.

    Unlike other entity selection heads, this receives the active card embedding
    as an additional input so the model can make card-dependent targeting decisions.
    Input per monster: [monster_emb, global, active_card_emb].
    """

    def __init__(self, dim_entity: int, dim_global: int, dim_ff: int):
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_entity + dim_global + dim_entity, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, 1),
        )

    def forward(
        self,
        x_entities: torch.Tensor,
        x_global: torch.Tensor,
        mask: torch.Tensor,
        sample: bool = True,
        x_active_card: torch.Tensor | None = None,
    ) -> HeadOutput:
        """
        Score and optionally sample from monsters.

        Args:
            x_entities: Monster embeddings (B, N, dim_entity)
            x_global: Global context vector (B, dim_global)
            mask: Valid monster mask (B, N), True = valid
            sample: Whether to sample an action
            x_active_card: Active card embedding (B, dim_entity), zeros if no card active
        """
        _, num_entities, _ = x_entities.shape

        x_global_exp = torch.unsqueeze(x_global, 1).expand(-1, num_entities, -1)

        if x_active_card is not None:
            x_active_exp = torch.unsqueeze(x_active_card, 1).expand(-1, num_entities, -1)
            x_input = torch.cat([x_entities, x_global_exp, x_active_exp], dim=-1)
        else:
            # Fallback: zero active card (shouldn't happen in practice during targeting)
            x_active_zero = torch.zeros_like(x_entities)
            x_input = torch.cat([x_entities, x_global_exp, x_active_zero], dim=-1)

        logits = torch.squeeze(self._scorer(x_input), -1)  # (B, N)

        return sample_grouped(logits, mask, sample, group_ids=None)


# =============================================================================
# Map Selection Head
# =============================================================================


class HeadMapSelect(nn.Module):
    """
    Head for selecting the next map node.

    Unlike entity selection heads, this operates on the encoded map representation
    rather than a sequence of entity embeddings.
    """

    def __init__(self, dim_map: int, dim_global: int, dim_ff: int, num_columns: int):
        """
        Args:
            dim_map: Dimension of the map encoding
            dim_global: Dimension of the global context vector
            dim_ff: Hidden dimension of the feedforward network
            num_columns: Number of map columns (MAP_WIDTH)
        """
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_map + dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_columns),
        )

    def forward(
        self,
        x_map: torch.Tensor,
        x_global: torch.Tensor,
        mask: torch.Tensor,
        sample: bool = True,
        group_ids: torch.Tensor | None = None,
    ) -> HeadOutput:
        """Score and optionally sample map node. `group_ids` accepted for
        signature uniformity with HeadEntitySelection but ignored — map
        positions aren't deduped (each column is a distinct path)."""
        del group_ids
        x_input = torch.cat([x_map, x_global], dim=-1)
        logits = self._scorer(x_input)  # (B, num_columns)
        return sample_grouped(logits, mask, sample)


# =============================================================================
# Value Head (Critic)
# =============================================================================


# =============================================================================
# Relic Selection Head
# =============================================================================


class HeadRelicSelect(nn.Module):
    """Head for selecting which relic offer to take.

    Today MAX_RELIC_REWARDS = 1, so this head's `select` decision is
    degenerate (one valid slot). Wired up now so that bumping
    MAX_RELIC_REWARDS later doesn't require model surgery — the head
    learns a richer distribution as soon as multiple offers appear.

    Operates on global context only; relic identity isn't yet plumbed
    through the entity transformer.
    """

    def __init__(self, dim_global: int, dim_ff: int, num_relics: int):
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, num_relics),
        )

    def forward(
        self,
        x_entities: torch.Tensor,  # unused, kept for signature parity with HeadEntitySelection
        x_global: torch.Tensor,
        mask: torch.Tensor,
        sample: bool = True,
        group_ids: torch.Tensor | None = None,
    ) -> HeadOutput:
        del x_entities, group_ids  # relic embedding not yet plumbed; no dedup
        logits = self._scorer(x_global)
        return sample_grouped(logits, mask, sample)


# =============================================================================
# Multi-Pick Retain Head
# =============================================================================


@dataclass
class MultiPickHeadOutput:
    """Output of HeadCardMultiPick — multi-pick over hand."""

    indices: torch.Tensor  # (B, MAX_K) int64; -1 for unused slots
    log_prob: torch.Tensor  # (B,) sum of per-step log probs
    entropy: torch.Tensor  # (B,) entropy of the *initial* (pre-mask) distribution


class HeadCardMultiPick(nn.Module):
    """Multi-pick: pick `num` distinct hand cards.

    Sequential sample-without-replacement under one logit pass:
      1. Score each card with a per-card logit (from card+global features).
      2. For each of `num` picks: softmax over still-valid cards, sample,
         mask the picked card, repeat.
      3. Total log_prob = sum of per-step log_probs.

    Used by COMBAT_AWAIT_RETAIN (pick cards to retain across end-of-turn)
    and COMBAT_CARD_DISCARD (slai's `CombatAwaitDiscard{num}` requires
    exactly `num` indices in one action). Two separate instances live on
    `ActorCritic` so retain and discard learn independent distributions.

    PPO recompute follows the same procedure replaying the recorded picks
    under the new policy.
    """

    def __init__(self, dim_entity: int, dim_global: int, dim_ff: int):
        super().__init__()

        self._scorer = nn.Sequential(
            nn.Linear(dim_entity + dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, 1),
        )

    def _score(self, x_entities: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
        """Per-card logits. (B, N) from (B, N, dim_entity) + (B, dim_global)."""
        _, num_entities, _ = x_entities.shape
        x_global_exp = torch.unsqueeze(x_global, 1).expand(-1, num_entities, -1)
        x_input = torch.cat([x_entities, x_global_exp], dim=-1)
        return torch.squeeze(self._scorer(x_input), -1)

    def forward(
        self,
        x_entities: torch.Tensor,  # (B, N, dim_entity)
        x_global: torch.Tensor,  # (B, dim_global)
        mask: torch.Tensor,  # (B, N) bool — initial validity (whole hand)
        nums: torch.Tensor,  # (B,) int — count to pick per sample
        sample: bool = True,
        group_ids: torch.Tensor | None = None,  # (B, N) int, -1 at invalid
    ) -> "MultiPickHeadOutput":
        """Per-step grouped sampling. After each pick, the chosen slot is
        marked invalid (mask=False, group_id=-1) so the next step's group
        partition correctly drops it. The K-identical case (e.g. discard
        3 of 3 Strikes) works: at step k, the remaining Strike slots
        share their group id; the within-group uniform pick collapses to
        whichever slot is left."""
        B, N, _ = x_entities.shape
        device = x_entities.device

        logits = self._score(x_entities, x_global)
        picks = torch.full((B, N), -1, dtype=torch.long, device=device)
        log_probs = torch.zeros(B, device=device)

        cur_mask = mask.clone()
        cur_group_ids = _init_cur_group_ids(cur_mask, group_ids)

        # Single host sync to bound the loop.
        max_num = int(nums.max().item()) if B > 0 else 0
        for k in range(max_num):
            # Only sample for samples still picking. Done samples (nums <= k)
            # may have exhausted their masks; building a Categorical over an
            # all-inf row would NaN.
            sp_idx = torch.nonzero(nums > k, as_tuple=True)[0]
            if sp_idx.numel() == 0:
                break

            out = sample_grouped(
                logits[sp_idx],
                cur_mask[sp_idx],
                sample=True,
                group_ids=cur_group_ids[sp_idx],
            )
            sp_pick = out.indices
            picks[sp_idx, k] = sp_pick
            log_probs[sp_idx] = log_probs[sp_idx] + out.log_probs

            cur_mask[sp_idx, sp_pick] = False
            cur_group_ids[sp_idx, sp_pick] = -1

        entropy = _initial_group_entropy(logits, mask, group_ids)
        return MultiPickHeadOutput(indices=picks, log_prob=log_probs, entropy=entropy)

    def recompute_log_prob(
        self,
        x_entities: torch.Tensor,
        x_global: torch.Tensor,
        mask: torch.Tensor,
        nums: torch.Tensor,
        recorded_picks: torch.Tensor,  # (B, N) int64 — recorded picks, -1 for unused
        group_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay recorded picks under current logits + group ids. Same
        still-picking-only structure as `forward`."""
        B = x_entities.shape[0]
        device = x_entities.device

        logits = self._score(x_entities, x_global)
        cur_mask = mask.clone()
        cur_group_ids = _init_cur_group_ids(cur_mask, group_ids)
        log_probs = torch.zeros(B, device=device)

        max_num = int(nums.max().item()) if B > 0 else 0
        for k in range(max_num):
            sp_idx = torch.nonzero(nums > k, as_tuple=True)[0]
            if sp_idx.numel() == 0:
                break

            sp_picked = recorded_picks[sp_idx, k]
            sp_logits = logits[sp_idx].masked_fill(~cur_mask[sp_idx], float("-inf"))
            sp_log_prob = _grouped_log_prob_only(
                sp_logits,
                sp_picked,
                group_ids=cur_group_ids[sp_idx],
            )
            log_probs[sp_idx] = log_probs[sp_idx] + sp_log_prob

            cur_mask[sp_idx, sp_picked] = False
            cur_group_ids[sp_idx, sp_picked] = -1

        entropy = _initial_group_entropy(logits, mask, group_ids)
        return log_probs, entropy


class HeadValue(nn.Module):
    """
    Value head for estimating state value (critic in actor-critic).

    Outputs a single scalar value estimate for the current state.
    """

    def __init__(self, dim_global: int, dim_ff: int):
        """
        Args:
            dim_global: Dimension of the global context vector
            dim_ff: Hidden dimension of the feedforward network
        """
        super().__init__()

        self._network = nn.Sequential(
            nn.Linear(dim_global, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, 1),
        )

    def forward(self, x_global: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value.

        Args:
            x_global: Global context vector (B, dim_global)

        Returns:
            Value estimate (B, 1)
        """
        return self._network(x_global)
