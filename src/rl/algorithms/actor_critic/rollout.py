import slai
import torch

from src.rl.algorithms.actor_critic.episode import EpisodeStats
from src.rl.algorithms.actor_critic.workers import Exchange
from src.rl.algorithms.actor_critic.workers import Worker
from src.rl.algorithms.actor_critic.workers import drain_episodes
from src.rl.algorithms.actor_critic.workers import snapshot_to_device
from src.rl.algorithms.actor_critic.workers import step_workers
from src.rl.models import ActorCritic
from src.rl.reward import RewardStream
from src.rl.types import Rollout
from src.rl.types import X_EntityMask
from src.rl.types import X_EntityPairs
from src.rl.types import X_GameState
from src.rl.types import X_Mask
from src.rl.types import X_Owned


_AT_REST = int(slai.ActionType.Rest)

# The smith is argless: the deck selection it queues arrives as a separate halt
_AT_SMITH = int(slai.ActionType.RestSmith)


def _pad_cat(tensors: list[torch.Tensor], fill: float) -> torch.Tensor:
    """Pad every tensor along each dim but the batch dim to the longest with `fill`, then cat
    along the batch dim: each step's snapshot is as wide as its widest row, the rollout as wide
    as its widest step."""
    shape = [max(sizes) for sizes in zip(*(t.shape[1:] for t in tensors))]
    padded = []
    for t in tensors:
        if list(t.shape[1:]) != shape:
            x_padded = t.new_full((t.shape[0], *shape), fill)
            x_padded[tuple(slice(0, n) for n in t.shape)] = t
            t = x_padded
        padded.append(t)
    return torch.cat(padded, dim=0)


def _pad_cat_states(steps: list[X_GameState]) -> X_GameState:
    """The rollout's states as one batch: every block padded to the rollout's
    widest (pairs (0, 0.0), kinds -1), the per-state tensors just concatenated."""
    batch_size = [sum(len(x) for x in steps)]
    return X_GameState(
        cards=_pad_cat_entity([x.cards for x in steps], batch_size),
        relics=_pad_cat_entity([x.relics for x in steps], batch_size),
        potions=_pad_cat_entity([x.potions for x in steps], batch_size),
        monsters=_pad_cat_entity([x.monsters for x in steps], batch_size),
        event_options=_pad_cat_entity([x.event_options for x in steps], batch_size),
        character=_pad_cat_entity([x.character for x in steps], batch_size),
        health_block=_pad_cat_owned([x.health_block for x in steps], batch_size),
        prices=_pad_cat_owned([x.prices for x in steps], batch_size),
        effects=_pad_cat_owned([x.effects for x in steps], batch_size),
        modifiers=_pad_cat_owned([x.modifiers for x in steps], batch_size),
        energy=torch.cat([x.energy for x in steps]),
        map_grid=torch.cat([x.map_grid for x in steps]),
        room_node_idx=torch.cat([x.room_node_idx for x in steps]),
        room_slice_kind=torch.cat([x.room_slice_kind for x in steps]),
        map_meta=torch.cat([x.map_meta for x in steps]),
        focus=torch.cat([x.focus for x in steps]),
        ascension=torch.cat([x.ascension for x in steps]),
        reward_meta=torch.cat([x.reward_meta for x in steps]),
        shop_meta=torch.cat([x.shop_meta for x in steps]),
        event_meta=torch.cat([x.event_meta for x in steps]),
        batch_size=batch_size,
    )


def _pad_cat_entity(blocks: list[X_EntityPairs], batch_size: list[int]) -> X_EntityPairs:
    return X_EntityPairs(
        x_column=_pad_cat([block.x_column for block in blocks], 0),
        x_value=_pad_cat([block.x_value for block in blocks], 0.0),
        x_slice_kind=_pad_cat([block.x_slice_kind for block in blocks], -1),
        batch_size=batch_size,
    )


def _pad_cat_owned(blocks: list[X_Owned], batch_size: list[int]) -> X_Owned:
    return type(blocks[0])(
        x_column=_pad_cat([block.x_column for block in blocks], 0),
        x_value=_pad_cat([block.x_value for block in blocks], 0.0),
        x_entity_kind=_pad_cat([block.x_entity_kind for block in blocks], -1),
        x_offset=_pad_cat([block.x_offset for block in blocks], 0),
        batch_size=batch_size,
    )


def _pad_cat_masks(steps: list[X_Mask]) -> X_Mask:
    """The rollout's masks as one batch, each block padded (False) to the same
    width as its entity block in _pad_cat_states."""
    batch_size = [sum(len(mb) for mb in steps)]
    return X_Mask(
        mask_action_type=torch.cat([mb.mask_action_type for mb in steps]),
        cards=_pad_cat_entity_mask([mb.cards for mb in steps], batch_size),
        relics=_pad_cat_entity_mask([mb.relics for mb in steps], batch_size),
        potions=_pad_cat_entity_mask([mb.potions for mb in steps], batch_size),
        monsters=_pad_cat_entity_mask([mb.monsters for mb in steps], batch_size),
        event_options=_pad_cat_entity_mask([mb.event_options for mb in steps], batch_size),
        character=_pad_cat_entity_mask([mb.character for mb in steps], batch_size),
        rooms=_pad_cat_entity_mask([mb.rooms for mb in steps], batch_size),
        monster_is_target=_pad_cat([mb.monster_is_target for mb in steps], False),
        batch_size=batch_size,
    )


def _pad_cat_entity_mask(blocks: list[X_EntityMask], batch_size: list[int]) -> X_EntityMask:
    return X_EntityMask(
        legal=_pad_cat([block.legal for block in blocks], False),
        needs_target=_pad_cat([block.needs_target for block in blocks], False),
        batch_size=batch_size,
    )


def collect_rollout(
    model: ActorCritic,
    exchange: Exchange,
    workers: list[Worker],
    rollout_length: int,
    device: torch.device,
    curriculum_c: float,
    generator: torch.Generator,
) -> tuple[Rollout, list[EpisodeStats], dict[str, float]]:
    """Play `rollout_length` lockstep steps of every game with `model`, sampling from
    `generator`: each step one forward over all games, then the workers step them. GAE is the
    learner's"""
    E = len(exchange.rows)
    T = rollout_length
    N = T * E
    K = len(RewardStream)
    exchange.curriculum_c.fill_(curriculum_c)
    chunks_x: list[X_GameState] = []  # per step, in row order
    chunks_mb: list[X_Mask] = []
    actions = torch.empty(N, 4, dtype=torch.long, device=device)
    log_probs = torch.empty(N, device=device)
    rewards = torch.empty(T, E, K, device=device)
    dones = torch.empty(T, E, device=device)

    # Rest-site preference: device-side counters, read once after the loop (no per-step sync)
    rest_selections = torch.zeros((), device=device)
    upgrade_selections = torch.zeros((), device=device)

    # Set model to eval, turn off gradient computation
    model.eval()
    with torch.no_grad():
        x, mb = snapshot_to_device(exchange.rows, device)
        for t in range(T):
            out = model.act(x, mb, generator=generator)
            chunks_x.append(x)
            chunks_mb.append(mb)
            step_rows = slice(t * E, (t + 1) * E)
            actions[step_rows] = torch.stack(
                [out.action_type, out.kind, out.rank, out.target], dim=1
            )
            log_probs[step_rows] = out.log_prob.sum(-1)

            # Campfire decision: both Rest and RestSmith legal — which was chosen?
            both = mb.mask_action_type[:, _AT_REST] & mb.mask_action_type[:, _AT_SMITH]
            chosen = out.action_type
            rest_selections.add_((both & (chosen == _AT_REST)).sum())
            upgrade_selections.add_((both & (chosen == _AT_SMITH)).sum())

            exchange.rows.actions.copy_(actions[step_rows])
            step_workers(workers)
            rewards[t] = exchange.rows.rewards
            dones[t] = exchange.rows.dones
            x, mb = snapshot_to_device(exchange.rows, device)

    # Set model to training mode
    model.train()

    # Surface the termination vs rollout-boundary-bootstrap split (the GAE truncation
    # path): how many envs were still mid-episode at T and relied on V(s_T).
    n_campfire_selections = (rest_selections + upgrade_selections).item()
    rollout_info = {
        "rollout/terminations": dones.sum().item(),
        "rollout/boundary_truncations": float(E) - dones[T - 1].sum().item(),
        # 0.0 also when no campfire was seen this rollout
        "rollout/rest_upgrade_rate": upgrade_selections.item() / n_campfire_selections
        if n_campfire_selections
        else 0.0,
    }

    return (
        Rollout(
            game_state=_pad_cat_states(chunks_x),
            mask_batch=_pad_cat_masks(chunks_mb),
            actions=actions,
            log_probs_behavior=log_probs,
            rewards=rewards,
            dones=dones,
            final_game_state=x,
            batch_size=[],
        ),
        drain_episodes(exchange.episodes),
        rollout_info,
    )
