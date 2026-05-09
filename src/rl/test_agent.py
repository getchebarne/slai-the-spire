"""
Test a trained agent by rendering a single game.

Usage:
    poetry run python -m src.rl.test_agent --exp-path experiments/ppo_hierarchical_v1
"""

import os
import random
import time

import click
import slai
import torch

from src.rl.action_space.masks import MaskBatch
from src.rl.action_space.masks import get_mask_batch
from src.rl.action_space.types import HeadTypePrimary
from src.rl.constants import ASCENSION_LEVEL
from src.rl.encoding.state import XGameState
from src.rl.encoding.state import encode_batch_view_game_state
from src.rl.env_wrapper import EnvWrapper
from src.rl.models import ActorCritic
from src.rl.models.heads import get_grouped_probs
from src.rl.utils import load_config


try:
    N_COL, _ = os.get_terminal_size()
except OSError:
    N_COL = 120


def _format_view(view: slai.GameState, awaiting_target: bool) -> str:
    """Compact textual rendering of the game state."""
    lines = [
        f"Phase: {type(view.phase).__name__}{' (awaiting target)' if awaiting_target else ''}",
        f"Char: HP {view.character.health}/{view.character.health_max}  Block {view.character.block}",
    ]
    if view.character.modifiers:
        mods = ", ".join(f"{type(m.kind).__name__}={m.stacks}" for m in view.character.modifiers)
        lines.append(f"  modifiers: {mods}")

    if view.monsters:
        lines.append("Monsters:")
        for i, m in enumerate(view.monsters):
            intent = m.intent
            intent_str = ""
            if intent.damage:
                intent_str = f"ATK {intent.damage}x{intent.instances or 1}"
            elif intent.block:
                intent_str = "BLOCK"
            elif intent.buff:
                intent_str = "BUFF"
            elif intent.debuff:
                intent_str = "DEBUFF"
            lines.append(f"  [{i}] {m.name}: HP {m.health}/{m.health_max} Block {m.block}  → {intent_str}")

    lines.append(f"Energy: {view.energy.current}/{view.energy.max}")
    if view.hand:
        lines.append("Hand:")
        for i, c in enumerate(view.hand):
            tgt = " (target)" if c.requires_target else ""
            playable = "" if c.cost <= view.energy.current and c.playable else " (unplayable)"
            lines.append(f"  [{i}] {c.name} cost={c.cost}{tgt}{playable}")

    if view.card_rewards:
        lines.append(f"Card rewards: {[c.name for c in view.card_rewards]}")
    return "\n".join(lines)


def get_card_probabilities(
    model: ActorCritic,
    x_game_state: XGameState,
    mask_batch: MaskBatch,
) -> torch.Tensor:
    """
    Get grouped probabilities for cards in hand from the card play head.

    Uses grouped softmax: identical cards are deduplicated so each card type
    gets a single probability (not split across duplicates).

    Returns:
        Tensor of per-position grouped probabilities (MAX_HAND_SIZE,).
        Identical cards share the same probability value.
    """
    # Run core encoder
    core_out = model.core(x_game_state)

    # Get selection mask for COMBAT_DEFAULT (which is the card play mask)
    mask = mask_batch.selection_masks[int(HeadTypePrimary.COMBAT_DEFAULT)]  # (1, MAX_HAND_SIZE)

    # Run card play head without sampling to get logits
    head_out = model.head_card_play(core_out.x_hand, core_out.x_global, mask, sample=False)

    # Get grouped probabilities (deduplicates identical cards)
    probs = get_grouped_probs(head_out.logits)  # (1, MAX_HAND_SIZE)

    return probs[0]  # Return first (only) batch item


def format_card_probabilities(
    view: slai.GameState,
    probs: torch.Tensor,
) -> str:
    """Format card probabilities grouped by card type."""
    seen: dict[str, dict] = {}
    order: list[str] = []

    for idx, card in enumerate(view.hand):
        if card.name not in seen:
            prob = probs[idx].item()
            if prob != prob:  # NaN guard
                prob = 0.0
            playable = card.cost <= view.energy.current and card.playable
            seen[card.name] = {"prob": prob, "count": 1, "playable": playable}
            order.append(card.name)
        else:
            seen[card.name]["count"] += 1

    lines = ["Card Play Probabilities:"]
    for name in order:
        info = seen[name]
        count_str = f" x{info['count']}" if info["count"] > 1 else ""
        status = "" if info["playable"] else " (unplayable)"
        bar_len = int(info["prob"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"  {name}{count_str:4} [{bar}] {info['prob']:5.1%}{status}")
    return "\n".join(lines)


def get_action_from_model(
    model: ActorCritic,
    wrapper: EnvWrapper,
    device: torch.device,
    show_card_probs: bool = False,
    greedy: bool = False,
) -> tuple[object, str | None]:
    """
    Get an action from the model for the wrapper's current state.

    Args:
        greedy: If True, use argmax instead of sampling (deterministic)

    Returns:
        (action, card_probs_str). action may be a slai.Action.* or a
        wrapper marker (PendingCardPlay / ResolveCardPlay).
    """
    view = wrapper.obs

    # Encode state
    x_game_state = encode_batch_view_game_state([view], device)

    # Get masks (wrapper-aware: routes correctly when awaiting target)
    mask_batch = get_mask_batch([wrapper], device)

    # Forward pass
    with torch.no_grad():
        output = model.forward_single(x_game_state, mask_batch, sample=not greedy)

        # Get card probabilities only when in true CombatDefault (not buffered).
        card_probs_str = None
        has_playable = any(
            c.cost <= view.energy.current and c.playable for c in view.hand
        )
        if (
            show_card_probs
            and isinstance(view.phase, slai.Phase.CombatDefault)
            and not wrapper.is_awaiting_target
            and view.hand
            and has_playable
        ):
            probs = get_card_probabilities(model, x_game_state, mask_batch)
            card_probs_str = format_card_probabilities(view, probs)

    return output.to_action(), card_probs_str


def run_game(
    model: ActorCritic,
    device: torch.device,
    delay: float = 0.5,
    verbose: bool = True,
    show_card_probs: bool = True,
    greedy: bool = False,
) -> tuple[int, int]:
    """
    Run a single game with the trained model.

    Returns:
        (final_floor, final_health)
    """
    wrapper = EnvWrapper(ascension=ASCENSION_LEVEL)
    wrapper.reset(seed=random.randint(0, 2**31 - 1))

    step_count = 0
    terminated = False
    while not terminated:
        if verbose:
            print(_format_view(wrapper.obs, wrapper.is_awaiting_target))
            print("-" * N_COL)

        action, card_probs_str = get_action_from_model(
            model,
            wrapper,
            device,
            show_card_probs=verbose and show_card_probs,
            greedy=greedy,
        )

        if verbose:
            if card_probs_str:
                print(card_probs_str)
                print("-" * N_COL)
            print(f"Action: {type(action).__name__} {getattr(action, '__dict__', '')}")
            print("-" * N_COL)
            time.sleep(delay)

        _, _, terminated, _, _ = wrapper.step(action)
        step_count += 1

    final_view = wrapper.obs
    final_floor = final_view.map.y_current or 0
    final_health = final_view.character.health

    if verbose:
        print(f"\n{'=' * N_COL}")
        print("GAME OVER")
        print(f"Final Floor: {final_floor}")
        print(f"Final Health: {final_health}")
        print(f"Steps: {step_count}")
        print("=" * N_COL)

    return final_floor, final_health


@click.command()
@click.option(
    "--exp-path",
    default="experiments/ppo_hierarchical_v1",
    help="Path to the experiment directory",
)
@click.option(
    "--delay",
    default=0.3,
    type=float,
    help="Delay between steps (seconds)",
)
@click.option(
    "--device",
    default="cpu",
    type=str,
    help="Device (cpu or cuda)",
)
@click.option(
    "--num-games",
    default=1,
    type=int,
    help="Number of games to run",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Run without rendering (just show final results)",
)
@click.option(
    "--no-probs",
    is_flag=True,
    help="Don't show card probabilities during combat",
)
@click.option(
    "--greedy",
    is_flag=True,
    help="Use greedy (argmax) selection instead of sampling",
)
def main(
    exp_path: str,
    delay: float,
    device: str,
    num_games: int,
    quiet: bool,
    no_probs: bool,
    greedy: bool,
):
    """Test a trained agent by running games."""
    config = load_config(f"{exp_path}/config.yml")
    model = ActorCritic(**config["model"])
    model.load_state_dict(torch.load(f"{exp_path}/model.pth", weights_only=True))
    model.eval()

    device = torch.device(device)
    model.to(device)

    print(f"Loaded model from {exp_path}")

    results = []
    for i in range(num_games):
        if not quiet:
            print(f"\n--- Game {i + 1}/{num_games} ---")

        floor, health = run_game(
            model,
            device,
            delay=delay,
            verbose=not quiet,
            show_card_probs=not no_probs,
            greedy=greedy,
        )
        results.append((floor, health))

        if quiet:
            print(f"Game {i + 1}: Floor {floor}, Health {health}")

    if num_games > 1:
        avg_floor = sum(r[0] for r in results) / num_games
        avg_health = sum(r[1] for r in results) / num_games
        print(f"\n--- Summary ({num_games} games) ---")
        print(f"Average Floor: {avg_floor:.1f}")
        print(f"Average Final Health: {avg_health:.1f}")


if __name__ == "__main__":
    main()
