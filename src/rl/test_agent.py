"""
Test a trained agent by rendering a single game.

Usage:
    poetry run python -m src.rl.test_agent --exp-path experiments/ppo_hierarchical_v1
"""

import os
import time

import click
import torch

from src.game.action import Action
from src.game.core.fsm import FSM
from src.game.create import create_game_state
from src.game.draw import get_action_str
from src.game.draw import get_view_game_state_str
from src.game.main import initialize_game_state
from src.game.main import step
from src.game.view.fsm import ViewFSM
from src.game.view.state import ViewGameState
from src.game.view.state import get_view_game_state
from src.rl.action_space.masks import get_masks
from src.rl.action_space.types import HeadType
from src.rl.constants import ASCENSION_LEVEL
from src.rl.encoding.state import XGameState
from src.rl.encoding.state import encode_batch_view_game_state
from src.rl.models import ActorCritic
from src.rl.models.heads import get_grouped_probs
from src.rl.utils import load_config


N_COL, _ = os.get_terminal_size()


def get_card_probabilities(
    model: ActorCritic,
    x_game_state: XGameState,
    secondary_masks: dict[HeadType, torch.Tensor],
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

    # Get card play mask
    mask = secondary_masks[HeadType.CARD_PLAY]  # (1, MAX_HAND_SIZE)

    # Run card play head without sampling to get logits
    head_out = model.head_card_play(core_out.x_hand, core_out.x_global, mask, sample=False)

    # Get grouped probabilities (deduplicates identical cards)
    probs = get_grouped_probs(head_out.logits)  # (1, MAX_HAND_SIZE)

    return probs[0]  # Return first (only) batch item


def format_card_probabilities(
    view_game_state: ViewGameState,
    probs: torch.Tensor,
) -> str:
    """Format card probabilities grouped by card type."""
    # Group cards by name (identical cards share the same probability)
    seen: dict[str, dict] = {}
    order: list[str] = []

    for idx, card in enumerate(view_game_state.hand):
        if card.name not in seen:
            prob = probs[idx].item()
            # Handle NaN (shouldn't happen with grouped probs, but safety)
            if prob != prob:
                prob = 0.0
            playable = card.cost <= view_game_state.energy.current
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
    view_game_state: ViewGameState,
    device: torch.device,
    show_card_probs: bool = False,
    greedy: bool = False,
) -> tuple[Action, str | None]:
    """
    Get an action from the model for the given game state.

    Args:
        greedy: If True, use argmax instead of sampling (deterministic)

    Returns:
        (action, card_probs_str) where card_probs_str is None if not in combat
    """
    # Encode state
    x_game_state = encode_batch_view_game_state([view_game_state], device)

    # Get masks
    primary_mask, secondary_masks = get_masks(view_game_state, device)

    # Forward pass
    with torch.no_grad():
        output = model.forward_single(
            x_game_state, primary_mask, secondary_masks, sample=not greedy
        )

        # Get card probabilities if in combat and requested
        card_probs_str = None
        has_playable = any(c.cost <= view_game_state.energy.current for c in view_game_state.hand)
        if (
            show_card_probs
            and view_game_state.fsm == ViewFSM.COMBAT_DEFAULT
            and view_game_state.hand
            and has_playable
        ):
            probs = get_card_probabilities(model, x_game_state, secondary_masks)
            card_probs_str = format_card_probabilities(view_game_state, probs)

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

    Args:
        greedy: If True, use argmax instead of sampling (deterministic)

    Returns:
        (final_floor, final_health)
    """
    game_state = create_game_state(ASCENSION_LEVEL)
    initialize_game_state(game_state)

    step_count = 0
    while game_state.fsm != FSM.GAME_OVER:
        view_game_state = get_view_game_state(game_state)

        if verbose:
            print(get_view_game_state_str(view_game_state))
            print("-" * N_COL)

        # Get action from model
        action, card_probs_str = get_action_from_model(
            model,
            view_game_state,
            device,
            show_card_probs=verbose and show_card_probs,
            greedy=greedy,
        )

        if verbose:
            # Show card probabilities if in combat
            if card_probs_str:
                print(card_probs_str)
                print("-" * N_COL)

            action_str = get_action_str(action, view_game_state, fast_mode=False)
            print(f"Action: {action_str}")
            print("-" * N_COL)
            time.sleep(delay)

        # Execute action
        step(game_state, action, fast_mode=False)
        step_count += 1

    # Get final state
    final_view = get_view_game_state(game_state)
    final_floor = final_view.map.y_current or 0
    final_health = final_view.character.health_current

    if verbose:
        print(f"\n{'=' * N_COL}")
        print("GAME OVER")
        print(f"Final Floor: {final_floor}")
        print(f"Final Health: {final_health}")
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
    # Load config and model
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

    # Summary
    if num_games > 1:
        avg_floor = sum(r[0] for r in results) / num_games
        avg_health = sum(r[1] for r in results) / num_games
        print(f"\n--- Summary ({num_games} games) ---")
        print(f"Average Floor: {avg_floor:.1f}")
        print(f"Average Final Health: {avg_health:.1f}")


if __name__ == "__main__":
    main()
