"""
Test a trained agent by rendering a single game.

Usage:
    poetry run python -m src.rl.test_agent --exp-path experiments/ppo_hierarchical_v1
"""

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
from src.game.view.state import get_view_game_state
from src.rl.action_space import FSM_ROUTING
from src.rl.action_space import HeadType
from src.rl.action_space import get_secondary_head_type
from src.rl.action_space.masks import get_valid_mask_batch
from src.rl.constants import ASCENSION_LEVEL
from src.rl.encoding.state import encode_batch_view_game_state
from src.rl.models import ActorCritic
from src.rl.utils import load_config


def get_action_from_model(
    model: ActorCritic,
    view_game_state,
    device: torch.device,
) -> Action:
    """Get an action from the model for the given game state."""
    fsm = FSM[view_game_state.fsm.name]  # Convert ViewFSM to FSM

    # Encode state
    x_game_state = encode_batch_view_game_state([view_game_state], device)

    # Get masks
    route = FSM_ROUTING[fsm]
    masks = {
        HeadType.ACTION_TYPE: get_valid_mask_batch(HeadType.ACTION_TYPE, [view_game_state], device)
    }
    for action_type in route.action_types:
        secondary_head = get_secondary_head_type(fsm, action_type)
        if secondary_head is not None and secondary_head not in masks:
            masks[secondary_head] = get_valid_mask_batch(secondary_head, [view_game_state], device)

    # Forward pass
    with torch.no_grad():
        output = model(x_game_state, fsm, masks, sample=True)

    return output.actor.to_action()


def run_game(
    model: ActorCritic,
    device: torch.device,
    delay: float = 0.5,
    verbose: bool = True,
) -> tuple[int, int]:
    """
    Run a single game with the trained model.

    Returns:
        (final_floor, final_health)
    """
    game_state = create_game_state(ASCENSION_LEVEL)
    initialize_game_state(game_state)

    step_count = 0
    while game_state.fsm != FSM.GAME_OVER:
        view_game_state = get_view_game_state(game_state)

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Step {step_count}")
            print("=" * 80)
            print(get_view_game_state_str(view_game_state))
            print("-" * 80)

        # Get action from model
        action = get_action_from_model(model, view_game_state, device)

        if verbose:
            action_str = get_action_str(action, view_game_state, fast_mode=False)
            print(f"Action: {action_str}")
            time.sleep(delay)

        # Execute action
        step(game_state, action, fast_mode=False)
        step_count += 1

    # Get final state
    final_view = get_view_game_state(game_state)
    final_floor = final_view.map.y_current or 0
    final_health = final_view.character.health_current

    if verbose:
        print(f"\n{'=' * 80}")
        print("GAME OVER")
        print(f"Final Floor: {final_floor}")
        print(f"Final Health: {final_health}")
        print("=" * 80)

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
def main(exp_path: str, delay: float, device: str, num_games: int, quiet: bool):
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

        floor, health = run_game(model, device, delay=delay, verbose=not quiet)
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
