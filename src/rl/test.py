import os

import click
import numpy as np
import torch

from src.game.const import MAX_SIZE_HAND
from src.game.core.fsm import FSM
from src.game.create import create_game_state
from src.game.draw import get_action_str
from src.game.draw import get_view_game_state_str
from src.game.main import initialize_game_state
from src.game.main import step
from src.game.view.state import ViewGameState
from src.game.view.state import get_view_game_state
from src.rl.algorithms.actor_critic.worker import _ASCENSION_LEVEL
from src.rl.models.actor_critic import ActorCritic
from src.rl.policies import PolicyBase
from src.rl.policies import PolicySoftmax
from src.rl.policies import SelectActionMetadata
from src.rl.utils import load_config


_NCOL, _ = os.get_terminal_size()


def _format_array_floats(array: np.ndarray, precision: int = 4) -> str:
    return np.array2string(
        array,
        precision=precision,
        suppress_small=True,
        formatter={"float_kind": lambda x: f"{x:.{precision}f}"},
    )


def run_simulation(
    policy: PolicyBase, num_games: int
) -> list[list[tuple[ViewGameState, SelectActionMetadata]]]:
    games = []
    for _ in range(num_games):
        # Start game
        game_state = create_game_state(_ASCENSION_LEVEL)
        initialize_game_state(game_state)

        # Create a list to store game views in each step
        view_game_states = []
        while not game_state.fsm == FSM.GAME_OVER:
            view_game_state = get_view_game_state(game_state)

            action, select_action_metadata = policy.select_action(view_game_state)
            step(game_state, action)

            view_game_states.append((view_game_state, action, select_action_metadata))

        games.append(view_game_states)

    return games


def display_game(game_replay: list[tuple[ViewGameState, SelectActionMetadata]]) -> None:
    for view_game_state, action, select_action_metadata in game_replay:
        view_game_state_str = get_view_game_state_str(view_game_state)
        action_str = get_action_str(action, view_game_state)
        print(view_game_state_str)
        print("-" * _NCOL)
        print(action_str)
        print("-" * _NCOL)

        # q_values_or_probs = (
        #     (
        #         select_action_metadata["q_values"]
        #         if "q_values" in select_action_metadata
        #         else select_action_metadata["probs"]
        #     )
        #     .numpy()
        #     .flatten()
        # )

        # if view_game_state.effect is None:
        #     cards_prob = q_values_or_probs[: len(view_game_state.hand)]
        #     monster_prob = q_values_or_probs[2 * MAX_SIZE_HAND]
        #     end_turn_prob = q_values_or_probs[2 * MAX_SIZE_HAND + 1]
        # else:
        #     cards_prob = q_values_or_probs[
        #         MAX_SIZE_HAND : MAX_SIZE_HAND + len(view_game_state.hand)
        #     ]
        #     monster_prob = q_values_or_probs[2 * MAX_SIZE_HAND]
        #     end_turn_prob = q_values_or_probs[2 * MAX_SIZE_HAND + 1]

        # print(
        #     f"CARDS:{_format_array_floats(cards_prob)} / "
        #     f"MONST:{monster_prob:.4f} / "
        #     f"ENDTN:{end_turn_prob:.4f}"
        # )


@click.command()
@click.option("--exp-path", help="Path to the experiment directory", required=True)
@click.option("--num-games", default=100, type=int, help="Number of games to run")
@click.option(
    "--device", default="cpu", type=str, help="Device to run simulations on (cpu or cuda)"
)
@click.option(
    "--sort-by",
    type=click.Choice(["health", "length"]),
    default="health",
    help="Sort criterion for games (health: final health, length: game duration)",
)
def main(exp_path, num_games, device, sort_by):
    """Run simulation of combat games using a trained model."""
    # Load config
    config = load_config(f"{exp_path}/config.yml")

    # Load model
    model = ActorCritic(**config["model"])
    model.load_state_dict(torch.load(f"{exp_path}/model.pth"))

    # Instance policy
    policy = PolicySoftmax(model, torch.device(device))
    games = run_simulation(policy, num_games=num_games)

    # Sort games based on criterion
    if sort_by == "health":
        games.sort(key=lambda x: x[-1][0].character.health_current)
        click.echo("Games sorted by final health (lowest to highest)")
    else:
        games.sort(key=lambda x: -1 * len(x))
        click.echo("Games sorted by game length (longest to shortest)")

    # Display
    display_game(games[0])


if __name__ == "__main__":
    main()
