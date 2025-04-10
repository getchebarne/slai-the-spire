import click
import numpy as np
import torch

from src.game.combat.constant import MAX_SIZE_HAND
from src.game.combat.create import create_combat_state
from src.game.combat.drawer import draw_combat
from src.game.combat.main import start_combat
from src.game.combat.main import step
from src.game.combat.utils import is_game_over
from src.game.combat.view import CombatView
from src.game.combat.view import view_combat
from src.rl.models.dqn import DeepQNetwork
from src.rl.policies import PolicyQMax
from src.rl.utils import load_config


def _format_array_floats(array: np.ndarray, precision: int = 4) -> str:
    return np.array2string(
        array,
        precision=precision,
        suppress_small=True,
        formatter={"float_kind": lambda x: f"{x:.{precision}f}"},
    )


def run_simulation(
    policy: PolicyQMax, num_games: int
) -> list[tuple[int, list[tuple[CombatView, np.ndarray]]]]:
    games = []
    for _ in range(num_games):
        # Start combat
        cs = create_combat_state()
        start_combat(cs)

        # Create a list to store combat views in each step
        combat_views = []
        while not is_game_over(cs.entity_manager):
            combat_view = view_combat(cs)

            action, meta = policy.select_action(combat_view)
            step(cs, action)

            combat_views.append((combat_view, meta["q_values"]))

        games.append((combat_view.character.health_current, combat_views))

    return games


def display_game(game: tuple[int, list]) -> None:
    for combat_view, probs in game[1]:
        draw_combat(combat_view)

        prob_array = probs.numpy().flatten()

        if combat_view.effect is None:
            cards_prob = prob_array[: len(combat_view.hand)]
            monster_prob = prob_array[2 * MAX_SIZE_HAND]
            end_turn_prob = prob_array[2 * MAX_SIZE_HAND + 1]
        else:
            cards_prob = prob_array[MAX_SIZE_HAND : MAX_SIZE_HAND + len(combat_view.hand)]
            monster_prob = prob_array[2 * MAX_SIZE_HAND]
            end_turn_prob = prob_array[2 * MAX_SIZE_HAND + 1]

        print(
            f"CARDS:{_format_array_floats(cards_prob)} / "
            f"MONST:{monster_prob:.4f} / "
            f"ENDTN:{end_turn_prob:.4f}"
        )


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
    model = DeepQNetwork(**config["model"])
    model.load_state_dict(torch.load(f"{exp_path}/model.pth"))

    # Instance policy
    policy = PolicyQMax(model, torch.device(device))
    games = run_simulation(policy, num_games=num_games)

    # Sort games based on criterion
    if sort_by == "health":
        games.sort(key=lambda x: x[0])
        click.echo("Games sorted by final health (lowest to highest)")
    else:
        games.sort(key=lambda x: -1 * len(x[1]))
        click.echo("Games sorted by game length (longest to shortest)")

    # Display
    display_game(games[0])


if __name__ == "__main__":
    main()
