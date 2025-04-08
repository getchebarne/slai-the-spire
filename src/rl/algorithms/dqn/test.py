from typing import Any

import numpy as np
import torch
import yaml

from src.game.combat.constant import MAX_SIZE_HAND
from src.game.combat.create import create_combat_state
from src.game.combat.drawer import draw_combat
from src.game.combat.main import start_combat
from src.game.combat.main import step
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat
from src.rl.models.dqn import DeepQNetwork
from src.rl.policies import PolicyQMax


BASE_PATH = "/Users/getchebarne/Desktop/slai-the-spire/experiments"


def format_probs(probs_array: np.ndarray, precision: int = 4) -> str:
    return np.array2string(
        probs_array,
        precision=precision,
        suppress_small=True,
        formatter={"float_kind": lambda x: f"{x:.{precision}f}"},
    )


def load_model(exp_name: str) -> tuple[DeepQNetwork, dict[str, Any]]:
    exp_path = f"{BASE_PATH}/{exp_name}"

    with open(f"{exp_path}/config.yml", "r") as file:
        config = yaml.safe_load(file)

    model = DeepQNetwork(config["model"]["dim_card"])

    model.load_state_dict(torch.load(f"{exp_path}/model.pth"))

    return model, config


def run_simulation(
    policy: PolicyQMax, num_games: int = 250, device: torch.device = torch.device("cpu")
) -> list[tuple[int, list]]:
    games = []

    for _ in range(num_games):
        cs = create_combat_state()
        start_combat(cs)

        combat_views = []
        while not is_game_over(cs.entity_manager):
            combat_view = view_combat(cs)

            action, meta = policy(combat_view)
            step(cs, action)

            combat_views.append((combat_view, meta["q_values"]))

        games.append((combat_view.character.health_current, combat_views))

    games.sort(key=lambda x: x[0])

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
            f"CARDS:{format_probs(cards_prob)} / "
            f"MONST:{monster_prob:.4f} / "
            f"ENDTN:{end_turn_prob:.4f}"
        )


if __name__ == "__main__":
    exp_name = "dqn/jaw/newera-ohe-3"
    device = torch.device("cpu")

    model, config = load_model(exp_name)
    policy = PolicyQMax(model, device)

    games = run_simulation(policy, num_games=100, device=device)

    worst_game = games[0]
    display_game(worst_game)
