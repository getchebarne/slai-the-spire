from typing import Any

import numpy as np
import torch
import yaml
from src.agents.a2c.encode import encode_combat_view
from src.agents.a2c.models.actor import Actor
from src.agents.a2c.models.actor import action_idx_to_action
from src.agents.a2c.models.actor import get_valid_action_mask

from src.game.combat.constant import MAX_SIZE_HAND
from src.game.combat.create import create_combat_state
from src.game.combat.drawer import draw_combat
from src.game.combat.main import start_combat
from src.game.combat.main import step
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


BASE_PATH = "/Users/getchebarne/Desktop/slai-the-spire/experiments"


def format_probs(probs_array: np.ndarray, precision: int = 4) -> str:
    return np.array2string(
        probs_array,
        precision=precision,
        suppress_small=True,
        formatter={"float_kind": lambda x: f"{x:.{precision}f}"},
    )


def load_model(exp_name: str) -> tuple[Actor, dict[str, Any]]:
    exp_path = f"{BASE_PATH}/{exp_name}"

    with open(f"{exp_path}/config.yml", "r") as file:
        config = yaml.safe_load(file)

    model = Actor(config["model_actor"]["dim_card"])

    model.load_state_dict(torch.load(f"{exp_path}/model_actor.pth"))

    return model, config


def run_simulation(
    model: Actor, num_games: int = 250, device: torch.device = torch.device("cpu")
) -> list[tuple[int, list]]:
    games = []

    for _ in range(num_games):
        cs = create_combat_state()
        start_combat(cs)

        combat_views = []
        while not is_game_over(cs.entity_manager):
            combat_view = view_combat(cs)

            valid_action_mask = torch.tensor(
                get_valid_action_mask(combat_view), dtype=torch.bool, device=device
            )
            encoding, index_mapping = encode_combat_view(combat_view, device)
            with torch.no_grad():
                prob = model(encoding, valid_action_mask)

            action_idx = torch.argmax(prob).item()
            action = action_idx_to_action(action_idx, combat_view, index_mapping)

            step(cs, action)

            prob_c = prob.clone()
            for idx_new, idx_old in index_mapping.items():
                prob_c[idx_old] = prob[idx_new]
                prob_c[idx_old + MAX_SIZE_HAND] = prob[idx_new + MAX_SIZE_HAND]

            combat_views.append((combat_view, prob_c, valid_action_mask))

        games.append((combat_view.character.health_current, combat_views))

    games.sort(key=lambda x: x[0])

    return games


def display_game(game: tuple[int, list]) -> None:
    for combat_view, probs, _ in game[1]:
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
    exp_name = "a2c/jaw/newera"
    device = torch.device("cpu")

    model, config = load_model(exp_name)
    model.to(device)

    games = run_simulation(model, num_games=100, device=device)

    worst_game = games[0]
    display_game(worst_game)
