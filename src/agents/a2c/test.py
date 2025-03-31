from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.agents.a2c.encode import encode_combat_view
from src.agents.a2c.model import ActorCritic
from src.agents.a2c.utils import action_idx_to_action
from src.agents.a2c.utils import get_valid_action_mask
from src.game.combat.constant import MAX_HAND_SIZE
from src.game.combat.create import create_combat_state
from src.game.combat.drawer import draw_combat
from src.game.combat.main import start_combat
from src.game.combat.main import step
from src.game.combat.utils import is_game_over
from src.game.combat.view import view_combat


def format_probs(probs_array: np.ndarray, precision: int = 4) -> str:
    return np.array2string(
        probs_array,
        precision=precision,
        suppress_small=True,
        formatter={"float_kind": lambda x: f"{x:.{precision}f}"},
    )


def load_model(exp_name: str) -> tuple[ActorCritic, dict[str, Any]]:
    base_path = Path("/Users/getchebarne/Desktop/slai-the-spire/experiments")
    exp_path = base_path / exp_name

    with open(exp_path / "config.yml", "r") as file:
        config = yaml.safe_load(file)

    model = ActorCritic(config["model"]["dim_card"], config["model"]["num_heads"])

    model.load_state_dict(torch.load(exp_path / "model.pth"))

    return model, config


def run_simulation(
    model: ActorCritic, num_games: int = 250, device: torch.device = torch.device("cpu")
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

            with torch.no_grad():
                prob, _ = model(
                    encode_combat_view(combat_view, device),
                    valid_action_mask,
                )

            action_idx = torch.argmax(prob).item()
            action = action_idx_to_action(action_idx, combat_view)

            step(cs, action)

            combat_views.append((combat_view, prob, valid_action_mask))

        games.append((combat_view.character.health_current, combat_views))

    games.sort(key=lambda x: x[0])

    return games


def display_game(game: tuple[int, list]) -> None:
    for combat_view, probs, _ in game[1]:
        draw_combat(combat_view)

        prob_array = probs.numpy().flatten()

        if combat_view.effect is None:
            cards_prob = prob_array[: len(combat_view.hand)]
            monster_prob = prob_array[2 * MAX_HAND_SIZE]
            end_turn_prob = prob_array[2 * MAX_HAND_SIZE + 1]
        else:
            cards_prob = prob_array[MAX_HAND_SIZE : MAX_HAND_SIZE + len(combat_view.hand)]
            monster_prob = prob_array[2 * MAX_HAND_SIZE]
            end_turn_prob = prob_array[2 * MAX_HAND_SIZE + 1]

        print(
            f"CARDS:{format_probs(cards_prob)} / "
            f"MONST:{monster_prob:.4f} / "
            f"ENDTN:{end_turn_prob:.4f}"
        )


if __name__ == "__main__":
    exp_name = "jaw/a2c-mlp"
    device = torch.device("cpu")

    model, config = load_model(exp_name)
    model.to(device)

    games = run_simulation(model, num_games=250, device=device)

    worst_game = games[0]
    display_game(worst_game)
