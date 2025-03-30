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


if __name__ == "__main__":
    # Select experiment
    exp_name = "dummy/a2c-sym"

    # Load config
    with open(
        f"/Users/getchebarne/Desktop/slai-the-spire/experiments/{exp_name}/config.yml", "r"
    ) as file:
        config = yaml.safe_load(file)

    # TODO: reuse funcs defined in train.py
    model = ActorCritic(
        config["model"]["layer_sizes_shared"],
        config["model"]["layer_sizes_actor"],
        config["model"]["layer_sizes_critic"],
        config["model"]["dim_card"],
    )
    model.load_state_dict(
        torch.load(f"/Users/getchebarne/Desktop/slai-the-spire/experiments/{exp_name}/model.pth")
    )
    device = torch.device("cpu")

    num_games = 250
    games = []
    for _ in range(num_games):
        # Instance combat manager
        cs = create_combat_state()
        start_combat(cs)

        combat_views = []
        while not is_game_over(cs.entity_manager):
            # Get combat view and draw it on the terminal
            combat_view = view_combat(cs)

            # Get action from agent
            valid_action_mask = torch.tensor(
                get_valid_action_mask(combat_view), dtype=torch.bool, device=device
            )
            with torch.no_grad():
                prob, value = model(
                    encode_combat_view(combat_view, device),
                    valid_action_mask,
                )

            action_idx = torch.argmax(prob).item()
            action = action_idx_to_action(action_idx, combat_view)

            # Game step
            step(cs, action)

            combat_views.append((combat_view, prob, valid_action_mask))

        games.append((combat_view.character.health_current, combat_views))

    # Sort from lowest HP to highest
    games.sort(key=lambda x: x[0])

    game = games[0]
    for combat_view, probs, valid in game[1]:
        draw_combat(combat_view)
        if combat_view.effect is None:
            print(
                f"CARDS:{probs.numpy().flatten()[:len(combat_view.hand)]} / MONST:{probs.numpy().flatten()[2*MAX_HAND_SIZE]} / ENDTN:{probs.numpy().flatten()[2*MAX_HAND_SIZE+1]}"
            )
        else:
            print(
                f"CARDS:{probs.numpy().flatten()[MAX_HAND_SIZE:MAX_HAND_SIZE+len(combat_view.hand)]} / MONST:{probs.numpy().flatten()[2*MAX_HAND_SIZE]} / ENDTN:{probs.numpy().flatten()[2*MAX_HAND_SIZE+1]}"
            )
