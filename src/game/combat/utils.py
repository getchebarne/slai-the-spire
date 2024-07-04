from src.game.combat.factories import defend
from src.game.combat.factories import dummy
from src.game.combat.factories import energy
from src.game.combat.factories import silent
from src.game.combat.factories import strike
from src.game.combat.state import Card
from src.game.combat.state import Effect
from src.game.combat.state import EffectTargetType
from src.game.combat.state import GameState


def add_effects_to_bot(state: GameState, *effects: Effect) -> None:
    state.effect_queue.extend(effects)


def add_effects_to_top(state: GameState, *effects: Effect) -> None:
    state.effect_queue.extendleft(effects[::-1])


def card_requires_target(card: Card) -> bool:
    for effect in card.effects:
        if effect.target_type == EffectTargetType.CARD_TARGET:
            return True

    return False


def is_game_over(state: GameState) -> bool:
    return state.get_character().health.current <= 0 or all(
        [monster.health.current <= 0 for monster in state.get_monsters()]
    )


def new_game() -> GameState:
    # Create GameState instance
    state = GameState()

    # Fill
    # TODO: create functions for this
    state.character_id = state.create_entity(silent())
    state.monster_ids = [state.create_entity(dummy())]
    state.energy_id = state.create_entity(energy())
    state.card_in_deck_ids = {
        state.create_entity(strike()),
        state.create_entity(strike()),
        state.create_entity(strike()),
        state.create_entity(strike()),
        state.create_entity(strike()),
        state.create_entity(defend()),
        state.create_entity(defend()),
        state.create_entity(defend()),
        state.create_entity(defend()),
        state.create_entity(defend()),
        # state.create_entity(survivor()),
        # state.create_entity(neutralize()),
    }

    return state
