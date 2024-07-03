from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.state import GameState
from src.game.combat.effect_queue import process_queue
from src.game.combat.utils import add_effects_to_bot
from src.game.combat.utils import card_requires_target


# TODO: should be EffectPlayCard
def play_card(context: GameState) -> None:
    if context.active_card.cost > context.energy.current:
        raise ValueError(
            f"Can't play card {context.active_card} with {context.energy.current} energy"
        )

    # Subtract energy
    context.energy.current -= context.active_card.cost

    # Send card to discard pile
    context.hand.remove(context.active_card)
    context.discard_pile.add(context.active_card)


def _handle_action_select_card(context: GameState, card_idx: int) -> None:
    context.active_card = context.hand[card_idx]


def _handle_action_select_monster(context: GameState, monster_idx: int) -> None:
    context.card_target = context.monsters[monster_idx]


def is_game_over(context: GameState) -> bool:
    return context.character.health.current <= 0 or all(
        [monster.health.current <= 0 for monster in context.monsters]
    )


def character_turn(context: GameState, action: Action) -> None:
    if action.type == ActionType.SELECT_CARD:
        card = context.hand[action.index]
        context.active_card = card
        if card_requires_target(card):
            # Wait for player input
            return

    if action.type == ActionType.SELECT_MONSTER:
        context.card_target = context.monsters[action.index]

    if action.type == ActionType.END_TURN:
        return

    # Play card. TODO: confirm?
    play_card(context)
    add_effects_to_bot(context, *context.active_card.effects)
    process_queue(context)

    # Untag card
    context.active_card = None
