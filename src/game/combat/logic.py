from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.context import GameContext
from src.game.combat.effect_queue import process_queue
from src.game.combat.utils import add_effects_to_bot
from src.game.combat.utils import card_requires_target
from src.game.combat.utils import get_active_card


# TODO: should be EffectPlayCard
def play_card(context: GameContext) -> None:
    active_card = get_active_card(context)
    if active_card.cost > context.energy.current:
        raise ValueError(f"Can't play card {active_card} with {context.energy.current} energy")

    # Subtract energy
    context.energy.current -= active_card.cost

    # Send card to discard pile
    context.hand.remove(active_card)
    context.discard_pile.add(active_card)


def _handle_action_select_card(context: GameContext, card_idx: int) -> None:
    context.active_card = context.hand[card_idx]


def _handle_action_select_monster(context: GameContext, monster_idx: int) -> None:
    context.card_target = context.monsters[monster_idx]


def is_game_over(context: GameContext) -> bool:
    return context.character.health.current <= 0 or all(
        [monster.health.current <= 0 for monster in context.monsters]
    )


def character_turn(context: GameContext, action: Action) -> None:
    if action.type == ActionType.SELECT_CARD:
        card = context.hand[action.index]
        card.is_active = True
        if card_requires_target(card):
            # Wait for player input
            return

    if action.type == ActionType.SELECT_MONSTER:
        context.card_target = context.monsters[action.index]

    # Play card. TODO: confirm?
    card = get_active_card(context)
    play_card(context)
    add_effects_to_bot(context, *card.effects)

    # Untag card
    card.is_active = False
    process_queue(context)
