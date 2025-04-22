from typing import Callable, TypeAlias

from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.engine.process_effect.block_gain import process_effect_block_gain
from src.game.engine.process_effect.block_reset import process_effect_block_reset
from src.game.engine.process_effect.card_active_clear import process_effect_card_active_clear
from src.game.engine.process_effect.card_active_set import process_effect_card_active_set
from src.game.engine.process_effect.card_discard import process_effect_card_discard
from src.game.engine.process_effect.card_draw import process_effect_card_draw
from src.game.engine.process_effect.card_play import process_effect_card_play
from src.game.engine.process_effect.card_shuffle_deck_into_draw_pile import \
    process_effect_shuffle_deck_into_draw_pile
from src.game.engine.process_effect.damage_deal import process_effect_damage_deal
from src.game.engine.process_effect.end_turn import process_effect_end_turn
from src.game.engine.process_effect.energy_gain import process_effect_energy_gain
from src.game.engine.process_effect.energy_loss import process_effect_energy_loss
from src.game.engine.process_effect.health_loss import process_effect_health_loss
from src.game.engine.process_effect.modifier_ritual_gain import process_effect_modifier_ritual_gain
from src.game.engine.process_effect.modifier_strength_gain import \
    process_effect_modifier_strength_gain
from src.game.engine.process_effect.modifier_tick import process_effect_modifier_tick
from src.game.engine.process_effect.modifier_vulnerable_gain import \
    process_effect_modifier_vulnerable_gain
from src.game.engine.process_effect.modifier_weak_gain import process_effect_modifier_weak_gain
from src.game.engine.process_effect.monster_move_update import process_effect_monster_move_update
from src.game.engine.process_effect.target_card_clear import process_effect_target_card_clear
from src.game.engine.process_effect.target_card_set import process_effect_target_card_set
from src.game.engine.process_effect.target_effect_clear import process_effect_target_effect_clear
from src.game.engine.process_effect.target_effect_set import process_effect_target_effect_set
from src.game.entity.manager import EntityManager


ProcessEffect: TypeAlias = Callable[
    [EntityManager, Effect],
    tuple[list[Effect], list[Effect]],
]

REGISTRY_EFFECT_TYPE_PROCESS_EFFECT: dict[EffectType, ProcessEffect] = {
    EffectType.BLOCK_GAIN: process_effect_block_gain,
    EffectType.BLOCK_RESET: process_effect_block_reset,
    EffectType.CARD_ACTIVE_CLEAR: process_effect_card_active_clear,
    EffectType.CARD_ACTIVE_SET: process_effect_card_active_set,
    EffectType.CARD_DISCARD: process_effect_card_discard,
    EffectType.CARD_DRAW: process_effect_card_draw,
    EffectType.CARD_DRAW: process_effect_card_draw,
    EffectType.CARD_PLAY: process_effect_card_play,
    EffectType.CARD_SHUFFLE_DECK_INTO_DRAW_PILE: process_effect_shuffle_deck_into_draw_pile,
    EffectType.DAMAGE_DEAL: process_effect_damage_deal,
    EffectType.END_TURN: process_effect_end_turn,
    EffectType.ENERGY_GAIN: process_effect_energy_gain,
    EffectType.ENERGY_LOSS: process_effect_energy_loss,
    EffectType.HEALTH_LOSS: process_effect_health_loss,
    EffectType.MODIFIER_TICK: process_effect_modifier_tick,
    EffectType.MONSTER_MOVE_UPDATE: process_effect_monster_move_update,
    EffectType.MODIFIER_RITUAL_GAIN: process_effect_modifier_ritual_gain,
    EffectType.MODIFIER_STRENGTH_GAIN: process_effect_modifier_strength_gain,
    EffectType.MODIFIER_VULNERABLE_GAIN: process_effect_modifier_vulnerable_gain,
    EffectType.MODIFIER_WEAK_GAIN: process_effect_modifier_weak_gain,
    EffectType.TARGET_CARD_CLEAR: process_effect_target_card_clear,
    EffectType.TARGET_CARD_SET: process_effect_target_card_set,
    EffectType.TARGET_EFFECT_CLEAR: process_effect_target_effect_clear,
    EffectType.TARGET_EFFECT_SET: process_effect_target_effect_set,
}
