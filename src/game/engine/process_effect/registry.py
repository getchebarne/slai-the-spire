from typing import Callable, TypeAlias

from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.engine.process_effect.add_to_hand_shiv import process_effect_add_to_hand_shiv
from src.game.engine.process_effect.block_gain import process_effect_block_gain
from src.game.engine.process_effect.block_reset import process_effect_block_reset
from src.game.engine.process_effect.card_active_clear import process_effect_card_active_clear
from src.game.engine.process_effect.card_active_set import process_effect_card_active_set
from src.game.engine.process_effect.card_discard import process_effect_card_discard
from src.game.engine.process_effect.card_draw import process_effect_card_draw
from src.game.engine.process_effect.card_exhaust import process_effect_card_exhaust
from src.game.engine.process_effect.card_play import process_effect_card_play
from src.game.engine.process_effect.card_remove import process_effect_card_remove
from src.game.engine.process_effect.card_reward_roll import process_effect_card_reward_roll
from src.game.engine.process_effect.card_reward_select import process_effect_card_reward_select
from src.game.engine.process_effect.card_upgrade import process_effect_card_upgrade
from src.game.engine.process_effect.combat_end import process_effect_combat_end
from src.game.engine.process_effect.combat_start import process_effect_combat_start
from src.game.engine.process_effect.damage_deal import process_effect_damage_deal
from src.game.engine.process_effect.damage_deal_physical import process_effect_damage_deal_physical
from src.game.engine.process_effect.death import process_effect_death
from src.game.engine.process_effect.energy_gain import process_effect_energy_gain
from src.game.engine.process_effect.energy_loss import process_effect_energy_loss
from src.game.engine.process_effect.health_gain import process_effect_health_gain
from src.game.engine.process_effect.health_loss import process_effect_health_loss
from src.game.engine.process_effect.map_node_active_set import process_effect_map_node_active_set
from src.game.engine.process_effect.modifier_accuracy_gain import (
    process_effect_modifier_accuracy_gain,
)
from src.game.engine.process_effect.modifier_mode_shift_gain import (
    process_effect_modifier_mode_shift_gain,
)
from src.game.engine.process_effect.modifier_next_turn_block_gain import (
    process_effect_modifier_next_turn_block_gain,
)
from src.game.engine.process_effect.modifier_next_turn_energy_gain import (
    process_effect_modifier_next_turn_energy_gain,
)
from src.game.engine.process_effect.modifier_ritual_gain import process_effect_modifier_ritual_gain
from src.game.engine.process_effect.modifier_sharp_hide_gain import (
    process_effect_modifier_sharp_hide_gain,
)
from src.game.engine.process_effect.modifier_sharp_hide_loss import (
    process_effect_modifier_sharp_hide_loss,
)
from src.game.engine.process_effect.modifier_strength_gain import (
    process_effect_modifier_strength_gain,
)
from src.game.engine.process_effect.modifier_tick import process_effect_modifier_tick
from src.game.engine.process_effect.modifier_vulnerable_gain import (
    process_effect_modifier_vulnerable_gain,
)
from src.game.engine.process_effect.modifier_weak_gain import process_effect_modifier_weak_gain
from src.game.engine.process_effect.monster_move_update import process_effect_monster_move_update
from src.game.engine.process_effect.room_enter import process_effect_room_enter
from src.game.engine.process_effect.target_card_clear import process_effect_target_card_clear
from src.game.engine.process_effect.target_card_set import process_effect_target_card_set
from src.game.engine.process_effect.turn_end import process_effect_turn_end
from src.game.engine.process_effect.turn_start import process_effect_turn_start
from src.game.engine.process_effect.modifier_blur_gain import process_effect_modifier_blur_gain
from src.game.entity.manager import EntityManager


ProcessEffect: TypeAlias = Callable[
    [EntityManager, Effect],
    tuple[list[Effect], list[Effect]],
]

REGISTRY_EFFECT_TYPE_PROCESS_EFFECT: dict[EffectType, ProcessEffect] = {
    EffectType.ADD_TO_HAND_SHIV: process_effect_add_to_hand_shiv,
    EffectType.BLOCK_GAIN: process_effect_block_gain,
    EffectType.BLOCK_RESET: process_effect_block_reset,
    EffectType.CARD_ACTIVE_CLEAR: process_effect_card_active_clear,
    EffectType.CARD_ACTIVE_SET: process_effect_card_active_set,
    EffectType.CARD_DISCARD: process_effect_card_discard,
    EffectType.CARD_DRAW: process_effect_card_draw,
    EffectType.CARD_EXHAUST: process_effect_card_exhaust,
    EffectType.CARD_PLAY: process_effect_card_play,
    EffectType.CARD_REMOVE: process_effect_card_remove,
    EffectType.CARD_REWARD_ROLL: process_effect_card_reward_roll,
    EffectType.CARD_REWARD_SELECT: process_effect_card_reward_select,
    EffectType.CARD_UPGRADE: process_effect_card_upgrade,
    EffectType.COMBAT_END: process_effect_combat_end,
    EffectType.COMBAT_START: process_effect_combat_start,
    EffectType.DAMAGE_DEAL: process_effect_damage_deal,
    EffectType.DAMAGE_DEAL_PHYSICAL: process_effect_damage_deal_physical,
    EffectType.DEATH: process_effect_death,
    EffectType.ENERGY_GAIN: process_effect_energy_gain,
    EffectType.ENERGY_LOSS: process_effect_energy_loss,
    EffectType.HEALTH_GAIN: process_effect_health_gain,
    EffectType.HEALTH_LOSS: process_effect_health_loss,
    EffectType.MAP_NODE_ACTIVE_SET: process_effect_map_node_active_set,
    EffectType.MODIFIER_ACCURACY_GAIN: process_effect_modifier_accuracy_gain,
    EffectType.MODIFIER_BLUR_GAIN: process_effect_modifier_blur_gain,
    EffectType.MODIFIER_MODE_SHIFT_GAIN: process_effect_modifier_mode_shift_gain,
    EffectType.MODIFIER_NEXT_TURN_BLOCK_GAIN: process_effect_modifier_next_turn_block_gain,
    EffectType.MODIFIER_NEXT_TURN_ENERGY_GAIN: process_effect_modifier_next_turn_energy_gain,
    EffectType.MODIFIER_SHARP_HIDE_GAIN: process_effect_modifier_sharp_hide_gain,
    EffectType.MODIFIER_SHARP_HIDE_LOSS: process_effect_modifier_sharp_hide_loss,
    EffectType.MODIFIER_TICK: process_effect_modifier_tick,
    EffectType.MONSTER_MOVE_UPDATE: process_effect_monster_move_update,
    EffectType.MODIFIER_RITUAL_GAIN: process_effect_modifier_ritual_gain,
    EffectType.MODIFIER_STRENGTH_GAIN: process_effect_modifier_strength_gain,
    EffectType.MODIFIER_VULNERABLE_GAIN: process_effect_modifier_vulnerable_gain,
    EffectType.MODIFIER_WEAK_GAIN: process_effect_modifier_weak_gain,
    EffectType.ROOM_ENTER: process_effect_room_enter,
    EffectType.TARGET_CARD_CLEAR: process_effect_target_card_clear,
    EffectType.TARGET_CARD_SET: process_effect_target_card_set,
    EffectType.TURN_END: process_effect_turn_end,
    EffectType.TURN_START: process_effect_turn_start,
}
