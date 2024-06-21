from src.game.ecs.systems.ai_dummy import AIDummySystem
from src.game.ecs.systems.apply_modifier_delta import ApplyModifierDeltaSystem
from src.game.ecs.systems.apply_weak import ApplyWeakSystem
from src.game.ecs.systems.before_turn_end import BeforeTurnEndSystem
from src.game.ecs.systems.card_defend import CardDefendSystem
from src.game.ecs.systems.card_neutralize import CardNeutralizeSystem
from src.game.ecs.systems.card_strike import CardStrikeSystem
from src.game.ecs.systems.card_survivor import CardSurvivorSystem
from src.game.ecs.systems.create_modifier_weak import CreateModifierWeakSystem
from src.game.ecs.systems.deal_damage import DealDamageSystem
from src.game.ecs.systems.destroy_effect import DestroyEffectSystem
from src.game.ecs.systems.discard_card import DiscardCardSystem
from src.game.ecs.systems.dispatch_effect import DispatchEffectSystem
from src.game.ecs.systems.draw_card import DrawCardSystem
from src.game.ecs.systems.enable_input import EnableInputSystem
from src.game.ecs.systems.gain_block import GainBlockSystem
from src.game.ecs.systems.intent_dummy import IntentDummySystem
from src.game.ecs.systems.move_defend_dummy import MoveDummyDefendSystem
from src.game.ecs.systems.move_dummy_attack import MoveDummyAttackSystem
from src.game.ecs.systems.play_card import PlayCardSystem
from src.game.ecs.systems.process_action import ProcessActionSystem
from src.game.ecs.systems.process_monster_turn import ProcessMonsterTurnSystem
from src.game.ecs.systems.refill_energy import RefillEnergySystem
from src.game.ecs.systems.set_block_to_zero import SetBlockToZeroSystem
from src.game.ecs.systems.shuffle_deck_into_draw_pile import ShuffleDeckIntoDrawPileSystem
from src.game.ecs.systems.shuffle_discard_pile_into_draw_pile import \
    ShuffleDiscardPileIntoDrawPileSystem
from src.game.ecs.systems.tag_card_target_modifiers import TagCardTargetModifiersSystem
from src.game.ecs.systems.target_effect import TargetEffectSystem
from src.game.ecs.systems.turn_end import TurnEndSystem
from src.game.ecs.systems.turn_start import TurnStartSystem


ALL_SYSTEMS = [
    ProcessActionSystem(),
    TagCardTargetModifiersSystem(),
    TurnStartSystem(),
    ProcessMonsterTurnSystem(),
    BeforeTurnEndSystem(),
    TurnEndSystem(),
    AIDummySystem(),
    IntentDummySystem(),
    CardStrikeSystem(),
    CardDefendSystem(),
    CardNeutralizeSystem(),
    CardSurvivorSystem(),
    PlayCardSystem(),
    MoveDummyAttackSystem(),
    MoveDummyDefendSystem(),
    DispatchEffectSystem(),
    TargetEffectSystem(),
    CreateModifierWeakSystem(),
    ApplyModifierDeltaSystem(),
    ShuffleDeckIntoDrawPileSystem(),
    ShuffleDiscardPileIntoDrawPileSystem(),
    ApplyWeakSystem(),
    DealDamageSystem(),
    DiscardCardSystem(),
    DrawCardSystem(),
    GainBlockSystem(),
    RefillEnergySystem(),
    SetBlockToZeroSystem(),
    DestroyEffectSystem(),
    EnableInputSystem(),
]
