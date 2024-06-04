from src.game.ecs.systems.deal_damage import DealDamageSystem
from src.game.ecs.systems.destroy_effect import DestroyEffectSystem
from src.game.ecs.systems.discard_card import DiscardCardSystem
from src.game.ecs.systems.dispatch_effect import DispatchEffectSystem
from src.game.ecs.systems.draw_card import DrawCardSystem
from src.game.ecs.systems.enable_input import EnableInputSystem
from src.game.ecs.systems.gain_block import GainBlockSystem
from src.game.ecs.systems.handle_input import HandleInputSystem
from src.game.ecs.systems.play_card import PlayCardSystem
from src.game.ecs.systems.process_monster_turn import ProcessMonsterTurnSystem
from src.game.ecs.systems.refill_energy import RefillEnergySystem
from src.game.ecs.systems.set_block_to_zero import SetBlockToZeroSystem
from src.game.ecs.systems.shuffle_deck_into_draw_pile import ShuffleDeckIntoDrawPileSystem
from src.game.ecs.systems.shuffle_discard_pile_into_draw_pile import (
    ShuffleDiscardPileIntoDrawPileSystem,
)
from src.game.ecs.systems.target_effect import TargetEffectSystem


ALL_SYSTEMS = [
    EnableInputSystem(),
    HandleInputSystem(),
    ProcessMonsterTurnSystem(),
    PlayCardSystem(),
    DispatchEffectSystem(),
    TargetEffectSystem(),
    ShuffleDeckIntoDrawPileSystem(),
    ShuffleDiscardPileIntoDrawPileSystem(),
    DealDamageSystem(),
    DiscardCardSystem(),
    DrawCardSystem(),
    GainBlockSystem(),
    RefillEnergySystem(),
    SetBlockToZeroSystem(),
    DestroyEffectSystem(),
]
