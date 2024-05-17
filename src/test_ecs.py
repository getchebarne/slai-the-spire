from src.game.core.components import ActiveCardComponent
from src.game.core.components import BlockComponent
from src.game.core.components import CardInHandComponent
from src.game.core.components import CharacterComponent
from src.game.core.components import EffectsOnUseComponent
from src.game.core.components import HealthComponent
from src.game.core.components import MonsterComponent
from src.game.core.components import NameComponent
from src.game.core.components import TargetComponent
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.core.effect import SelectionType
from src.game.core.manager import ECSManager
from src.game.core.systems import ApplyEffectsSystem
from src.game.core.systems import PlayCardSystem
from src.game.core.systems import TargetEffectsSystem


systems = [PlayCardSystem(), TargetEffectsSystem(), ApplyEffectsSystem()]

manager = ECSManager()

# Create "Strike" card in the hand
strike_entity_id = manager.create_entity(
    CardInHandComponent(position=0),
    NameComponent(value="Strike"),
    EffectsOnUseComponent(
        effects=[Effect(EffectType.DAMAGE, 6, [MonsterComponent], SelectionType.SPECIFIC)]
    ),
)
# Create "Defend" card in the hand
defend_entity_id = manager.create_entity(
    CardInHandComponent(position=1),
    NameComponent(value="Defend"),
    EffectsOnUseComponent(
        effects=[Effect(EffectType.BLOCK, 5, [CharacterComponent], SelectionType.NONE)]
    ),
)
# Create a character
char_entity_id = manager.create_entity(
    NameComponent("Silent"),
    CharacterComponent(),
    HealthComponent(50),
    BlockComponent(),
)
# Create a monster
monster_entity_id = manager.create_entity(
    NameComponent("Dummy"),
    MonsterComponent(),
    HealthComponent(50),
    BlockComponent(),
)
# Set the active card
manager.add_component(defend_entity_id, ActiveCardComponent())

# Set the target entity
manager.add_component(monster_entity_id, TargetComponent())

# Game loop
while True:
    for system in systems:
        system(manager)
