from src.game.ecs.components import ActiveCardComponent
from src.game.ecs.components import BlockComponent
from src.game.ecs.components import CardCostComponent
from src.game.ecs.components import CardInHandComponent
from src.game.ecs.components import CharacterComponent
from src.game.ecs.components import DealDamageEffectComponent
from src.game.ecs.components import EffectQueryComponentsComponent
from src.game.ecs.components import EffectSelectionTypeComponent
from src.game.ecs.components import EnergyComponent
from src.game.ecs.components import GainBlockEffectComponent
from src.game.ecs.components import HasEffectsComponent
from src.game.ecs.components import HealthComponent
from src.game.ecs.components import MonsterComponent
from src.game.ecs.components import NameComponent
from src.game.ecs.components import SelectionType
from src.game.ecs.components import TargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems import GainBlockSystem
from src.game.ecs.systems import PlayCardSystem
from src.game.ecs.systems import TargetEffectsSystem
from src.game.ecs.systems import DealDamageSystem


systems = [PlayCardSystem(), TargetEffectsSystem(), GainBlockSystem(), DealDamageSystem()]

manager = ECSManager()

# Create a "GainBlock" effect
gain_block_entity_id = manager.create_entity(
    GainBlockEffectComponent(value=5),
    EffectQueryComponentsComponent([CharacterComponent]),
    EffectSelectionTypeComponent(SelectionType.NONE),
)
# Create a "DealDamage" effect
deal_damage_entity_id = manager.create_entity(
    DealDamageEffectComponent(value=6),
    EffectQueryComponentsComponent([MonsterComponent]),
    EffectSelectionTypeComponent(SelectionType.SPECIFIC),
)
# Create "Defend" card in the hand
defend_entity_id = manager.create_entity(
    NameComponent("Defend"),
    CardCostComponent(1),
    HasEffectsComponent([gain_block_entity_id]),
    CardInHandComponent(position=0),
)
# Create "Strike" card in the hand
strike_entity_id = manager.create_entity(
    NameComponent("Strike"),
    CardCostComponent(1),
    HasEffectsComponent([deal_damage_entity_id]),
    CardInHandComponent(position=1),
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
# Create the energy
energy_entity_id = manager.create_entity(EnergyComponent(3))

# Set the active card
manager.add_component(defend_entity_id, ActiveCardComponent())
# manager.add_component(strike_entity_id, ActiveCardComponent())

# Set the target entity
manager.add_component(monster_entity_id, TargetComponent())

# Game loop
while True:
    print(manager.get_component_for_entity(char_entity_id, BlockComponent).current)
    print(manager.get_component_for_entity(monster_entity_id, HealthComponent).current)
    for system in systems:
        system(manager)

    print(manager.get_component_for_entity(char_entity_id, BlockComponent).current)
    print(manager.get_component_for_entity(monster_entity_id, HealthComponent).current)
    break
