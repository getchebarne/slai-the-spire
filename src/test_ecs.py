from src.game.ecs.components import ActiveCardComponent
from src.game.ecs.components import BlockComponent
from src.game.ecs.components import CardCostComponent
from src.game.ecs.components import CardInHandComponent
from src.game.ecs.components import CharacterComponent
from src.game.ecs.components import EffectQueryComponentsComponent
from src.game.ecs.components import EffectSelectionTypeComponent
from src.game.ecs.components import EnergyComponent
from src.game.ecs.components import GainBlockEffectComponent
from src.game.ecs.components import HealthComponent
from src.game.ecs.components import MonsterComponent
from src.game.ecs.components import HasEffectsComponent
from src.game.ecs.components import NameComponent
from src.game.ecs.components import SelectionType
from src.game.ecs.components import TargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems import GainBlockSystem
from src.game.ecs.systems import PlayCardSystem
from src.game.ecs.systems import TargetEffectsSystem


systems = [PlayCardSystem(), TargetEffectsSystem(), GainBlockSystem()]

manager = ECSManager()

# Create a "GainBlock" effect
gain_block_entity_id = manager.create_entity(
    GainBlockEffectComponent(value=5),
    EffectQueryComponentsComponent([CharacterComponent]),
    EffectSelectionTypeComponent(SelectionType.NONE),
)
# Create "Defend" card in the hand
defend_entity_id = manager.create_entity(
    NameComponent("Defend"),
    CardCostComponent(1),
    HasEffectsComponent([gain_block_entity_id]),
    CardInHandComponent(position=0),
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

# Set the target entity
# manager.add_component(monster_entity_id, TargetComponent())

# Game loop
while True:
    print(manager.get_component_for_entity(char_entity_id, BlockComponent).current)
    for system in systems:
        system(manager)

    print(manager.get_component_for_entity(char_entity_id, BlockComponent).current)
    break
