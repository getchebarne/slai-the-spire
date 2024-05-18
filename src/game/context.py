from collections import defaultdict
from typing import Generator, Optional

from src.game.core.state import BattleState
from src.game.ecs.components import BaseComponent


STARTING_ENERGY = 3


class Context:
    def __init__(self, state: BattleState = BattleState.NONE):
        # This is a dictionary of dictionaries. The outer dictionary is keyed by the type of the
        # component, and the inner dictionary is keyed by the entity ID. This allows us to quickly
        # look up components by type and entity ID
        self.components = defaultdict(dict)

        # This is the current state of the game
        self.state = state

    def get_component(self, entity_id: int, component_type: type) -> Optional[BaseComponent]:
        return self.components[component_type].get(entity_id)

    def add_component(self, entity_id: int, component: BaseComponent):
        self.components[type(component)][entity_id] = component

    def remove_component(self, entity_id: int, component_type: type):
        self.components[component_type].pop(entity_id, None)

    def get_entities_with_component(self, component_type: type) -> Generator[int, None, None]:
        for entity_id, _ in self.components[component_type].items():
            yield entity_id

    def get_entities_with_components(
        self, component_types: list[type]
    ) -> Generator[int, None, None]:
        entity_ids = set(self.components[component_types[0]].keys())
        for component_type in component_types[1:]:
            entity_ids.intersection_update(self.components[component_type].keys())
        for entity_id in entity_ids:
            yield entity_id
