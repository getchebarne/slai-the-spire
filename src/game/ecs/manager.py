from itertools import count
from typing import Iterator, Optional

from src.game.ecs.components.base import BaseComponent


# TODO: implement cache functions
class ECSManager:
    def __init__(self):
        self._components: dict[type[BaseComponent], set[int]] = dict()
        self._entities: dict[int, dict[type[BaseComponent], BaseComponent]] = dict()
        self._dead_entities: set[int] = set()  # TODO: unused for now
        self._entity_count: count = count(start=1, step=1)

    def create_entity(self, *components: BaseComponent) -> int:
        # Get the next entity identifier
        entity = next(self._entity_count)

        # Create a new entry in the entities dictionary
        if entity not in self._entities:
            self._entities[entity] = {}
        else:
            raise ValueError(f"Entity {entity} already exists")

        for component in components:
            component_type = type(component)

            # Check if the component type is already in the database. If not, create a new entry
            if component_type not in self._components:
                self._components[component_type] = set()

            # Add the entity to the component's set
            self._components[component_type].add(entity)

            # Add the component to the entity's dictionary
            self._entities[entity][component_type] = component

        return entity

    def destroy_entity(self, entity: int) -> None:
        # Remove the entity from the entities dictionary
        self._entities.pop(entity)

        # Remove the entity from all component sets
        for component_type in self._components:
            self._components[component_type].discard(entity)

    def get_entity(self, entity: int) -> dict[type[BaseComponent], BaseComponent]:
        return self._entities[entity]

    def get_component_for_entity(
        self, entity: int, component_type: type[BaseComponent]
    ) -> Optional[BaseComponent]:
        # TODO: improve error messages
        try:
            components = self._entities[entity]
        except KeyError as e:
            raise KeyError(f"{e}: Entity {entity} does not exist.")

        return components.get(component_type, None)

    def get_component(
        self, component_type: type[BaseComponent]
    ) -> Iterator[tuple[int, BaseComponent]]:
        for entity in self._components.get(component_type, []):
            yield entity, self._entities[entity][component_type]

    def get_components(
        self, *component_types: type[BaseComponent]
    ) -> Iterator[tuple[int, list[BaseComponent]]]:
        try:
            for entity in set.intersection(*[self._components[ct] for ct in component_types]):
                yield entity, [self._entities[entity][ct] for ct in component_types]
        except KeyError:
            pass

    def destroy_component(self, component_type: type[BaseComponent]) -> None:
        if component_type in self._components:
            self._components.pop(component_type)

        for entity, entity_components in self._entities.items():
            if component_type in entity_components:
                entity_components.pop(component_type)

    def add_component(self, entity: int, component_instance: BaseComponent) -> None:
        if type(component_instance) is type:
            raise ValueError("you're stupid")

        # Get the component's type
        component_type = type(component_instance)

        # Check if the component type is already in the database. If not, create a new entry
        if component_type not in self._components:
            self._components[component_type] = set()

        # Add the entity to the component's set
        self._components[component_type].add(entity)

        # Add the component to the entity's dictionary
        self._entities[entity][component_type] = component_instance

    def remove_component(self, entity: int, component_type: type[BaseComponent]) -> BaseComponent:
        # Remove the entity from the component's set
        self._components[component_type].discard(entity)

        # If the component doesn't have any entities associated to it, remove it
        if not self._components[component_type]:
            del self._components[component_type]

        # Remove the component from the entity's dictionary and return it
        return self._entities[entity].pop(component_type)
