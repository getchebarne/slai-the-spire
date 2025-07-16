import random

from src.game.core.effect import Effect
from src.game.core.effect import EffectSelectionType
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.engine.process_effect.registry import REGISTRY_EFFECT_TYPE_PROCESS_EFFECT
from src.game.entity.manager import EntityManager
from src.game.state import GameState
from src.game.types_ import EffectQueue


class EffectNeedsInputTargets(Exception):
    pass


def add_to_bot(effect_queue: EffectQueue, *effects: Effect) -> None:
    for effect in effects:
        effect_queue.append(effect)


def add_to_top(effect_queue: EffectQueue, *effects: Effect) -> None:
    for effect in reversed(effects):
        effect_queue.appendleft(effect)


def _resolve_effect_target_type(
    effect_target_type: EffectTargetType, entity_manager: EntityManager, id_source: int
) -> list[int]:
    if effect_target_type == EffectTargetType.CARD_IN_HAND:
        return entity_manager.id_cards_in_hand.copy()

    if effect_target_type == EffectTargetType.CARD_REWARD:
        return entity_manager.id_card_reward.copy()

    if effect_target_type == EffectTargetType.CARD_TARGET:
        return [entity_manager.id_card_target]

    if effect_target_type == EffectTargetType.CHARACTER:
        return [entity_manager.id_character]

    if effect_target_type == EffectTargetType.MONSTER:
        return entity_manager.id_monsters.copy()

    if effect_target_type == EffectTargetType.MAP_NODE:
        # TODO: improve
        if entity_manager.id_map_node_active is None:
            return [
                id_node
                for id_node in enumerate(entity_manager.id_map_nodes[0])
                if id_node is not None
            ]

        map_node_active = entity_manager.map_node_active
        y_next = map_node_active.y + 1
        return [entity_manager.id_map_nodes[y_next][x] for x in map_node_active.x_next]

    if effect_target_type == EffectTargetType.SOURCE:
        return [id_source]

    raise ValueError(f"Unsupported effect target type: {effect_target_type}")


def _resolve_effect_selection_type(
    effect_selection_type: EffectSelectionType, id_queries: list[int], effect_id_target: int | None
) -> list[int]:
    if effect_selection_type == EffectSelectionType.RANDOM:
        if id_queries:
            return [random.choice(id_queries)]

        return []

    if effect_selection_type == EffectSelectionType.INPUT:
        # TODO: make more readable?
        if effect_id_target is None:
            # Verify if we need to prompt the player to select from query entities
            # or if no selection is needed
            if len(id_queries) > 1:
                raise EffectNeedsInputTargets

            return id_queries

        return [effect_id_target]

    raise ValueError(f"Unsupported effect selection type: {effect_selection_type}")


def process_effect_queue(game_state: GameState) -> None:
    entity_manager = game_state.entity_manager
    effect_queue = game_state.effect_queue

    while effect_queue:
        effect = effect_queue.popleft()
        if effect.type == EffectType.GAME_END:
            add_to_top(effect_queue, effect)
            return

        id_source = effect.id_source
        id_target = effect.id_target

        if id_target is None:
            if effect.target_type is None:
                # Assign a list with a single `None` target so the effect is applied once
                id_targets = [None]

            else:
                # Get effect's query entities
                id_queries = _resolve_effect_target_type(
                    effect.target_type, entity_manager, id_source
                )

                # Select from those entities
                if effect.selection_type is None:
                    id_targets = id_queries

                else:
                    try:
                        id_targets = _resolve_effect_selection_type(
                            effect.selection_type, id_queries, effect.id_target
                        )

                    except EffectNeedsInputTargets:
                        # Need to wait for player to select the effect's target. Put effect back
                        # into queue at position 0 and return id_queries to tag them as selectable
                        add_to_top(effect_queue, effect)

                        return

        else:
            # TODO: could Effect have multiple target entities when created? think
            id_targets = [id_target]

        if effect.type == EffectType.COMBAT_END:
            # Clear effect queue before entering the room. This clears "ghost" effects that may
            # remain in the queue after the combat is over (e.g., draw and discard effects after
            # killing the last monster w/ "Dagger Throw")
            game_state.effect_queue.clear()

        for id_target in id_targets:
            # Process the effect & get new effects to add to the queue
            effects_bot, effects_top = REGISTRY_EFFECT_TYPE_PROCESS_EFFECT[effect.type](
                entity_manager,
                value=effect.value,
                id_source=id_source,
                id_target=id_target,
                ascension_level=game_state.ascension_level,
            )

            # Add new effects to the queue
            add_to_bot(effect_queue, *effects_bot)
            add_to_top(effect_queue, *effects_top)
