import random
from typing import TYPE_CHECKING

from src.game.core.effect import Effect
from src.game.core.effect import EffectSelectionType
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.engine.process_effect.registry import REGISTRY_EFFECT_TYPE_PROCESS_EFFECT
from src.game.entity.manager import EntityManager
from src.game.state import GameState
from src.game.types_ import EffectQueue


if TYPE_CHECKING:
    from src.game.entity.base import EntityBase


class EffectNeedsInputTargets(Exception):
    pass


def add_to_bot(effect_queue: EffectQueue, *effects: Effect) -> None:
    for effect in effects:
        effect_queue.append(effect)


def add_to_top(effect_queue: EffectQueue, *effects: Effect) -> None:
    for effect in reversed(effects):
        effect_queue.appendleft(effect)


def _resolve_effect_target_type(
    effect_target_type: EffectTargetType,
    entity_manager: EntityManager,
    source: "EntityBase | None",
) -> list["EntityBase"]:
    if effect_target_type == EffectTargetType.CARD_IN_HAND:
        return entity_manager.hand.copy()

    if effect_target_type == EffectTargetType.CARD_REWARD:
        return entity_manager.card_reward.copy()

    if effect_target_type == EffectTargetType.CARD_TARGET:
        return [entity_manager.card_target]

    if effect_target_type == EffectTargetType.CHARACTER:
        return [entity_manager.character]

    if effect_target_type == EffectTargetType.MONSTER:
        return entity_manager.monsters.copy()

    if effect_target_type == EffectTargetType.MAP_NODE:
        if entity_manager.map_node_active is None:
            # Starting position: return all valid nodes in the first row
            return [node for node in entity_manager.map_nodes[0] if node is not None]

        y_next = entity_manager.map_node_active.y + 1
        return [entity_manager.map_nodes[y_next][x] for x in entity_manager.map_node_active.x_next]

    if effect_target_type == EffectTargetType.SOURCE:
        return [source]

    raise ValueError(f"Unsupported effect target type: {effect_target_type}")


def _resolve_effect_selection_type(
    effect_selection_type: EffectSelectionType,
    candidates: list["EntityBase"],
    effect_target: "EntityBase | None",
) -> list["EntityBase"]:
    if effect_selection_type == EffectSelectionType.RANDOM:
        if candidates:
            return [random.choice(candidates)]
        return []

    if effect_selection_type == EffectSelectionType.INPUT:
        if effect_target is None:
            # Check if we need to prompt the player to select from candidates
            num_target = 1
            if len(candidates) > num_target:
                raise EffectNeedsInputTargets
            return candidates

        return [effect_target]

    raise ValueError(f"Unsupported effect selection type: {effect_selection_type}")


def process_effect_queue(game_state: GameState) -> None:
    entity_manager = game_state.entity_manager
    effect_queue = game_state.effect_queue

    while effect_queue:
        effect = effect_queue.popleft()
        if effect.type == EffectType.GAME_END:
            add_to_top(effect_queue, effect)
            return

        source = effect.source
        target = effect.target

        if target is None:
            if effect.target_type is None:
                # Assign a list with a single `None` target so the effect is applied once
                targets = [None]
            else:
                # Get effect's candidate targets
                candidates = _resolve_effect_target_type(
                    effect.target_type, entity_manager, source
                )

                # Select from those candidates
                if effect.selection_type is None:
                    targets = candidates
                else:
                    try:
                        targets = _resolve_effect_selection_type(
                            effect.selection_type, candidates, effect.target
                        )
                    except EffectNeedsInputTargets:
                        # Need to wait for player to select the effect's target
                        add_to_top(effect_queue, effect)
                        return
        else:
            targets = [target]

        if effect.type == EffectType.COMBAT_END:
            # Clear effect queue before processing. This clears "ghost" effects that may
            # remain in the queue after combat is over (e.g., draw and discard effects after
            # killing the last monster w/ "Dagger Throw")
            game_state.effect_queue.clear()

        for target in targets:
            # Process the effect & get new effects to add to the queue
            effects_bot, effects_top = REGISTRY_EFFECT_TYPE_PROCESS_EFFECT[effect.type](
                entity_manager,
                value=effect.value,
                source=source,
                target=target,
                ascension_level=game_state.ascension_level,
            )

            # Add new effects to the queue
            add_to_bot(effect_queue, *effects_bot)
            add_to_top(effect_queue, *effects_top)
