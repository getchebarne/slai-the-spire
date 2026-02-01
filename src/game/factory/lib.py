from typing import Callable, TypeAlias, TypeVar, get_args, get_origin, get_type_hints

from src.game.entity.card import EntityCard
from src.game.entity.character import EntityCharacter
from src.game.entity.monster import EntityMonster
from src.game.types_ import AscensionLevel
from src.game.types_ import CardUpgraded


# Type variable to maintain input/output types
F = TypeVar("F", bound=Callable)
FactoryCard: TypeAlias = Callable[[CardUpgraded], EntityCard]
FactoryCharacter: TypeAlias = Callable[[AscensionLevel], tuple[EntityCharacter, list[EntityCard]]]
FactoryMonster: TypeAlias = Callable[
    [AscensionLevel], tuple[EntityMonster, Callable[[EntityMonster, AscensionLevel], str]]
]

FACTORY_LIB_CARD: dict[str, FactoryCard] = {}
FACTORY_LIB_CHARACTER: dict[str, FactoryCharacter] = {}
FACTORY_LIB_MONSTER: dict[str, FactoryMonster] = {}


def register_factory(name: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        # Get the return type from annotations
        return_type = get_type_hints(func).get("return")

        if return_type is EntityCard:
            FACTORY_LIB_CARD[name] = func

        elif get_origin(return_type) is tuple and get_args(return_type) == (
            (EntityCharacter, list[EntityCard])  # TODO: add starter relic
        ):
            FACTORY_LIB_CHARACTER[name] = func

        elif get_origin(return_type) is tuple and get_args(return_type) == (
            (EntityMonster, Callable[[EntityMonster], str])  # TODO: add starter relic
        ):
            FACTORY_LIB_MONSTER[name] = func

        else:
            raise TypeError(f"Unsupported return type {return_type} for factory {name}")

        return func

    return decorator
