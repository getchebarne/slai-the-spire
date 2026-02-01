import random
from collections import defaultdict

from src.game.const import MAX_SIZE_COMBAT_CARD_REWARD
from src.game.core.effect import Effect
from src.game.core.effect import EffectSelectionType
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.card import CardRarity
from src.game.entity.character import _CARD_REWARD_ROLL_OFFSET_BASE
from src.game.entity.manager import EntityManager
from src.game.factory.lib import FACTORY_LIB_CARD
from src.game.factory.lib import FactoryCard


_CARD_REWARD_ROLL_OFFSET_MIN = -40
_CHANCE_RARE = 3
_CHANCE_UNCOMMON = 40

# Create a mapping of `CardRarity` to factories to use for rolling card rewards
# TODO: improve?
FACTORY_LIB_CARD_RARITY: dict[CardRarity, dict[str, FactoryCard]] = defaultdict(dict)
for card_name, factory_card in FACTORY_LIB_CARD.items():
    card_rarity = factory_card(False).rarity
    FACTORY_LIB_CARD_RARITY[card_rarity][card_name] = factory_card


# TODO: upgraded cards
def process_effect_card_reward_roll(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    character = entity_manager.character

    card_name_rolled = []
    for _ in range(MAX_SIZE_COMBAT_CARD_REWARD):
        roll = random.randint(0, 98) + character.card_reward_roll_offset

        if roll < _CHANCE_RARE:
            card_rarity = CardRarity.RARE
            character.card_reward_roll_offset = _CARD_REWARD_ROLL_OFFSET_BASE

        elif roll < _CHANCE_UNCOMMON:
            card_rarity = CardRarity.UNCOMMON

        else:
            card_rarity = CardRarity.COMMON
            character.card_reward_roll_offset = max(
                character.card_reward_roll_offset - 1, _CARD_REWARD_ROLL_OFFSET_MIN
            )

        card_name = random.choice(list(FACTORY_LIB_CARD_RARITY[card_rarity].keys()))
        while card_name in card_name_rolled:
            card_name = random.choice(list(FACTORY_LIB_CARD_RARITY[card_rarity].keys()))

        card_name_rolled.append(card_name)
        card = FACTORY_LIB_CARD_RARITY[card_rarity][card_name](False)
        entity_manager.card_reward.append(card)

    return [
        Effect(
            EffectType.CARD_REWARD_SELECT,
            target_type=EffectTargetType.CARD_REWARD,
            selection_type=EffectSelectionType.INPUT,
        )
    ], []
