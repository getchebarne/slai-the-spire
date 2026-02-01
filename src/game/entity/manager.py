from dataclasses import dataclass, field

from src.game.entity.card import EntityCard
from src.game.entity.character import EntityCharacter
from src.game.entity.energy import EntityEnergy
from src.game.entity.map_node import EntityMapNode
from src.game.entity.monster import EntityMonster


@dataclass
class EntityManager:
    character: EntityCharacter | None = None
    monsters: list[EntityMonster] = field(default_factory=list)
    energy: EntityEnergy | None = None

    # Permanent deck (persists across combats)
    deck: list[EntityCard] = field(default_factory=list)

    # Combat piles (transient, cleared after combat)
    draw_pile: list[EntityCard] = field(default_factory=list)
    hand: list[EntityCard] = field(default_factory=list)
    disc_pile: list[EntityCard] = field(default_factory=list)
    exhaust_pile: list[EntityCard] = field(default_factory=list)

    # Card reward options after combat
    card_reward: list[EntityCard] = field(default_factory=list)

    # Currently selected card (for targeting)
    card_active: EntityCard | None = None

    # Target of the active card (monster being targeted)
    card_target: EntityMonster | None = None

    # Map
    map_nodes: list[list[EntityMapNode | None]] = field(default_factory=list)
    map_node_active: EntityMapNode | None = None
    map_node_boss: EntityMapNode | None = None
