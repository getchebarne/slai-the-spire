from typing import Callable

from src.game.ai.dummy import get_move_name_dummy
from src.game.ai.fungi_beast import get_move_name_fungi_beast
from src.game.ai.jaw_worm import get_move_name_jaw_worm
from src.game.entity.monster import EntityMonster


AI_REGISTRY: dict[str, Callable[[EntityMonster], str]] = {
    "Dummy": get_move_name_dummy,
    "Jaw Worm": get_move_name_jaw_worm,
    "Fungi Beast": get_move_name_fungi_beast,
}
