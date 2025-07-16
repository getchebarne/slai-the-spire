from dataclasses import replace

from src.game.draw_3.color import BG_CYAN
from src.game.draw_3.color import BG_GRAY
from src.game.draw_3.color import BG_RED
from src.game.draw_3.color import FG_BLACK
from src.game.draw_3.grid import Grid
from src.game.draw_3.grid import init_grid
from src.game.draw_3.grid import paste_grid
from src.game.draw_3.grid import put_border
from src.game.draw_3.grid import put_str
from src.game.entity.actor import EntityActor
from src.game.entity.actor import ModifierType
from src.game.entity.character import EntityCharacter
from src.game.entity.monster import EntityMonster
from src.game.entity.monster import Intent
from src.game.utils import get_corrected_intent_damage


MODIFIER_TYPE_ABBV = {
    ModifierType.MODE_SHIFT: "MSH",
    ModifierType.RITUAL: "RIT",
    ModifierType.SHARP_HIDE: "SHD",
    ModifierType.SPORE_CLOUD: "SPC",
    ModifierType.STRENGTH: "STR",
    ModifierType.WEAK: "WEK",
    ModifierType.VULNERABLE: "VLN",
}


def get_grid_character(
    character: EntityCharacter, height: int, width: int, margin_bar: int, margin_name: int
) -> Grid:
    return _get_grid_actor(character, height, width, margin_bar, margin_name)


def get_grid_monster(
    monster: EntityMonster,
    character: EntityCharacter,
    height: int,
    width: int,
    margin_bar: int,
    margin_name: int,
) -> Grid:
    grid = _get_grid_actor(monster, height, width, margin_bar, margin_name)

    # Intent
    intent = monster.moves[monster.move_name_current].intent
    if intent.damage is not None:
        intent = replace(
            intent,
            damage=get_corrected_intent_damage(intent.damage, monster, character),
        )

    intent_str = _get_str_intent(intent)
    grid = grid = put_str(grid, intent_str, y=len(grid) - 1, x=margin_name)

    return grid


def get_grid_actor_height(actor: EntityActor, height_base: int) -> int:
    if actor.modifier_map:
        return height_base + 1 + (len(actor.modifier_map) + 1) // 2

    return height_base


def _get_grid_actor(
    actor: EntityActor, height_base: int, width: int, margin_bar: int, margin_name: int
) -> Grid:
    # Intialize grid
    height = get_grid_actor_height(actor, height_base)
    grid = init_grid(height, width)

    # Health bar
    width_health_bar = width - 2 * margin_bar
    grid_bar_health = _get_bar_health(
        width_health_bar, actor.health_current, actor.health_max, actor.block_current
    )
    grid = paste_grid(grid_bar_health, grid, y=height_base // 2, x=(width - width_health_bar) // 2)
    grid = put_border(grid)

    # Name
    name_text = f" {actor.name} "
    if isinstance(actor, EntityCharacter):
        x = margin_name
    else:
        x = width - (len(name_text) - 1) - margin_bar

    grid = grid = put_str(grid, name_text, y=0, x=x)

    # Modifiers
    y_line = -1
    for idx, (modifier_type, modifier_data) in enumerate(actor.modifier_map.items()):
        text_modifier_type = MODIFIER_TYPE_ABBV[modifier_type]
        text_stacks = (
            str(modifier_data.stacks_current)
            if modifier_data.stacks_current >= 10
            else f" {modifier_data.stacks_current}"
        )
        text_gap = " " * (width_health_bar // 2 - len(text_modifier_type) - len(text_stacks) - 2)
        text = f"{text_modifier_type}:{text_gap}{text_stacks}"
        if idx % 2 == 0:
            text_line = text
            y_line += 1
        else:
            text_line = f"{text_line} │ {text}"

        grid = grid = put_str(grid, text_line, y=height_base - 1 + y_line, x=margin_bar)

    return grid


def _get_bar_health(width: int, health_current: int, health_max: int, block_current: int) -> Grid:
    grid = init_grid(1, width)

    text_health = f"{health_current}/{health_max}"
    text_block = "" if block_current == 0 else str(block_current)
    text_health_len = len(text_health)
    text_block_len = len(text_block)

    text_health_x = (width - text_health_len) // 2
    width_fill = int((health_current / health_max) * width)

    for x in range(width):
        # Get color
        if block_current > 0:
            color_code = FG_BLACK + BG_CYAN
        elif x < width_fill:
            color_code = FG_BLACK + BG_RED
        else:
            color_code = FG_BLACK + BG_GRAY

        # Get character
        if x < text_block_len:
            ch = text_block[x]
        elif x == text_block_len and block_current > 0:
            ch = "│"
        elif text_health_x <= x < text_health_x + text_health_len:
            ch = text_health[x - text_health_x]
        else:
            ch = " "

        grid = grid = put_str(grid, ch, y=0, x=x, color_code=color_code)

    return grid


def _get_str_intent(intent: Intent) -> str:
    str_ = ""

    if intent.damage is not None:
        if intent.instances is None:
            str_ = f"{str_}Hit {intent.damage}"

        else:
            str_ = f"{str_}Hit {intent.damage} x {intent.instances}"

    if intent.block:
        if str_ != "":
            str_ = f"{str_} & blocking"

        else:
            str_ = "Blocking"

    if intent.buff:
        if str_ != "":
            str_ = f"{str_} & buffing"

        else:
            str_ = "Buffing"

    return f" {str_} "


def _get_grid_actor_height(actor: EntityActor, height_base: int) -> int:
    return
