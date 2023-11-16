from __future__ import annotations

import os
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from game.battle.engine import BattleView


H_LENGTH = 100


def align_right(text: str) -> str:
    columns, _ = os.get_terminal_size(0)
    return "\n".join([line.rjust(H_LENGTH) for line in text.split("\n")])


class BattleDrawer:
    def __call__(self, battle_view: BattleView) -> None:
        print(align_right(battle_view.monsters.__str__()))
        print(battle_view.char)
        print(battle_view.char.energy, battle_view.hand)
        print("-" * H_LENGTH)
