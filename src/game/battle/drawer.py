from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from game.battle.engine import BattleView


class BattleDrawer:
    def __call__(self, battle_view: BattleView) -> None:
        print(battle_view.monsters)
        print(battle_view.char)
        print(battle_view.hand)
        print("-" * 100)
