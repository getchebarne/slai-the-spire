from game.battle.comm import BattleView


class BattleDrawer:
    def __call__(self, battle_view: BattleView) -> None:
        print(battle_view.monsters)
        print(battle_view.char)
        print(battle_view.hand)
        print("-" * 100)
