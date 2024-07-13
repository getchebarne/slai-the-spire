from src.game.combat.state import State


StateView = State


def view_state(state: State) -> StateView:
    return StateView(state)
