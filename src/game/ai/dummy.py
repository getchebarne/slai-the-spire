class DummyAI:
    def next_move_name(self, current_move_name: str) -> str:
        if current_move_name == "Attack":
            return "Defend"

        elif current_move_name == "Defend":
            return "Attack"

        else:
            raise ValueError(f"Unknown move {current_move_name}")
