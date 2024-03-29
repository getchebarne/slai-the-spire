import random
from typing import Optional


class DummyAI:
    def next_move_name(self, current_move_name: Optional[str]) -> str:
        if current_move_name == "Attack":
            return "Defend"

        elif current_move_name == "Defend":
            return "Attack"

        else:
            raise ValueError(f"Unknown move {current_move_name}")

    def first_move_name(self) -> str:
        return random.choice(["Attack", "Defend"])
