import random
from typing import Optional


class DummyAI:
    def next_move_name(self, current_move_name: Optional[str]) -> str:
        if current_move_name is None:
            # First move
            return random.choice(["Attack", "Defend"])

        if current_move_name == "Attack":
            return "Defend"

        elif current_move_name == "Defend":
            return "Attack"

        else:
            raise ValueError(f"Unknown move {current_move_name}")
