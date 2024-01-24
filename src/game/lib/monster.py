import importlib
import sqlite3
from dataclasses import dataclass

from game.constants import DB_PATH
from game.logic.move.base import BaseMoveLogic


@dataclass
class MonsterEntry:
    base_health: int
    moves: dict[str, BaseMoveLogic]


# Connect to the SQLite database
connection = sqlite3.connect(DB_PATH)
connection.row_factory = sqlite3.Row

# Create a cursor object
cursor = connection.cursor()

# Execute database query
cursor.execute(
    """
    SELECT
        *
    FROM
        MonsterLib
        JOIN MonsterMoves USING (monster_name)
    """
)
# Initialize monster library. The monster library is implemented as a dictionary mapping
# monster_name to a MonsterEntry instance
monster_lib = {}

# Fetch all rows. TODO: wrap in function
rows = cursor.fetchall()
for row in rows:
    monster_name = row["monster_name"]
    move_name = row["move_name"]

    # Get the move's logic
    logic_module = importlib.import_module(f"game.logic.move.{move_name.lower()}")
    move_logic = getattr(logic_module, f"{move_name}Logic")()

    # If monster_name is not in monster_lib, create a new MonsterEntry instance
    if monster_name not in monster_lib:
        monster_lib[monster_name] = MonsterEntry(
            base_health=row["base_health"], moves={move_name: move_logic}
        )

    # If monster_name is in monster_lib, add move_name to the monster's moves
    else:
        monster_lib[monster_name].moves[move_name] = move_logic
