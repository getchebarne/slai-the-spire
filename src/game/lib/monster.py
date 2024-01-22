import sqlite3
from dataclasses import dataclass
from typing import Dict, List

from game.constants import DB_PATH


@dataclass
class MonsterEntry:
    base_health: int
    move_names: List[str]


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
monster_lib: Dict[str, MonsterEntry] = {}

# Fetch all rows. TODO: wrap in function
rows = cursor.fetchall()
for row in rows:
    monster_name = row["monster_name"]
    move_name = row["move_name"]

    # If monster_name is not in monster_lib, create a new MonsterEntry instance
    if monster_name not in monster_lib:
        monster_lib[monster_name] = MonsterEntry(
            base_health=row["base_health"], move_names=[move_name]
        )

    # If monster_name is in monster_lib, check if the move_name is in the monster's moves
    elif move_name not in monster_lib[monster_name].move_names:
        monster_lib[monster_name].move_names.append(move_name)
