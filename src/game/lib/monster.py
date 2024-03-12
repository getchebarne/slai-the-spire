import importlib
import sqlite3
from dataclasses import dataclass

from game.constants import DB_PATH
from game.ai.base import BaseAI


@dataclass
class MonsterEntry:
    base_health: int
    ai: BaseAI


# Connect to the SQLite database
connection = sqlite3.connect(DB_PATH)
connection.row_factory = sqlite3.Row

# Create a cursor object
cursor = connection.cursor()

# Execute database query
cursor.execute("SELECT * FROM MonsterLib")

# Initialize monster library. The monster library is implemented as a dictionary mapping
# monster_name to a MonsterEntry instance
monster_lib = {}

# Fetch all rows. TODO: wrap in function
rows = cursor.fetchall()
for row in rows:
    monster_name = row["monster_name"]
    base_health = row["base_health"]

    # Get the monster's AI
    ai_module = importlib.import_module(f"game.ai.{monster_name.lower()}")
    ai = getattr(ai_module, f"{monster_name}AI")()

    # Add monster to monster_lib
    monster_lib[monster_name] = MonsterEntry(base_health=base_health, ai=ai)
