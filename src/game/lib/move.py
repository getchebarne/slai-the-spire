import importlib
import sqlite3

from game.constants import DB_PATH
from game.logic.move.base import BaseMoveLogic

# Connect to the SQLite database
connection = sqlite3.connect(DB_PATH)
connection.row_factory = sqlite3.Row

# Create a cursor object
cursor = connection.cursor()

# Execute database query
cursor.execute("SELECT * FROM MonsterMoves")

# Initialize move library. The move library is implemented as a dictionary mapping
# (monster_name, move_name) to a BaseMoveLogic instance
move_lib: dict[str, BaseMoveLogic] = {}

# Fetch all rows. TODO: wrap in function
rows = cursor.fetchall()
for row in rows:
    monster_name = row["monster_name"]
    move_name = row["move_name"]

    # Get the move's logic
    logic_module = importlib.import_module(f"game.logic.move.{move_name.lower()}")
    move_logic = getattr(logic_module, f"{move_name}Logic")()

    # Add move to move_lib
    move_lib[(monster_name, move_name)] = move_logic
