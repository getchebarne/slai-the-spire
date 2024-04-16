import importlib
import sqlite3
from dataclasses import dataclass

from game.constants import DB_PATH
from game.logic.card.base import BaseCardLogic


@dataclass
class ModifierEntry:
    modifier_desc: str
    modifier_stacks_duration: bool
    modifier_stacks_counter: bool
    modifier_logic: BaseCardLogic


# Connect to the SQLite database
connection = sqlite3.connect(DB_PATH)
connection.row_factory = sqlite3.Row

# Create a cursor object
cursor = connection.cursor()

# Execute database query
cursor.execute("""SELECT * FROM Modifier""")

# Initialize modifier library. The modifier library is implemented as a dictionary mapping
# modifier_name to a an instance of ModidiferEntry
modifier_lib: dict[str, ModifierEntry] = {}

# Iterate over the rows of the query result
for row in cursor.fetchall():
    modifier_name = row["modifier_name"]

    # Get the modifier's logic
    logic_module = importlib.import_module(f"game.logic.modifier.{modifier_name.lower()}")
    modifier_logic = getattr(logic_module, f"{modifier_name}Logic")()

    # Create a ModifierEntry instance and add it to the modifier library
    modifier_lib[modifier_name] = ModifierEntry(
        modifier_desc=row["modifier_desc"],
        modifier_stacks_duration=row["stacks_duration"],
        modifier_stacks_counter=row["stacks_counter"],
        modifier_logic=modifier_logic,
    )
