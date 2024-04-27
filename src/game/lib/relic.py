import importlib
import sqlite3
from dataclasses import dataclass

from src.game.constants import DB_PATH
from src.game.logic.relic.base import BaseRelicLogic


@dataclass
class RelicEntry:
    relic_desc: str
    relic_rarity: str
    relic_logic: BaseRelicLogic


# Connect to the SQLite database
connection = sqlite3.connect(DB_PATH)
connection.row_factory = sqlite3.Row

# Create a cursor object
cursor = connection.cursor()

# Execute database query
cursor.execute("SELECT * FROM RelicLib")

# Initialize relic library. The relic library is implemented as a dictionary mapping relic_name to
# an instance of RelicEntry
relic_lib = {}
for row in cursor.fetchall():
    relic_name = row["relic_name"]

    # Get the relic's logic
    script_name = relic_name.lower().replace(" ", "_")
    relic_logic_name = f"{relic_name.title().replace(' ', '')}Logic"
    logic_module = importlib.import_module(f"src.game.logic.relic.{script_name}")
    relic_logic = getattr(logic_module, relic_logic_name)()

    # Create a RelicEntry instance and add it to the relic library
    relic_lib[relic_name] = RelicEntry(
        relic_desc=row["relic_desc"],
        relic_rarity=row["relic_rarity"],
        relic_logic=relic_logic,
    )
