import sqlite3
from dataclasses import dataclass
from typing import Any, Dict

from game.constants import DB_PATH


@dataclass
class CharacterEntry:
    base_health: int
    start_relic_name: str


# Connect to the SQLite database
connection = sqlite3.connect(DB_PATH)
connection.row_factory = sqlite3.Row

# Create a cursor object
cursor = connection.cursor()

# Execute database query
cursor.execute("SELECT * FROM CharacterLib")

# Initialize card library. The card library is implemented as a dictionary mapping card_name to a
# dictionary containing card information
char_lib: Dict[str, Dict[str, Any]] = {}

# Fetch all rows. TODO: wrap in function
rows = cursor.fetchall()
for row in rows:
    char_lib[row["char_name"]] = CharacterEntry(
        base_health=row["base_health"], start_relic_name=row["start_relic_name"]
    )
