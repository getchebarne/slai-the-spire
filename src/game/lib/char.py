import sqlite3
from dataclasses import dataclass

from src.game.constants import DB_PATH


@dataclass
class CharEntry:
    base_health: int
    starter_relic_name: str


# Connect to the SQLite database
connection = sqlite3.connect(DB_PATH)
connection.row_factory = sqlite3.Row

# Create a cursor object
cursor = connection.cursor()

# Execute database query
cursor.execute("SELECT * FROM CharacterLib")

# Initialize character library. The character library is implemented as a dictionary mapping
# char_name to an instance of CharEntry
char_lib = {
    row["char_name"]: CharEntry(row["base_health"], row["start_relic_name"])
    for row in cursor.fetchall()
}
