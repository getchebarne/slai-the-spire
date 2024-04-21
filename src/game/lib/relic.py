import sqlite3
from dataclasses import dataclass

from src.game.constants import DB_PATH


@dataclass
class RelicEntry:
    relic_desc: str
    relic_rarity: str


# Connect to the SQLite database
connection = sqlite3.connect(DB_PATH)
connection.row_factory = sqlite3.Row

# Create a cursor object
cursor = connection.cursor()

# Execute database query
cursor.execute("SELECT * FROM RelicLib")

# Initialize relic library. The relic library is implemented as a dictionary mapping relic_name to
# an instance of RelicEntry
card_lib = {
    row["relic_name"]: RelicEntry(row["relic_desc"], row["relic_rarity"])
    for row in cursor.fetchall()
}
