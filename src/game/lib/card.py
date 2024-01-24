import sqlite3
from dataclasses import dataclass

from game.constants import DB_PATH


@dataclass
class CardEntry:
    card_desc: str
    card_cost: int
    card_type: str
    card_rarity: str


# Connect to the SQLite database
connection = sqlite3.connect(DB_PATH)
connection.row_factory = sqlite3.Row

# Create a cursor object
cursor = connection.cursor()

# Execute database query
cursor.execute("""SELECT * FROM CardLib""")

# Initialize card library. The card library is implemented as a dictionary mapping card_name to a
# an instance of CardEntry
card_lib = {
    row["card_name"]: CardEntry(
        row["card_desc"], row["card_cost"], row["card_type"], row["card_rarity"]
    )
    for row in cursor.fetchall()
}
