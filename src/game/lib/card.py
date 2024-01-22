import sqlite3
from dataclasses import dataclass
from typing import Dict

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
# dictionary containing card information
card_lib: Dict[str, CardEntry] = {}

# Fetch all rows. TODO: wrap in function
rows = cursor.fetchall()
for row in rows:
    card_name = row["card_name"]

    # If card_name is not in card_lib, create a new card instance
    if card_name not in card_lib:
        card_lib[card_name] = CardEntry(
            card_desc=row["card_desc"],
            card_cost=row["card_cost"],
            card_type=row["card_type"],
            card_rarity=row["card_rarity"],
        )
