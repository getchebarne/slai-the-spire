import importlib
import sqlite3
from dataclasses import dataclass

from src.game.constants import DB_PATH
from src.game.logic.card.base import BaseCardLogic


@dataclass
class CardEntry:
    card_desc: str
    card_cost: int  # TODO: change name to card_base_cost
    card_type: str
    card_rarity: str
    card_logic: BaseCardLogic


# Connect to the SQLite database
connection = sqlite3.connect(DB_PATH)
connection.row_factory = sqlite3.Row

# Create a cursor object
cursor = connection.cursor()

# Execute database query
cursor.execute("""SELECT * FROM CardLib""")

# Initialize card library. The card library is implemented as a dictionary mapping card_name to a
# an instance of CardEntry
card_lib: dict[str, CardEntry] = {}

# Iterate over the rows of the query result
for row in cursor.fetchall():
    card_name = row["card_name"]

    # Get the card's logic
    logic_module = importlib.import_module(f"src.game.logic.card.{card_name.lower()}")
    card_logic = getattr(logic_module, f"{card_name}Logic")()

    # Create a CardEntry instance and add it to the card library
    card_lib[card_name] = CardEntry(
        card_desc=row["card_desc"],
        card_cost=row["card_cost"],
        card_type=row["card_type"],
        card_rarity=row["card_rarity"],
        card_logic=card_logic,
    )
