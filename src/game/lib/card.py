import importlib
import sqlite3
from dataclasses import dataclass

from src.game.constants import DB_PATH
from src.game.logic.card.base import BaseCardLogic


@dataclass(frozen=True)
class CardEntry:
    description: str
    cost: int  # TODO: change name to card_base_cost
    type: str
    rarity: str
    logic: BaseCardLogic


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
    logic_instance = getattr(logic_module, f"{card_name}Logic")()

    # Create a CardEntry instance and add it to the card library
    card_lib[card_name] = CardEntry(
        description=row["card_desc"],
        cost=row["card_cost"],
        type=row["card_type"],
        rarity=row["card_rarity"],
        logic=logic_instance,
    )
