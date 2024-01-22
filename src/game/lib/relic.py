import sqlite3
from typing import Any, Dict

from game.constants import DB_PATH


# Connect to the SQLite database
connection = sqlite3.connect(DB_PATH)
connection.row_factory = sqlite3.Row

# Create a cursor object
cursor = connection.cursor()

# Execute database query
cursor.execute(
    """
    SELECT
        *
    FROM
        Relic
        LEFT JOIN RelicBattleEndEffects USING (relic_name)
        LEFT JOIN RelicBattleStartEffects USING (relic_name)
    """
)
# Initialize card library. The card library is implemented as a dictionary mapping card_name to a
# dictionary containing card information
card_lib: Dict[str, Dict[str, Any]] = {}

# Fetch all rows. TODO: wrap in function
rows = cursor.fetchall()
