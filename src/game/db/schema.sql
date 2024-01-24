-- CardType
CREATE TABLE
    CardType (card_type TEXT PRIMARY KEY NOT NULL);

-- CardRarity
CREATE TABLE
    CardRarity (card_rarity TEXT PRIMARY KEY NOT NULL);

-- RelicRarity
CREATE TABLE
    RelicRarity (relic_rarity TEXT PRIMARY KEY NOT NULL);

-- CardLib
CREATE TABLE
    CardLib (
        card_name TEXT PRIMARY KEY NOT NULL,
        card_desc TEXT NOT NULL,
        card_cost INTEGER,
        card_type TEXT NOT NULL,
        card_rarity TEXT NOT NULL,
        FOREIGN KEY (card_type) REFERENCES CardType (card_type),
        FOREIGN KEY (card_rarity) REFERENCES CardRarity (card_rarity)
    );

-- RelicLib TODO: add colour
CREATE TABLE
    RelicLib (
        relic_name TEXT PRIMARY KEY NOT NULL,
        relic_desc TEXT NOT NULL,
        relic_rarity TEXT NOT NULL,
        FOREIGN KEY (relic_rarity) REFERENCES RelicRarity (relic_rarity)
    );

-- Modifier
CREATE TABLE
    Modifier (
        modifier_name TEXT PRIMARY KEY NOT NULL,
        modifier_desc TEXT NOT NULL,
        stacks_duration BOOLEAN NOT NULL,
        stacks_counter BOOLEAN NOT NULL
    );

-- MonsterLib
CREATE TABLE
    MonsterLib (monster_name TEXT PRIMARY KEY NOT NULL, base_health INT NOT NULL);

-- MonsterMoves
CREATE TABLE
    MonsterMoves (
        monster_name TEXT NOT NULL,
        move_name TEXT NOT NULL,
        PRIMARY KEY (monster_name, move_name),
        FOREIGN KEY (monster_name) REFERENCES MonsterLib (monster_name)
    );

-- CharacterLib
CREATE TABLE
    CharacterLib (
        char_name TEXT PRIMARY KEY NOT NULL,
        base_health INT NOT NULL,
        start_relic_name TEXT NOT NULL,
        FOREIGN KEY (start_relic_name) REFERENCES RelicLib (relic_name)
    );

-- StarterDeck
CREATE TABLE
    StarterDeck (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        char_name TEXT NOT NULL,
        card_name TEXT NOT NULL,
        FOREIGN KEY (char_name) REFERENCES CharacterLib (char_name),
        FOREIGN KEY (card_name) REFERENCES CardLib (card_name)
    );