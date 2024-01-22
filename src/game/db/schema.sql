-- EffectType
CREATE TABLE
    EffectType (effect_type TEXT PRIMARY KEY NOT NULL);

-- EffectTargetType
CREATE TABLE
    EffectTargetType (effect_target_type TEXT PRIMARY KEY NOT NULL);

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

-- CardEffects
CREATE TABLE
    CardEffects (
        card_name TEXT NOT NULL,
        effect_type TEXT NOT NULL,
        effect_target_type TEXT NOT NULL,
        effect_value INTEGER NOT NULL,
        PRIMARY KEY (card_name, effect_type),
        FOREIGN KEY (card_name) REFERENCES CardLib (card_name),
        FOREIGN KEY (effect_type) REFERENCES EffectType (effect_type),
        FOREIGN KEY (effect_target_type) REFERENCES EffectTargetType (effect_target_type)
    );

-- Relic. TODO: add class
CREATE TABLE
    Relic (
        relic_name TEXT PRIMARY KEY NOT NULL,
        relic_desc TEXT NOT NULL,
        relic_rarity TEXT NOT NULL,
        FOREIGN KEY (relic_rarity) REFERENCES RelicRarity (relic_rarity)
    );

-- RelicBattleEndEffects
CREATE TABLE
    RelicBattleEndEffects (
        relic_name TEXT NOT NULL,
        effect_type TEXT NOT NULL,
        effect_value INTEGER NOT NULL,
        PRIMARY KEY (relic_name, effect_type),
        FOREIGN KEY (relic_name) REFERENCES Relic (relic_name),
        FOREIGN KEY (effect_type) REFERENCES EffectType (effect_type)
    );

-- RelicBattleStartEffects
CREATE TABLE
    RelicBattleStartEffects (
        relic_name TEXT NOT NULL,
        effect_type TEXT NOT NULL,
        effect_value INTEGER NOT NULL,
        PRIMARY KEY (relic_name, effect_type),
        FOREIGN KEY (relic_name) REFERENCES Relic (relic_name),
        FOREIGN KEY (effect_type) REFERENCES EffectType (effect_type)
    );

-- RelicTurnEndEffects
CREATE TABLE
    RelicTurnEndEffects (
        relic_name TEXT NOT NULL,
        effect_type TEXT NOT NULL,
        effect_value INTEGER NOT NULL,
        PRIMARY KEY (relic_name, effect_type),
        FOREIGN KEY (relic_name) REFERENCES Relic (relic_name),
        FOREIGN KEY (effect_type) REFERENCES EffectType (effect_type)
    );

-- RelicTurnStartEffects
CREATE TABLE
    RelicTurnStartEffects (
        relic_name TEXT NOT NULL,
        effect_type TEXT NOT NULL,
        effect_value INTEGER NOT NULL,
        PRIMARY KEY (relic_name, effect_type),
        FOREIGN KEY (relic_name) REFERENCES Relic (relic_name),
        FOREIGN KEY (effect_type) REFERENCES EffectType (effect_type)
    );

-- Modifier
CREATE TABLE
    Modifier (
        modifier_name TEXT PRIMARY KEY NOT NULL,
        modifier_desc TEXT NOT NULL,
        stacks_duration BOOLEAN NOT NULL,
        stacks_counter BOOLEAN NOT NULL
    );

-- ModifierTurnEndEffects
CREATE TABLE
    ModifierTurnEndEffects (
        modifier_name TEXT NOT NULL,
        effect_type TEXT NOT NULL,
        PRIMARY KEY (modifier_name, effect_type),
        FOREIGN KEY (modifier_name) REFERENCES Modifier (modifier_name),
        FOREIGN KEY (effect_type) REFERENCES EffectType (effect_type)
    );

-- ModifierTurnStartEffects
CREATE TABLE
    ModifierTurnStartEffects (
        modifier_name TEXT NOT NULL,
        effect_type TEXT NOT NULL,
        PRIMARY KEY (modifier_name, effect_type),
        FOREIGN KEY (modifier_name) REFERENCES Modifier (modifier_name),
        FOREIGN KEY (effect_type) REFERENCES EffectType (effect_type)
    );

-- ModifierBattleEndEffects
CREATE TABLE
    ModifierBattleEndEffects (
        modifier_name TEXT NOT NULL,
        effect_type TEXT NOT NULL,
        PRIMARY KEY (modifier_name, effect_type),
        FOREIGN KEY (modifier_name) REFERENCES Modifier (modifier_name),
        FOREIGN KEY (effect_type) REFERENCES EffectType (effect_type)
    );

-- MonsterLib
CREATE TABLE
    MonsterLib (monster_name TEXT PRIMARY KEY NOT NULL, base_health INT NOT NULL);

-- MonsterMoves. TODO: normalize
CREATE TABLE
    MonsterMoves (
        monster_name TEXT NOT NULL,
        move_name TEXT NOT NULL,
        effect_type TEXT NOT NULL,
        effect_target_type TEXT NOT NULL,
        effect_value INT NOT NULL,
        PRIMARY KEY (monster_name, move_name, effect_type),
        FOREIGN KEY (monster_name) REFERENCES MonsterLib (monster_name),
        FOREIGN KEY (effect_type) REFERENCES EffectType (effect_type),
        FOREIGN KEY (effect_target_type) REFERENCES EffectTargetType (effect_target_type)
    );

-- CharacterLib
CREATE TABLE
    CharacterLib (
        char_name TEXT PRIMARY KEY NOT NULL,
        base_health INT NOT NULL,
        start_relic_name TEXT NOT NULL,
        FOREIGN KEY (start_relic_name) REFERENCES Relic (relic_name)
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

-- CurrentEntity
CREATE TABLE
    CurrentEntity (
        entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_name TEXT NOT NULL,
        current_health INT NOT NULL,
        max_health INT NOT NULL,
        current_block INT NOT NULL,
        FOREIGN KEY (entity_name) REFERENCES CharacterLib (char_name),
        FOREIGN KEY (entity_name) REFERENCES MonsterLib (monster_name)
    );

-- CurrentEntityModifiers
CREATE TABLE
    CurrentEntityModifiers (
        entity_id INTEGER NOT NULL,
        modifier_name TEXT NOT NULL,
        modifier_stacks INTEGER NOT NULL,
        PRIMARY KEY (entity_id, modifier_name),
        FOREIGN KEY (entity_id) REFERENCES CurrentEntity (entity_id),
        FOREIGN KEY (modifier_name) REFERENCES Modifier (modifier_name)
    );

-- Energy
CREATE TABLE
    Energy (current_energy INT NOT NULL, max_energy INT NOT NULL);

-- Deck
CREATE TABLE
    Deck (
        card_id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_name TEXT NOT NULL,
        FOREIGN KEY (card_name) REFERENCES CardLib (card_name)
    );

-- Hand
CREATE TABLE
    Hand (
        idx INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        card_name TEXT NOT NULL,
        FOREIGN KEY (card_id) REFERENCES Deck (card_id),
        FOREIGN KEY (card_name) REFERENCES CardLib (card_name)
    );

-- DiscardPile
CREATE TABLE
    DiscardPile (
        idx INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        card_name TEXT NOT NULL,
        FOREIGN KEY (card_id) REFERENCES Deck (card_id),
        FOREIGN KEY (card_name) REFERENCES CardLib (card_name)
    );

-- DrawPile
CREATE TABLE
    DrawPile (
        idx INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        card_name TEXT NOT NULL,
        FOREIGN KEY (card_id) REFERENCES Deck (card_id),
        FOREIGN KEY (card_name) REFERENCES CardLib (card_name)
    );