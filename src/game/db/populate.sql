-- TODO: change population order
-- CardType
INSERT INTO
    CardType (card_type)
VALUES
    ("ATTACK"),
    ("SKILL"),
    ("POWER"),
    ("CURSE"),
    ("STATUS");

-- CardRarity
INSERT INTO
    CardRarity (card_rarity)
VALUES
    ("BASIC"),
    ("COMMON"),
    ("UNCOMMON"),
    ("RARE"),
    ("SPECIAL");

-- RelicRarity
INSERT INTO
    RelicRarity (relic_rarity)
VALUES
    ("STARTER"),
    ("COMMON"),
    ("UNCOMMON"),
    ("RARE"),
    ("BOSS"),
    ("SHOP"),
    ("EVENT");

-- CardLib
INSERT INTO
    CardLib (card_name, card_desc, card_cost, card_type, card_rarity)
VALUES
    ("Strike", "Deal 6 damage.", 1, "ATTACK", "BASIC"),
    ("Defend", "Gain 5 block.", 1, "SKILL", "BASIC"),
    ("Neutralize", "Deal 3 damage. Apply 1 Weak.", 0, "ATTACK", "BASIC");

-- RelicLib
INSERT INTO
    RelicLib (relic_name, relic_desc, relic_rarity)
VALUES
    (
        "Ring of the Snake",
        "At the start of each combat, draw 2 additional cards.",
        "STARTER"
    ),
    ("Burning Blood", "At the end of combat, heal 6 HP.", "STARTER"),
    ("Vajra", "At the start of each combat, gain 1 Strength.", "COMMON"),
    (
        "Orichalcum",
        "If you end your turn without Block, gain 6 Block.",
        "COMMON"
    ),
    (
        "Mercury Hourglass",
        "At the start of your turn, deal 3 damage to ALL enemies.",
        "UNCOMMON"
    );

-- Modifier
INSERT INTO
    Modifier (modifier_name, modifier_desc, stacks_duration, stacks_counter)
VALUES
    ("Strength", "Increases attack damage by X (per hit).", 0, 0),
    ("Weak", "Target deals 25% less attack damage.", 1, 0),
    (
        "Poison",
        "At the beginning of its turn, the target loses X HP and 1 stack of poison.",
        1,
        0
    );

-- MonsterLib
INSERT INTO
    MonsterLib (monster_name, base_health)
VALUES
    ("Dummy", 60);

-- MonsterMoves
INSERT INTO
    MonsterMoves (monster_name, move_name)
VALUES
    ("Dummy", "Attack"),
    ("Dummy", "Defend");

-- CharacterLib
INSERT INTO
    CharacterLib (char_name, base_health, start_relic_name)
VALUES
    ("Silent", 60, "Ring of the Snake"),
    ("Ironclad", 60, "Burning Blood");

-- StarterDeck
INSERT INTO
    StarterDeck (char_name, card_name)
VALUES
    ("Silent", "Strike"),
    ("Silent", "Strike"),
    ("Silent", "Strike"),
    ("Silent", "Strike"),
    ("Silent", "Strike"),
    ("Silent", "Defend"),
    ("Silent", "Defend"),
    ("Silent", "Defend"),
    ("Silent", "Defend"),
    ("Silent", "Defend"),
    ("Silent", "Neutralize");