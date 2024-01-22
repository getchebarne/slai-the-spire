-- EffectType
INSERT INTO
    EffectType (effect_type)
VALUES
    ("DAMAGE"),
    ("BLOCK"),
    ("WEAK"),
    ("PLUS_STR"),
    ("HEAL"),
    ("DRAW");

-- EffectTargetType
INSERT INTO
    EffectTargetType (effect_target_type)
VALUES
    ("CHAR"),
    ("SINGLE_MONSTER"),
    ("ALL_MONSTERS"),
    ("RANDOM_MONSTER");

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

-- CardEffects
INSERT INTO
    CardEffects (card_name, effect_type, effect_target_type, effect_value)
VALUES
    ("Strike", "DAMAGE", "SINGLE_MONSTER", 6),
    ("Defend", "BLOCK", "CHAR", 5),
    ("Neutralize", "DAMAGE", "SINGLE_MONSTER", 3),
    ("Neutralize", "WEAK", "SINGLE_MONSTER", 1);

-- Relic
INSERT INTO
    Relic (relic_name, relic_desc, relic_rarity)
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

-- RelicBattleEndEffects
INSERT INTO
    RelicBattleEndEffects (relic_name, effect_type, effect_value)
VALUES
    ("Burning Blood", "HEAL", 6);

-- RelicBattleStartEffects
INSERT INTO
    RelicBattleStartEffects (relic_name, effect_type, effect_value)
VALUES
    ("Ring of the Snake", "DRAW", 2),
    ("Vajra", "PLUS_STR", 1);

-- RelicTurnEndEffects
INSERT INTO
    RelicTurnEndEffects (relic_name, effect_type, effect_value)
VALUES
    ("Orichalcum", "BLOCK", 6);

-- RelicTurnStartEffects
INSERT INTO
    RelicTurnStartEffects (relic_name, effect_type, effect_value)
VALUES
    ("Mercury Hourglass", "DAMAGE", 3);

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

-- ModifierTurnEndEffects
INSERT INTO
    ModifierTurnEndEffects (modifier_name, effect_type)
VALUES
    ("Poison", "DAMAGE");

-- ModifierTurnStartEffects
-- ModifierBattleEndEffects
-- MonsterLib
INSERT INTO
    MonsterLib (monster_name, base_health)
VALUES
    ("Dummy", 60);

-- MonsterMoves
INSERT INTO
    MonsterMoves (monster_name, move_name, effect_type, effect_target_type, effect_value)
VALUES
    ("Dummy", "Attack", "DAMAGE", "CHAR", 6),
    ("Dummy", "Defend", "BLOCK", "SINGLE_MONSTER", 5);

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

---------------------
-- Initialize game --
---------------------
-- CurrentEntity
INSERT INTO
    CurrentEntity (entity_name, current_health, max_health, current_block)
VALUES
    ("Silent", 60, 60, 0),
    ("Dummy", 60, 60, 0);

-- Energy
INSERT INTO
    Energy (current_energy, max_energy)
VALUES
    (3, 3);

-- Deck
INSERT INTO
    Deck (card_name)
VALUES
    ("Strike"),
    ("Strike"),
    ("Strike"),
    ("Strike"),
    ("Strike"),
    ("Defend"),
    ("Defend"),
    ("Defend"),
    ("Defend"),
    ("Defend"),
    ("Neutralize");