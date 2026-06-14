import warnings
from typing import NamedTuple

import numpy as np
import torch
from slai import Card
from slai import Effect
from slai import GameState
from slai import CardColor
from slai import CardCostKind
from slai import CardKind
from slai import CardName
from slai import CardRarity
from slai import ModifierKind
from slai import Screen
from slai import members

from src.rl.encoding.effect import ENCODING_DIM_EFFECTS
from src.rl.encoding.effect import encode_effects_into
from src.rl.index import KIND_TOKENS
from src.rl.index import LOCAL_SLICE
from src.rl.index import NUM_KIND_TOKENS
from src.rl.index import TOKEN_SIZE
from src.rl.index import Token
from src.rl.index import TokenContext
from src.rl.index import TokenKind
from src.rl.index import token_entities

# Enum maps
_MAP_CARD_NAME = {card_name: i for i, card_name in enumerate(members(CardName))}
_MAP_CARD_KIND = {card_kind: i for i, card_kind in enumerate(members(CardKind))}
_MAP_CARD_COLOR = {card_color: i for i, card_color in enumerate(members(CardColor))}
_MAP_CARD_RARITY = {card_rarity: i for i, card_rarity in enumerate(members(CardRarity))}
_MAP_CARD_COST_KIND = {
    card_cost_kind: i
    for i, card_cost_kind in enumerate(m for m in dir(CardCostKind) if not m.startswith("_"))
}

# Cost scalar
_COST_MIN = 0
_COST_MAX = 5
_COST_VECTOR_DIM = _COST_MAX - _COST_MIN + 1

# Encoding dimension: identity+energy features (cached) + a per-step combat suffix
# (modifier-adjusted block/damage + two decision bits) that depends on live combat
# state, so it can't be identity-cached.
ENCODING_DIM_CARD_CACHED = (
    len(_MAP_CARD_NAME)  # Name OHE
    + len(_MAP_CARD_KIND)  # Kind OHE
    + len(_MAP_CARD_COLOR)  # Color OHE
    + len(_MAP_CARD_RARITY)  # Rarity OHE
    + len(_MAP_CARD_COST_KIND)  # Cost kind OHE
    + _COST_VECTOR_DIM  # Cost vector
    + ENCODING_DIM_EFFECTS  # Per-EffectKind feature blocks (effect.py)
    + 1  # Cost scalar
    + 1  # Upgraded
    + 1  # Exhaust
    + 1  # Innate
    + 1  # Ethereal
    + 1  # Retain
    + 1  # Requires target
    + 1  # Playable
    + 1  # Signed energy delta (energy_current - cost)
    + 1  # Cost base scalar
    + 1  # Cost zero-once (free-to-play-once)
    + 1  # Affordable (cost <= energy_current)
)
ENCODING_DIM_CARD_COMBAT = 4  # adj block, adj damage, covers-incoming, can-lethal
ENCODING_DIM_CARD = ENCODING_DIM_CARD_CACHED + ENCODING_DIM_CARD_COMBAT

# Combat-suffix normalization caps (Act-1 ranges)
_ADJ_BLOCK_CAP = 40
_ADJ_DMG_CAP = 40

# Encoding cache: the identity+energy PREFIX depends only on (identity_hash, energy)
_CARD_ROW_CACHE: dict[tuple[int, int], np.ndarray] = {}
_CARD_ROW_CACHE_MAX = 100_000
# Base (pre-modifier) damage/hits/block per identity, for the combat suffix
_CARD_COMBAT_BASE_CACHE: dict[int, tuple[int, int, int]] = {}

# Set tracking card tokens that have triggered truncation warnings
_WARNED_TRUNCATED: set[Token] = set()

# The full owned deck is a valid token only in non-combat screens (deck-edit + shop/
# reward synergy + map planning). In combat it's hidden — the same cards are visible
# via the draw/hand/discard/exhaust piles (their union is the deck) under the shared
# card projection. Skipping its encode here both saves the work AND masks it: the
# skipped slots keep x_pad=False, which is the deck's visibility mask downstream.
_DECK_SCREENS = frozenset(
    {Screen.Map, Screen.Chest, Screen.RestSite, Screen.Shop, Screen.Reward, Screen.Event}
)


def _encode_card_into(card: Card, energy_current: int, pos: int, out: np.ndarray) -> int:
    # Name OHE
    out[pos + _MAP_CARD_NAME[card.name]] = 1.0
    pos += len(_MAP_CARD_NAME)

    # Kind OHE
    out[pos + _MAP_CARD_KIND[card.kind]] = 1.0
    pos += len(_MAP_CARD_KIND)

    # Color OHE
    out[pos + _MAP_CARD_COLOR[card.color]] = 1.0
    pos += len(_MAP_CARD_COLOR)

    # Rarity OHE
    out[pos + _MAP_CARD_RARITY[card.rarity]] = 1.0
    pos += len(_MAP_CARD_RARITY)

    # Cost kind OHE
    out[pos + _MAP_CARD_COST_KIND[type(card.cost_kind).__name__]] = 1.0
    pos += len(_MAP_CARD_COST_KIND)

    # Cost vector
    cost = max(_COST_MIN, min(card.cost, _COST_MAX))
    out[pos + cost] = 1.0
    pos += _COST_VECTOR_DIM

    # Per-EffectKind effect blocks
    pos = encode_effects_into(card.effects, pos, out)

    # Scalars
    out[pos] = cost / _COST_MAX
    out[pos + 1] = float(card.upgraded)
    out[pos + 2] = float(card.exhaust)
    out[pos + 3] = float(card.innate)
    out[pos + 4] = float(card.ethereal)
    out[pos + 5] = float(card.retain)
    out[pos + 6] = float(card.requires_target)
    out[pos + 7] = float(card.playable)
    # Signed energy headroom: >0 surplus, <0 shortfall (abs() destroyed direction)
    out[pos + 8] = max(-_COST_MAX, min(energy_current - card.cost, _COST_MAX)) / _COST_MAX
    out[pos + 9] = max(_COST_MIN, min(card.cost_base, _COST_MAX)) / _COST_MAX
    out[pos + 10] = float(card.cost_zero_once)
    # Affordable bit (engine `playable` excludes energy; mirrors shop.py's flag)
    out[pos + 11] = float(card.cost <= energy_current)
    pos += 12

    return pos


# ---- Combat suffix (post-cache; depends on live monster + player-modifier state) ----


class _CombatCtx(NamedTuple):
    strength: int
    dexterity: int
    vigor: int
    weak: bool
    frail: bool
    double_damage: bool
    char_block: int
    total_incoming: int
    min_eff_hp: int  # min over monsters of (health + block); huge if none


def _combat_context(game_state: GameState) -> _CombatCtx:
    """Player attack/block modifiers + the turn's total incoming damage and weakest
    monster, read once per state for the hand's combat suffix."""
    mods = {m.kind: m.stacks for m in game_state.character.modifiers}
    total_incoming = 0
    min_eff_hp = 1_000_000
    for monster in game_state.monsters:
        intent = monster.intent
        total_incoming += (intent.damage or 0) * (intent.instances or 1)
        eff_hp = monster.health + monster.block
        if eff_hp < min_eff_hp:
            min_eff_hp = eff_hp
    return _CombatCtx(
        strength=mods.get(ModifierKind.Strength, 0),
        dexterity=mods.get(ModifierKind.Dexterity, 0),
        vigor=mods.get(ModifierKind.Vigor, 0),
        weak=ModifierKind.Weak in mods,
        frail=ModifierKind.Frail in mods,
        double_damage=ModifierKind.DoubleDamage in mods,
        char_block=game_state.character.block,
        total_incoming=total_incoming,
        min_eff_hp=min_eff_hp,
    )


def _card_combat_base(card: Card) -> tuple[int, int, int]:
    """(base flat damage total, hit count, base block) from a card's effects — the
    pre-modifier values. Identity-derived (effects are part of identity), so cached."""
    base = _CARD_COMBAT_BASE_CACHE.get(card.identity_hash)
    if base is None:
        dmg = hits = block = 0
        for effect in card.effects:
            if isinstance(effect, Effect.DamagePhysical):
                dmg += effect.amount
                hits += 1
            elif isinstance(effect, Effect.BlockGain):
                block += effect.amount
        base = (dmg, hits, block)
        if len(_CARD_COMBAT_BASE_CACHE) >= _CARD_ROW_CACHE_MAX:
            _CARD_COMBAT_BASE_CACHE.clear()
        _CARD_COMBAT_BASE_CACHE[card.identity_hash] = base
    return base


def _encode_card_combat_into(card: Card, ctx: _CombatCtx, out: np.ndarray) -> None:
    """Write the combat suffix into `out` (length ENCODING_DIM_CARD_COMBAT): the
    modifier-adjusted block/damage this card would produce now plus two P1/P5 decision
    bits. Replicates the engine scaling (utils.rs scale_attack_damage,
    process_effect_block_gain.rs); target-agnostic (no Vulnerable/Intangible, set at L3)."""
    base_dmg, hits, base_block = _card_combat_base(card)

    # Block: (base + Dexterity) * 0.75^Frail, floored at 0.
    adj_block = base_block + ctx.dexterity if base_block else 0
    if ctx.frail:
        adj_block = adj_block * 0.75
    adj_block = max(0, int(adj_block))

    # Damage: Strength + Vigor added per hit, * 0.75^Weak, * 2^DoubleDamage, floored.
    if hits:
        adj_dmg = base_dmg + (ctx.strength + max(ctx.vigor, 0)) * hits
        if ctx.weak:
            adj_dmg = adj_dmg * 0.75
        if ctx.double_damage:
            adj_dmg = adj_dmg * 2.0
        adj_dmg = max(0, int(adj_dmg))
    else:
        adj_dmg = 0

    out[0] = min(adj_block / _ADJ_BLOCK_CAP, 1.0)
    out[1] = min(adj_dmg / _ADJ_DMG_CAP, 1.0)
    # Covers remaining incoming: my block + this card's block >= the turn's total damage.
    out[2] = float(ctx.char_block + adj_block >= ctx.total_incoming)
    # Can lethal: this card's damage >= the weakest monster's effective HP.
    out[3] = float(adj_dmg >= ctx.min_eff_hp)


def encode_card_into_w_cache(card: Card, energy_current: int, out: np.ndarray) -> None:
    """Write the cached identity+energy PREFIX into out[:ENCODING_DIM_CARD_CACHED]. The
    combat suffix (out[ENCODING_DIM_CARD_CACHED:]) is written separately, post-cache."""
    cache_key = (card.identity_hash, energy_current)
    prefix = _CARD_ROW_CACHE.get(cache_key)
    if prefix is None:
        prefix = np.zeros(ENCODING_DIM_CARD_CACHED, dtype=np.float32)
        _encode_card_into(card, energy_current, 0, prefix)
        prefix.flags.writeable = False  # guard the cached master copy
        if len(_CARD_ROW_CACHE) >= _CARD_ROW_CACHE_MAX:
            _CARD_ROW_CACHE.clear()
        _CARD_ROW_CACHE[cache_key] = prefix
    out[:ENCODING_DIM_CARD_CACHED] = prefix


def encode_batch_cards(
    batch_game_state: list[GameState],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every card token (registry CARD kind) into one concatenated
    (B, N_CARDS, ENCODING_DIM_CARD) tensor + mask; tokens live at their
    index.LOCAL_SLICE positions."""
    batch_size = len(batch_game_state)
    num_tokens = NUM_KIND_TOKENS[TokenKind.CARD]

    # Allocate arrays
    x_out = np.zeros((batch_size, num_tokens, ENCODING_DIM_CARD), dtype=np.float32)
    x_pad = np.zeros((batch_size, num_tokens), dtype=bool)

    for b, game_state in enumerate(batch_game_state):
        for token in KIND_TOKENS[TokenKind.CARD]:
            # Deck is hidden in combat (cards visible via the piles); skipping the
            # encode leaves x_pad=False, which is its visibility mask.
            if (
                token == Token(TokenKind.CARD, TokenContext.OWNED)
                and game_state.screen not in _DECK_SCREENS
            ):
                continue

            cards = token_entities(token, game_state)
            energy_current = game_state.energy.energy_current
            token_size = TOKEN_SIZE[token]
            if len(cards) > token_size:
                # Truncate
                if token not in _WARNED_TRUNCATED:
                    _WARNED_TRUNCATED.add(token)
                    warnings.warn(
                        f"{token.context.name} pile of {len(cards)} cards truncated to"
                        f" encoder cap ({token_size})"
                    )

                cards = cards[:token_size]

            offset = LOCAL_SLICE[token].start
            # Combat suffix only on the hand (the playable cards): modifier-adjusted
            # block/damage + decision bits for THIS turn. ctx is read once per state.
            ctx = (
                _combat_context(game_state)
                if token.context == TokenContext.HAND and cards
                else None
            )
            for i, card in enumerate(cards):
                encode_card_into_w_cache(card, energy_current, x_out[b, offset + i])
                if ctx is not None:
                    _encode_card_combat_into(
                        card, ctx, x_out[b, offset + i, ENCODING_DIM_CARD_CACHED:]
                    )

                # Tag mask
                x_pad[b, offset + i] = True

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_pad).to(device),
    )
