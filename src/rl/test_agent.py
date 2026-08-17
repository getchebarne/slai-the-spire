"""
Test a trained agent by rendering a single game.

Usage:
    poetry run python -m src.rl.test_agent --exp-path experiments/ppo_hierarchical_v1
"""

import os
import random
import sys
import time

import click
import slai
import torch
from slai import IntentKind


_INTENT_BLOCK_KINDS = frozenset({IntentKind.Block, IntentKind.AttackBlock, IntentKind.BlockBuff})
_INTENT_BUFF_KINDS = frozenset({IntentKind.Buff, IntentKind.AttackBuff, IntentKind.BlockBuff})
_INTENT_DEBUFF_KINDS = frozenset(
    {IntentKind.Debuff, IntentKind.AttackDebuff, IntentKind.DebuffPowerful}
)

from src.rl.constants import ASCENSION_LEVEL
from src.rl.constants import FAST_MODE
from src.rl.encoding.screen import Focus
from src.rl.encoding.screen import derive_focus
from src.rl.encoding.state import encode_batch_game_state
from src.rl.masks import build_masks
from src.rl.models import ActorCritic
from src.rl.reward import cumulative_floor
from src.rl.types import SliceKind
from src.rl.types import TGameState
from src.rl.types import TMask
from src.rl.utils import action_from_actiontype
from src.rl.utils import load_config


try:
    N_COL, _ = os.get_terminal_size()
except OSError:
    N_COL = 120


# ANSI styling — mirrors play/__main__.py's curses color pairs.
_USE_COLOR = sys.stdout.isatty()
_RED = "\033[31m" if _USE_COLOR else ""
_CYAN = "\033[36m" if _USE_COLOR else ""
_RESET = "\033[0m" if _USE_COLOR else ""


def _hp(s: str) -> str:
    return f"{_RED}{s}{_RESET}"


def _block(s: str) -> str:
    return f"{_CYAN}{s}{_RESET}"


def _ansi_len(s: str) -> int:
    """Visible length, stripping ANSI escape sequences."""
    return len(s.replace(_RED, "").replace(_CYAN, "").replace(_RESET, ""))


def _variant_name(value: object) -> str:
    """`ModifierKind.Strength` → `Strength` (slai pyclass enums repr that way)."""
    s = repr(value)
    return s.rsplit(".", 1)[-1] if "." in s else s


def _event_name(event: slai.Event) -> str:
    """`EventKindWorldOfGoop` → `World of Goop`-ish (`WorldOfGoop`)."""
    return type(event.kind).__name__.removeprefix("EventKind")


def _effects_summary(effects) -> str:
    return ", ".join(type(e).__name__.removeprefix("Effect") for e in effects)


def is_card_playable(card, e) -> bool:
    return card.cost <= e and card.playable


# Modifier abbreviations — copied from play/__main__.py::MOD_ABBR.
_MOD_ABBR = {
    slai.ModifierKind.Accuracy: "acc",
    slai.ModifierKind.AfterImage: "aimg",
    slai.ModifierKind.Angry: "ang",
    slai.ModifierKind.Artifact: "art",
    slai.ModifierKind.Asleep: "slp",
    slai.ModifierKind.Blur: "blur",
    slai.ModifierKind.Burst: "brst",
    slai.ModifierKind.Choke: "chk",
    slai.ModifierKind.CorpseExplosion: "cexp",
    slai.ModifierKind.CurlUp: "curl",
    slai.ModifierKind.Dexterity: "dex",
    slai.ModifierKind.DoubleDamage: "ddmg",
    slai.ModifierKind.DrawCardNextTurn: "ndraw",
    slai.ModifierKind.Enrage: "enr",
    slai.ModifierKind.Entangled: "entg",
    slai.ModifierKind.Envenom: "env",
    slai.ModifierKind.Frail: "frail",
    slai.ModifierKind.InfiniteBlades: "iblad",
    slai.ModifierKind.Intangible: "intg",
    slai.ModifierKind.Metallicize: "metl",
    slai.ModifierKind.ModeShift: "mshft",
    slai.ModifierKind.NextTurnBlock: "ntblk",
    slai.ModifierKind.NextTurnEnergy: "nterg",
    slai.ModifierKind.NoDraw: "nodr",
    slai.ModifierKind.NoxiousFumes: "nox",
    slai.ModifierKind.Phantasmal: "phant",
    slai.ModifierKind.Poison: "poi",
    slai.ModifierKind.Retain: "ret",
    slai.ModifierKind.Ritual: "rit",
    slai.ModifierKind.Shackled: "shk",
    slai.ModifierKind.SharpHide: "shrp",
    slai.ModifierKind.Splittable: "splt",
    slai.ModifierKind.SporeCloud: "spore",
    slai.ModifierKind.Strength: "str",
    slai.ModifierKind.Thievery: "thief",
    slai.ModifierKind.Thorns: "thorn",
    slai.ModifierKind.ThousandCuts: "cuts",
    slai.ModifierKind.ToolsOfTheTrade: "tools",
    slai.ModifierKind.Vulnerable: "vuln",
    slai.ModifierKind.Weak: "weak",
    slai.ModifierKind.WraithForm: "wrth",
}


def _fmt_modifiers(mods: list) -> str:
    return "  ".join(
        f"{_MOD_ABBR.get(m.kind, _variant_name(m.kind).lower())} {m.stacks}" for m in mods
    )


def _fmt_action(action: slai.Action) -> str:
    """Format a flat slai.Action for the log (idx meaning per ACTION_SPEC_REGISTRY)."""
    at = action.action_type
    i = action.idxs
    name = _variant_name(at)
    if at == slai.ActionType.CardPlay:
        tgt = f", target={i[1]}" if len(i) > 1 else ""
        return f"CardPlay(hand={i[0]}{tgt})"
    if at == slai.ActionType.PotionUse:
        tgt = f", target={i[1]}" if len(i) > 1 else ""
        return f"PotionUse(slot={i[0]}{tgt})"
    if at in (slai.ActionType.CardDiscard, slai.ActionType.CardRetain):
        return f"{name}(indices={list(i)})"
    if at == slai.ActionType.PotionDiscard:
        return f"PotionDiscard(slot={i[0]})"
    if at == slai.ActionType.RoomSelect:
        return f"RoomSelect(col={i[0]})"
    # All remaining single-idx actions (card-pick, reward/shop selects, event option)
    if i:
        return f"{name}(idx={i[0]})"
    return name  # TurnEnd, Rest, RoomExit, ChestOpen, RewardTakeRelic/Potion/Gold


def _describe_action(view: slai.GameState, action: slai.Action) -> str:
    """Human-readable description of `action`, resolved against the PRE-step view
    (the action's idxs reference entities of the state it was chosen in)."""
    at = action.action_type
    i = action.idxs
    AT = slai.ActionType
    combat = view.combat
    hand = combat.hand if combat is not None else []

    def card(pile, k):
        return _variant_name(pile[k].name) if k < len(pile) else f"card #{k}"

    def potion(pile, k):
        return _variant_name(pile[k].name) if k < len(pile) else f"potion #{k}"

    def target(s: str) -> str:
        monsters = combat.monsters if combat is not None else []
        if len(i) > 1 and i[1] < len(monsters):
            return f"{s} targeting {_variant_name(monsters[i[1]].name)}"
        return s

    if at == AT.CardPlay:
        return target(f"played {card(hand, i[0])}")
    if at == AT.TurnEnd:
        return "ended turn"
    if at == AT.PotionUse:
        return target(f"drank {potion(view.potions, i[0])}")
    if at == AT.PotionDiscard:
        return f"discarded potion {potion(view.potions, i[0])}"
    if at == AT.CardDiscard:
        return f"discarded {card(hand, i[0])}"
    if at == AT.CardRetain:
        return f"retained {card(hand, i[0])}"
    if at == AT.CardExhaust:
        return f"exhausted {card(hand, i[0])}"
    if at == AT.CardSetup:
        return f"set aside {card(hand, i[0])} (Setup)"
    if at == AT.CardNightmare:
        return f"copied {card(hand, i[0])} (Nightmare)"
    if at == AT.CardDiscover:
        return f"discovered {card(combat.discover if combat is not None else [], i[0])}"
    if at == AT.CardMoveToHand:
        return f"moved {card(combat.pile_draw if combat is not None else [], i[0])} to hand"
    if at == AT.CardBottle:
        return f"bottled {card(view.deck, i[0])}"
    if at == AT.RoomSelect:
        y = 0 if view.map.y_current is None else view.map.y_current + 1
        rooms = view.map.rooms
        kind = ""
        if y < len(rooms) and i[0] < len(rooms[y]) and rooms[y][i[0]] is not None:
            kind = f" ({_variant_name(rooms[y][i[0]].room_kind)})"
        return f"moved to column {i[0]}{kind} on floor {y}"
    if at == AT.RoomExit:
        return "left the room"
    if at == AT.ChestOpen:
        return "opened the chest"
    if at == AT.Rest:
        return "rested"
    if at == AT.RestDig:
        return "dug for a relic (Shovel)"
    if at == AT.RestLift:
        return "lifted (Girya)"
    if at == AT.RestToke:
        return "burned a card (Peace Pipe)"
    if at == AT.PickSkip:
        return "skipped the pick"
    if at == AT.CardUpgrade:
        return f"upgraded {card(view.deck, i[0])}"
    if at == AT.CardPurge:
        return f"purged {card(view.deck, i[0])}"
    if at == AT.CardTransform:
        return f"transformed {card(view.deck, i[0])}"
    if at == AT.CardDuplicate:
        return f"duplicated {card(view.deck, i[0])}"
    if at == AT.RewardTakeCard:
        r = view.reward
        if r is not None and i[0] < len(r.cards):
            return f"added {card(r.cards[i[0]], i[1])} to the deck (bundle {i[0]})"
        return f"added reward card {list(i)} to the deck"
    if at == AT.RewardSingingBowl:
        return f"took +2 max HP over bundle {i[0]} (Singing Bowl)"
    if at == AT.RewardTakeGold:
        gold = view.reward.gold if view.reward is not None else None
        return f"took {gold} gold" if gold is not None else "took the gold"
    if at == AT.RewardTakeRelic:
        relics = view.reward.relics if view.reward is not None else []
        if i and i[0] < len(relics):
            return f"took relic {_variant_name(relics[i[0]].name)}"
        return "took the relic"
    if at == AT.RewardTakePotion:
        potions = view.reward.potions if view.reward is not None else []
        if i and i[0] < len(potions):
            return f"took potion {_variant_name(potions[i[0]].name)}"
        return "took the potion"
    if at == AT.ShopBuyCard:
        return f"bought {card(view.shop.cards, i[0])} for {view.shop.card_prices[i[0]]}g"
    if at == AT.ShopBuyRelic:
        name = _variant_name(view.shop.relics[i[0]].name)
        return f"bought relic {name} for {view.shop.relic_prices[i[0]]}g"
    if at == AT.ShopBuyPotion:
        name = _variant_name(view.shop.potions[i[0]].name)
        return f"bought potion {name} for {view.shop.potion_prices[i[0]]}g"
    if at == AT.ShopPurge:
        return f"purged {card(view.deck, i[0])} at the shop for {view.shop.purge_cost}g"
    if at == AT.EventOptionSelect:
        ev = view.event
        if ev is not None and i and i[0] < len(ev.options):
            effects = _effects_summary(ev.options[i[0]])
            return f"chose option {i[0]} [{effects or 'no-op'}] ({_event_name(ev)})"
        return f"chose event option {i[0]}"
    return _fmt_action(action)  # unmapped kinds fall back to the raw form


def _enemy_row(i: int, m: slai.Monster) -> str:
    """Render a monster row WITHOUT padding — pre-pad outside this fn."""
    intent = m.intent
    if intent.damage:
        intent_str = f"ATK {intent.damage}x{intent.instances or 1}"
    elif intent.kind in _INTENT_BLOCK_KINDS:
        intent_str = "BLOCK"
    elif intent.kind in _INTENT_BUFF_KINDS:
        intent_str = "BUFF"
    elif intent.kind in _INTENT_DEBUFF_KINDS:
        intent_str = "DEBUFF"
    else:
        intent_str = ""
    hp = _hp(f"HP {m.health}/{m.health_max}")
    blk = f"  {_block(f'Block {m.block}')}" if m.block > 0 else ""
    return f"[{i}] {_variant_name(m.name)}: {hp}{blk}  → {intent_str}"


def _format_view(view: slai.GameState) -> str:
    """Compact textual rendering of the game state.

    Two-column layout: character/hand on the left, monsters right-padded
    so name/HP/intent columns line up vertically (mirrors `play/__main__.py`).
    """
    combat = view.combat
    header = f"Focus: {derive_focus(view).name}  (act {view.act})"
    if view.pending is not None:
        header += f"  pending={type(view.pending).__name__}"
    if combat is not None and combat.discover:
        header += f"  discover={[_variant_name(c.name) for c in combat.discover]}"
    lines = [header]

    char_line = (
        f"Char: {_hp(f'HP {view.character.health}/{view.character.health_max}')}"
        f"  {_block(f'Block {view.character.block}')}"
    )
    lines.append(char_line)
    if view.character.modifiers:
        lines.append(f"  modifiers: {_fmt_modifiers(view.character.modifiers)}")

    if view.relics:
        relics_str = ", ".join(_variant_name(r.name) for r in view.relics)
        lines.append(f"Relics: {relics_str}")

    if combat is not None and combat.monsters:
        # Right-pad monster rows so they all start at the same column —
        # matches play/__main__.py's `block_x = right_edge - max(...)`.
        rendered = [_enemy_row(i, m) for i, m in enumerate(combat.monsters)]
        max_w = max(_ansi_len(r) for r in rendered)
        left_pad = max(0, N_COL - max_w)
        lines.append("Monsters:")
        for i, (m, row) in enumerate(zip(combat.monsters, rendered)):
            lines.append(" " * left_pad + row)
            if m.modifiers:
                lines.append(" " * left_pad + f"  modifiers: {_fmt_modifiers(m.modifiers)}")

    if combat is not None:
        lines.append(f"Energy: {combat.energy.energy_current}/{combat.energy.energy_max}")
        if combat.pile_stasis:
            lines.append(f"Stasis: {[_variant_name(c.name) for c in combat.pile_stasis]}")
        if combat.bomb_countdown:
            lines.append(f"The Bomb: {combat.bomb_countdown} turns")
        if combat.hand:
            lines.append("Hand:")
            energy_current = combat.energy.energy_current
            for i, c in enumerate(combat.hand):
                tgt = " (target)" if c.requires_target else ""
                playable = "" if is_card_playable(c, energy_current) else " (unplayable)"
                lines.append(f"  [{i}] {_variant_name(c.name)} cost={c.cost}{tgt}{playable}")

    belt = [f"[{s}] {_variant_name(p.name)}" for s, p in enumerate(view.potions) if p is not None]
    if belt:
        lines.append(f"Potions: {'  '.join(belt)}")

    if view.reward is not None:
        r = view.reward
        for bi, bundle in enumerate(r.cards):
            lines.append(f"Reward cards [{bi}]: {[_variant_name(c.name) for c in bundle]}")
        if r.relics:
            lines.append(f"Reward relics: {[_variant_name(x.name) for x in r.relics]}")
        if r.potions:
            lines.append(f"Reward potions: {[_variant_name(p.name) for p in r.potions]}")
        if r.gold is not None:
            lines.append(f"Reward gold: {r.gold}")

    if view.shop is not None:
        s = view.shop
        lines.append(f"Shop gold={view.character.gold} purge={s.purge_cost}")
        lines.append(
            f"  cards: {[(_variant_name(c.name), p) for c, p in zip(s.cards, s.card_prices)]}"
        )
        lines.append(
            f"  relics: {[(_variant_name(r.name), p) for r, p in zip(s.relics, s.relic_prices)]}"
        )
        lines.append(
            f"  potions: {[(_variant_name(p.name), pr) for p, pr in zip(s.potions, s.potion_prices)]}"
        )

    if view.event is not None:
        e = view.event
        lines.append(f"Event: {_event_name(e)}{'  (consumed)' if e.consumed else ''}")
        for i, effects in enumerate(e.options):
            lines.append(f"  [{i}] {_effects_summary(effects) or 'no-op'}")
    return "\n".join(lines)


def get_card_probabilities(
    model: ActorCritic,
    t_game_state: TGameState,
    t_mask_batch: TMask,
) -> torch.Tensor:
    """
    Get grouped probabilities for cards in hand from the card play head.

    Uses grouped softmax: identical cards are deduplicated so each card type
    gets a single probability (not split across duplicates).

    Returns:
        Tensor of per-position play probabilities (MAX_HAND_SIZE,). The mask is
        identity-deduped, so the first copy of each card carries the type's
        probability and later copies show 0.
    """
    t_core_out = model.core(t_game_state)
    t_mask = t_mask_batch.mask_action_idx[
        str(int(slai.ActionType.CardPlay))
    ]  # (1, MAX_SIZE_HAND), deduped
    t_op = model._operation_embedding(
        torch.tensor([int(slai.ActionType.CardPlay)], device=t_mask.device)
    )
    t_keys = model._pointer_key_card(t_core_out.pool[SliceKind.CARD_HAND])
    t_head_out = model._head_level_2(t_keys, t_core_out.global_, t_op, t_mask)
    t_masked = t_head_out.masked_fill(~t_mask, float("-inf"))
    t_probs = torch.softmax(t_masked, dim=-1)
    return t_probs[0]


def format_card_probabilities(
    view: slai.GameState,
    t_probs: torch.Tensor,
) -> str:
    """Format card probabilities grouped by card type."""
    seen: dict[str, dict] = {}
    order: list[str] = []

    hand = view.combat.hand if view.combat is not None else []
    energy_current = view.combat.energy.energy_current if view.combat is not None else 0
    for idx, card in enumerate(hand):
        display = _variant_name(card.name)
        if display not in seen:
            prob = t_probs[idx].item()
            if prob != prob:  # NaN guard
                prob = 0.0
            playable = is_card_playable(card, energy_current)
            seen[display] = {"prob": prob, "count": 1, "playable": playable}
            order.append(display)
        else:
            seen[display]["count"] += 1

    lines = ["Card Play Probabilities:"]
    for name in order:
        info = seen[name]
        count_str = f" x{info['count']}" if info["count"] > 1 else ""
        status = "" if info["playable"] else " (unplayable)"
        bar_len = int(info["prob"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"  {name}{count_str:4} [{bar}] {info['prob']:5.1%}{status}")
    return "\n".join(lines)


def get_action_from_model(
    model: ActorCritic,
    view: slai.GameState,
    legal_actions: list,
    device: torch.device,
    show_card_probs: bool = False,
    greedy: bool = False,
) -> tuple[object, str | None]:
    """Get an action from the model for the given view (masks come from the
    engine's `legal_actions`). Returns (action, card_probs_str)."""
    t_game_state = encode_batch_game_state([view], device)
    t_mask_batch = build_masks([view], [legal_actions], device)

    with torch.no_grad():
        t_action, _ = model.forward(t_game_state, t_mask_batch, greedy=greedy)

        card_probs_str = None
        combat = view.combat
        in_combat = derive_focus(view) is Focus.COMBAT and view.pending is None
        if show_card_probs and in_combat and combat.hand:
            energy_current = combat.energy.energy_current
            if any(is_card_playable(c, energy_current) for c in combat.hand):
                t_probs = get_card_probabilities(model, t_game_state, t_mask_batch)
                card_probs_str = format_card_probabilities(view, t_probs)

    return action_from_actiontype(t_action, 0, view), card_probs_str


def run_game(
    model: ActorCritic,
    device: torch.device,
    delay: float = 0.5,
    verbose: bool = True,
    show_card_probs: bool = True,
    greedy: bool = False,
    check_legal: bool = False,
) -> tuple[int, int]:
    """
    Run a single game with the trained model.

    Returns:
        (final_floor, final_health)
    """
    env = slai.GameEnv(ascension=ASCENSION_LEVEL, fast_mode=FAST_MODE)
    obs = env.reset(seed=random.randint(0, 2**31 - 1))

    step_count = 0
    illegal = 0
    terminated = False
    while not terminated:
        legal = env.get_legal_actions()
        if not legal:
            break
        if verbose:
            print(_format_view(obs))
            print("-" * N_COL)

        action, card_probs_str = get_action_from_model(
            model,
            obs,
            legal,
            device,
            show_card_probs=verbose and show_card_probs,
            greedy=greedy,
        )

        if check_legal:
            legal_set = {(int(a.action_type), tuple(a.idxs)) for a in legal}
            if (int(action.action_type), tuple(action.idxs)) not in legal_set:
                illegal += 1
                print(
                    f"  !! ILLEGAL action {_fmt_action(action)} on focus "
                    f"{derive_focus(obs).name}"
                )

        if verbose:
            if card_probs_str:
                print(card_probs_str)
                print("-" * N_COL)
            print(f"Action: {_describe_action(obs, action)}")
            print("-" * N_COL)
            time.sleep(delay)

        obs, terminated = env.step(action)
        step_count += 1

    if check_legal and illegal:
        print(f"  WARNING: {illegal} illegal actions this game")

    final_floor = cumulative_floor(obs)
    final_health = obs.character.health

    if verbose:
        print(f"\n{'=' * N_COL}")
        print("GAME OVER" + (" — WON" if obs.game_over and final_health > 0 else ""))
        print(f"Final Floor: {final_floor} (act {obs.act})")
        print(f"Final Health: {final_health}")
        print(f"Steps: {step_count}")
        print("=" * N_COL)

    return final_floor, final_health


@click.command()
@click.option(
    "--exp-path",
    default="experiments/ppo_hierarchical_v1",
    help="Path to the experiment directory",
)
@click.option(
    "--delay",
    default=0.3,
    type=float,
    help="Delay between steps (seconds)",
)
@click.option(
    "--device",
    default="cpu",
    type=str,
    help="Device (cpu or cuda)",
)
@click.option(
    "--num-games",
    default=1,
    type=int,
    help="Number of games to run",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Run without rendering (just show final results)",
)
@click.option(
    "--no-probs",
    is_flag=True,
    help="Don't show card probabilities during combat",
)
@click.option(
    "--greedy",
    is_flag=True,
    help="Use greedy (argmax) selection instead of sampling",
)
@click.option(
    "--check-legal",
    is_flag=True,
    help="Assert every model action is in env.get_legal_actions()",
)
@click.option(
    "--random",
    "use_random",
    is_flag=True,
    help="Use a fresh untrained model (no checkpoint needed)",
)
def main(
    exp_path: str,
    delay: float,
    device: str,
    num_games: int,
    quiet: bool,
    no_probs: bool,
    greedy: bool,
    check_legal: bool,
    use_random: bool,
):
    """Test an agent by running games."""
    if use_random:
        model = ActorCritic(
            dim_entity=32,
            dim_global=64,
            transformer_dim_ff=64,
            transformer_num_heads=2,
            transformer_num_blocks=1,
            gnn_num_layers=15,
            map_encoder_dim=16,
            dim_ff_primary=32,
            dim_ff_value=32,
            dim_op=32,
            dim_key=16,
        )
    else:
        config = load_config(f"{exp_path}/config.yml")
        model = ActorCritic(**config["model"])
        ckpt = torch.load(f"{exp_path}/checkpoint.pth", weights_only=True, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device(device)
    model.to(device)

    print("Using fresh random model" if use_random else f"Loaded model from {exp_path}")

    results = []
    for i in range(num_games):
        if not quiet:
            print(f"\n--- Game {i + 1}/{num_games} ---")

        floor, health = run_game(
            model,
            device,
            delay=delay,
            verbose=not quiet,
            show_card_probs=not no_probs,
            greedy=greedy,
            check_legal=check_legal,
        )
        results.append((floor, health))

        if quiet:
            print(f"Game {i + 1}: Floor {floor}, Health {health}")

    if num_games > 1:
        avg_floor = sum(r[0] for r in results) / num_games
        avg_health = sum(r[1] for r in results) / num_games
        print(f"\n--- Summary ({num_games} games) ---")
        print(f"Average Floor: {avg_floor:.1f}")
        print(f"Average Final Health: {avg_health:.1f}")


if __name__ == "__main__":
    main()
