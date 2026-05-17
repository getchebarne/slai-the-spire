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

_INTENT_BLOCK_KINDS = frozenset(
    {IntentKind.Block, IntentKind.AttackBlock, IntentKind.BlockBuff}
)
_INTENT_BUFF_KINDS = frozenset(
    {IntentKind.Buff, IntentKind.AttackBuff, IntentKind.BlockBuff}
)
_INTENT_DEBUFF_KINDS = frozenset(
    {IntentKind.Debuff, IntentKind.AttackDebuff, IntentKind.DebuffPowerful}
)

from src.rl.action_space.masks import MaskBatch
from src.rl.action_space.masks import get_mask_batch
from src.rl.action_space.masks import is_card_playable
from src.rl.action_space.types import HeadTypePrimary
from src.rl.constants import ASCENSION_LEVEL
from src.rl.encoding.state import XGameState
from src.rl.encoding.state import encode_batch_view_game_state
from src.rl.models import ActorCritic
from src.rl.models.heads import get_grouped_probs
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
        f"{_MOD_ABBR.get(m.kind, _variant_name(m.kind).lower())} {m.stacks}"
        for m in mods
    )


def _fmt_action(action: slai.Action) -> str:
    """Format a flat slai.Action for the log. Dispatches on
    `action.action_type` and pulls fields by position from `action.idxs`
    using the schema documented in `slai.ACTION_SPEC_REGISTRY`."""
    at = action.action_type
    i = action.idxs
    if at == slai.ActionType.CardPlay:
        tgt = f", target={i[1]}" if len(i) > 1 else ""
        return f"CardPlay(hand={i[0]}{tgt})"
    if at == slai.ActionType.CardDiscard:
        return f"CardDiscard(indices={list(i)})"
    if at == slai.ActionType.CardRetain:
        return f"CardRetain(indices={list(i)})"
    if at == slai.ActionType.CardSetup:
        return f"CardSetup(hand={i[0]})"
    if at == slai.ActionType.CardNightmare:
        return f"CardNightmare(hand={i[0]})"
    if at == slai.ActionType.RoomSelect:
        return f"RoomSelect(col={i[0]})"
    if at == slai.ActionType.CardRewardSelect:
        return f"CardRewardSelect(idx={i[0]})"
    if at == slai.ActionType.RelicRewardSelect:
        return f"RelicRewardSelect(idx={i[0]})"
    if at == slai.ActionType.RestSiteCardUpgrade:
        return f"RestSiteCardUpgrade(deck={i[0]})"
    return at.name  # EndTurn, *Skip, RestSiteRest


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
    return f"[{i}] {m.display_name}: {hp}{blk}  → {intent_str}"


def _format_view(view: slai.GameState) -> str:
    """Compact textual rendering of the game state.

    Two-column layout: character/hand on the left, monsters right-padded
    so name/HP/intent columns line up vertically (mirrors `play/__main__.py`).
    """
    lines = [f"Phase: {type(view.phase).__name__}"]

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

    if view.monsters:
        # Right-pad monster rows so they all start at the same column —
        # matches play/__main__.py's `block_x = right_edge - max(...)`.
        rendered = [_enemy_row(i, m) for i, m in enumerate(view.monsters)]
        max_w = max(_ansi_len(r) for r in rendered)
        left_pad = max(0, N_COL - max_w)
        lines.append("Monsters:")
        for i, (m, row) in enumerate(zip(view.monsters, rendered)):
            lines.append(" " * left_pad + row)
            if m.modifiers:
                lines.append(" " * left_pad + f"  modifiers: {_fmt_modifiers(m.modifiers)}")

    lines.append(f"Energy: {view.energy.current}/{view.energy.max}")
    if view.hand:
        lines.append("Hand:")
        for i, c in enumerate(view.hand):
            tgt = " (target)" if c.requires_target else ""
            playable = "" if is_card_playable(c, view.energy.current) else " (unplayable)"
            lines.append(f"  [{i}] {c.display_name} cost={c.cost}{tgt}{playable}")

    if view.rewards_card:
        lines.append(f"Card rewards: {[c.display_name for c in view.rewards_card]}")
    if view.rewards_relic:
        lines.append(f"Relic rewards: {[_variant_name(r.name) for r in view.rewards_relic]}")
    return "\n".join(lines)


def get_card_probabilities(
    model: ActorCritic,
    x_game_state: XGameState,
    mask_batch: MaskBatch,
) -> torch.Tensor:
    """
    Get grouped probabilities for cards in hand from the card play head.

    Uses grouped softmax: identical cards are deduplicated so each card type
    gets a single probability (not split across duplicates).

    Returns:
        Tensor of per-position grouped probabilities (MAX_HAND_SIZE,).
        Identical cards share the same probability value.
    """
    # Run core encoder
    core_out = model.core(x_game_state)

    # Get selection mask for COMBAT_DEFAULT (which is the card play mask)
    mask = mask_batch.selection_masks[int(HeadTypePrimary.COMBAT_DEFAULT)]  # (1, MAX_HAND_SIZE)

    # Run card play head without sampling to get logits
    head_out = model.head_card_play(
        core_out.x_hand, core_out.x_global, mask, sample=False,
        group_ids=mask_batch.hand_group_ids,
    )

    # Get grouped probabilities (deduplicates identical cards via the
    # same group_ids the policy uses for sampling).
    probs = get_grouped_probs(head_out.logits, group_ids=mask_batch.hand_group_ids)

    return probs[0]  # Return first (only) batch item


def format_card_probabilities(
    view: slai.GameState,
    probs: torch.Tensor,
) -> str:
    """Format card probabilities grouped by card type."""
    seen: dict[str, dict] = {}
    order: list[str] = []

    for idx, card in enumerate(view.hand):
        display = card.display_name
        if display not in seen:
            prob = probs[idx].item()
            if prob != prob:  # NaN guard
                prob = 0.0
            playable = is_card_playable(card, view.energy.current)
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
    device: torch.device,
    show_card_probs: bool = False,
    greedy: bool = False,
) -> tuple[object, str | None]:
    """
    Get an action from the model for the given view.

    Args:
        greedy: If True, use argmax instead of sampling (deterministic)

    Returns:
        (action, card_probs_str). action is a `slai.Action` instance.
    """
    x_game_state = encode_batch_view_game_state([view], device)
    mask_batch = get_mask_batch([view], device)

    with torch.no_grad():
        output = model.forward_single(x_game_state, mask_batch, sample=not greedy)

        card_probs_str = None
        has_playable = any(is_card_playable(c, view.energy.current) for c in view.hand)
        if (
            show_card_probs
            and isinstance(view.phase, slai.Phase.CombatDefault)
            and view.hand
            and has_playable
        ):
            probs = get_card_probabilities(model, x_game_state, mask_batch)
            card_probs_str = format_card_probabilities(view, probs)

    return output.to_action(), card_probs_str


def run_game(
    model: ActorCritic,
    device: torch.device,
    delay: float = 0.5,
    verbose: bool = True,
    show_card_probs: bool = True,
    greedy: bool = False,
) -> tuple[int, int]:
    """
    Run a single game with the trained model.

    Returns:
        (final_floor, final_health)
    """
    env = slai.GameEnv(ascension=ASCENSION_LEVEL)
    obs = env.reset(seed=random.randint(0, 2**31 - 1))

    step_count = 0
    terminated = False
    while not terminated:
        if verbose:
            print(_format_view(obs))
            print("-" * N_COL)

        action, card_probs_str = get_action_from_model(
            model,
            obs,
            device,
            show_card_probs=verbose and show_card_probs,
            greedy=greedy,
        )

        if verbose:
            if card_probs_str:
                print(card_probs_str)
                print("-" * N_COL)
            print(f"Action: {_fmt_action(action)}")
            print("-" * N_COL)
            time.sleep(delay)

        obs, terminated = env.step(action)
        step_count += 1

    final_floor = obs.map.y_current or 0
    final_health = obs.character.health

    if verbose:
        print(f"\n{'=' * N_COL}")
        print("GAME OVER")
        print(f"Final Floor: {final_floor}")
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
def main(
    exp_path: str,
    delay: float,
    device: str,
    num_games: int,
    quiet: bool,
    no_probs: bool,
    greedy: bool,
):
    """Test a trained agent by running games."""
    config = load_config(f"{exp_path}/config.yml")
    model = ActorCritic(**config["model"])
    model.load_state_dict(torch.load(f"{exp_path}/model.pth", weights_only=True))
    model.eval()

    device = torch.device(device)
    model.to(device)

    print(f"Loaded model from {exp_path}")

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
