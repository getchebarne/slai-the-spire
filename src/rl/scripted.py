from slai import Action
from slai import ActionType
from slai import ChestKind
from slai import EventName
from slai import GameEnv
from slai import GameState
from slai import RelicName
from slai import RelicTier
from slai import Reward

from src.rl.encoding.mask import potion_can_arrive
from src.rl.encoding.mask import potion_discard_wasted


# Most scripted actions per policy step, so forced turn ends can't hang a worker
_ACTIONS_SCRIPTED_MAX = 32

# A Reward's gold costs no discount: every other action costs one γ
_ACTION_TYPES_GOLD = frozenset({ActionType.RewardTakeGold, ActionType.RewardTakeGoldStolen})

_ACTION_TYPES_POTION = frozenset({ActionType.PotionDiscard, ActionType.PotionUse})

_ACTION_TYPES_ROOM = frozenset({ActionType.RoomExit, ActionType.RoomSelect})

# The Events whose first option runs for the policy, with the Relic that leaves it to the policy:
# Ectoplasm blocks Golden Idol's gold, Du-Vu Doll gives Strength for the Curses a drink removes
_EVENTS_OPTION_FIRST_SCRIPTED: dict[EventName, RelicName | None] = {
    EventName.Duplicator: None,
    EventName.GoldenIdol: RelicName.Ectoplasm,
    EventName.Purifier: None,
    EventName.TheDivineFountain: RelicName.DuVuDoll,
    EventName.Transmogrifier: None,
    EventName.UpgradeShrine: None,
}

# Free to take from a Reward; boss Relics exclude each other, a Special one can carry a cost
_RELIC_TIERS_SCRIPTED = frozenset({RelicTier.Common, RelicTier.Uncommon, RelicTier.Rare})

# The Special Relics taken from a Reward; Cursed Tome's books aren't, as Necronomicon adds a Curse
_RELICS_SPECIAL_SCRIPTED = frozenset({RelicName.OddMushroom, RelicName.RedMask})

# Bottled Relics fix an opening hand, Juzu Bracelet changes Rooms, the others fill the hand
_RELICS_UNSCRIPTED = frozenset(
    {
        RelicName.BottledFlame,
        RelicName.BottledLightning,
        RelicName.BottledTornado,
        RelicName.DeadBranch,
        RelicName.JuzuBracelet,
        RelicName.NinjaScroll,
    }
)


def _reward_take_free(game_state: GameState, reward: Reward, action: Action) -> bool:
    """Return whether `action` takes something from the Reward that costs nothing.

    Parameters
    ----------
    game_state : GameState
        The state `action` is legal in.
    reward : Reward
        The state's Reward.
    action : Action
        One legal action.

    Returns
    -------
    bool
        True for the gold, for a scripted Special Relic, for a Relic of a scripted tier whose name
        isn't unscripted, as long as no Sapphire Key is linked to it and it isn't Tiny Chest under
        Cursed Key, and for a Potion when every offered one fits in the belt, or under Sozu, where
        a take obtains nothing.
    """
    if action.action_type in _ACTION_TYPES_GOLD:
        return True
    match action.action_type:
        case ActionType.RewardTakeRelic:
            relic = next(relic for relic in reward.relics if relic.id == action.ids[0])
            relic_linked = reward.relic_linked  # Taking it gives up the Sapphire Key

            # Under Cursed Key, each Chest that Tiny Chest adds costs a Curse
            tiny_chest_cursed = relic.name == RelicName.TinyChest and any(
                relic_held.name == RelicName.CursedKey for relic_held in game_state.relics
            )
            return (
                (relic.tier in _RELIC_TIERS_SCRIPTED or relic.name in _RELICS_SPECIAL_SCRIPTED)
                and relic.name not in _RELICS_UNSCRIPTED
                and (relic_linked is None or relic_linked.id != relic.id)
                and not tiny_chest_cursed
            )
        case ActionType.RewardTakePotion:
            num_slots_free = game_state.potion_slots_max - len(game_state.potions)
            return len(reward.potions) <= num_slots_free or any(
                relic.name == RelicName.Sozu for relic in game_state.relics
            )
        case _:
            return False


def _action_scripted(game_state: GameState, actions: list[Action]) -> Action | None:
    """Return the action to take for the policy, if any.

    Parameters
    ----------
    game_state : GameState
        The state `actions` are legal in.
    actions : list[Action]
        The state's legal actions, in the engine's order.

    Returns
    -------
    Action | None
        The action to take, or None if the policy has a choice.
    """
    # A Reward's free takes, in the engine's order: Relics first, so a Potion Belt's slots count
    reward = game_state.reward
    if reward is not None:
        for action in actions:
            if _reward_take_free(game_state, reward, action):
                return action

    # A Chest's opening, unless it adds Cursed Key's Curse, which a boss Chest's never does
    chest = game_state.chest
    if chest is not None and (
        chest.kind == ChestKind.Boss
        or not any(relic.name == RelicName.CursedKey for relic in game_state.relics)
    ):
        for action in actions:
            if action.action_type == ActionType.ChestOpen:
                return action

    # An Event's first option, unless a held Relic leaves it to the policy
    event = game_state.event
    if (
        event is not None
        and event.name in _EVENTS_OPTION_FIRST_SCRIPTED
        and not any(
            relic.name == _EVENTS_OPTION_FIRST_SCRIPTED[event.name] for relic in game_state.relics
        )
    ):
        for action in actions:
            if (
                action.action_type == ActionType.EventOptionSelect
                and action.ids[0] == event.options[0].id
            ):
                return action

    # The only action left once wasted discards are out
    actions_open = [action for action in actions if not potion_discard_wasted(game_state, action)]
    if len(actions_open) == 1:
        return actions_open[0]

    # Leaving a Room or entering the next: a Potion can wait, unless another can arrive here
    actions_no_potion = [
        action for action in actions if action.action_type not in _ACTION_TYPES_POTION
    ]
    if (
        len(actions_no_potion) == 1
        and actions_no_potion[0].action_type in _ACTION_TYPES_ROOM
        and not potion_can_arrive(game_state)
    ):
        return actions_no_potion[0]
    return None


def step_scripted(env: GameEnv, action: Action) -> tuple[GameState, bool, int]:
    """Run `env.step(action)`, then the scripted actions until the policy has a choice.

    All of them run inside this one policy step, whose discount counts them.

    Parameters
    ----------
    env : GameEnv
        The game to step.
    action : Action
        The policy's action.

    Returns
    -------
    game_state : GameState
        The state after the last action taken.
    terminated : bool
        Whether the game ended.
    num_discounts : int
        The actions taken, gold takes aside: k in the step's discount, γ^k.
    """
    game_state, terminated = env.step(action)
    num_discounts = int(action.action_type not in _ACTION_TYPES_GOLD)
    for _ in range(_ACTIONS_SCRIPTED_MAX):
        if terminated:
            break
        action_scripted = _action_scripted(game_state, env.get_legal_actions())
        if action_scripted is None:
            break
        game_state, terminated = env.step(action_scripted)
        num_discounts += action_scripted.action_type not in _ACTION_TYPES_GOLD
    return game_state, terminated, num_discounts
