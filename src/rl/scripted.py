from slai import Action
from slai import ActionType
from slai import ChestKind
from slai import EventName
from slai import GameEnv
from slai import GameState
from slai import Potion
from slai import PotionName
from slai import Relic
from slai import RelicName
from slai import RelicTier
from slai import Reward

from src.rl.tokens import get_entities
from src.rl.tokens import relic_held
from src.rl.utils import action_masked
from src.rl.utils import curse_useful


# Most scripted actions per policy step, so forced turn ends can't hang a worker
_ACTIONS_SCRIPTED_MAX = 32

# A Reward's takes move no game time, so they cost no discount; every other action costs one
_ACTION_TYPES_REWARD_TAKE = frozenset(
    {
        ActionType.RewardSingingBowl,
        ActionType.RewardTakeCard,
        ActionType.RewardTakeGold,
        ActionType.RewardTakeGoldStolen,
        ActionType.RewardTakePotion,
        ActionType.RewardTakeRelic,
        ActionType.RewardTakeSapphireKey,
    }
)

# The Events whose first option runs for the policy, with the Relic that leaves it to the policy
_EVENTS_OPTION_FIRST_SCRIPTED: dict[EventName, RelicName | None] = {
    EventName.GoldenIdol: RelicName.Ectoplasm,
    EventName.UpgradeShrine: None,
}

# Free to take from a Reward; boss Relics exclude each other, a Special one can carry a cost
_RELIC_TIERS_SCRIPTED = frozenset({RelicTier.Common, RelicTier.Uncommon, RelicTier.Rare})

# The Potions Mark of the Bloom voids: Fairy can't revive, Regeneration can't heal
_POTIONS_VOIDED_BY_MARK_OF_THE_BLOOM = frozenset({PotionName.Fairy, PotionName.Regeneration})

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
        Cursed Key, Omamori with a Relic that gives a Curse a use held, or Peace Pipe with Coffee
        Dripper held, and for a
        Potion when every offered one fits in the belt, no other Entropic Brew is held or offered,
        and it isn't one that Mark of the Bloom voids while held.
    """
    match action.action_type:
        # A Reward's gold is always free to take
        case ActionType.RewardTakeGold | ActionType.RewardTakeGoldStolen:
            return True

        case ActionType.RewardTakeRelic:
            (relic,) = get_entities(game_state, action.ids[:1], Relic)
            id_relic_linked = reward.id_relic_linked

            # Under Cursed Key, each Chest that Tiny Chest adds costs a Curse
            tiny_chest_cursed = relic.name == RelicName.TinyChest and relic_held(
                game_state, RelicName.CursedKey
            )

            # Omamori would negate the Curses a held Relic gives a use
            omamori_negates_wanted = relic.name == RelicName.Omamori and curse_useful(game_state)

            # Under Coffee Dripper, Toke can be a Rest Site's only option, a forced Card removal
            peace_pipe_forced = relic.name == RelicName.PeacePipe and relic_held(
                game_state, RelicName.CoffeeDripper
            )
            
            return (
                (relic.tier in _RELIC_TIERS_SCRIPTED or relic.name in _RELICS_SPECIAL_SCRIPTED)
                and relic.name not in _RELICS_UNSCRIPTED
                and id_relic_linked != relic.id
                and not tiny_chest_cursed
                and not omamori_negates_wanted
                and not peace_pipe_forced
            )
        
        case ActionType.RewardTakePotion:
            (potion,) = get_entities(game_state, action.ids[:1], Potion)

            # Another Entropic Brew, held or offered, could fill the free slots with random Potions
            entropic_brew_other = any(
                potion_other.name == PotionName.EntropicBrew
                for potion_other in get_entities(
                    game_state, (*game_state.id_potions, *reward.id_potions), Potion
                )
                if potion_other.id != potion.id
            )

            # Under Mark of the Bloom it would only fill a slot
            potion_voided = potion.name in _POTIONS_VOIDED_BY_MARK_OF_THE_BLOOM and relic_held(
                game_state, RelicName.MarkOfTheBloom
            )
            num_slots_free = game_state.potion_slots_max - len(game_state.id_potions)
            
            return (
                not entropic_brew_other
                and not potion_voided
                and len(reward.id_potions) <= num_slots_free
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
    # Only what the masks leave open, so no rule takes an action the policy couldn't
    actions_open = [action for action in actions if not action_masked(game_state, action)]

    # A Reward's free takes, in the engine's order: Relics first, so a Potion Belt's slots count
    reward = game_state.reward
    if reward is not None:
        for action in actions_open:
            if _reward_take_free(game_state, reward, action):
                return action

    # A Chest's opening, unless it adds Cursed Key's Curse, which a boss Chest's never does
    chest = game_state.chest
    if chest is not None and (
        chest.chest_kind == ChestKind.Boss or not relic_held(game_state, RelicName.CursedKey)
    ):
        for action in actions_open:
            if action.action_type == ActionType.ChestOpen:
                return action

    # An Event's first option, unless a held Relic leaves it to the policy
    event = game_state.event
    if event is not None and event.name in _EVENTS_OPTION_FIRST_SCRIPTED:
        relic_name_unscripting = _EVENTS_OPTION_FIRST_SCRIPTED[event.name]
        if relic_name_unscripting is None or not relic_held(game_state, relic_name_unscripting):
            for action in actions_open:
                if (
                    action.action_type == ActionType.EventOptionSelect
                    and action.ids[0] == event.id_event_options[0]
                ):
                    return action

    # The only action left
    if len(actions_open) == 1:
        return actions_open[0]
    
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
        The actions taken, Reward takes aside: k in the step's discount, γ^k.
    """
    game_state, terminated = env.step(action)
    num_discounts = int(action.action_type not in _ACTION_TYPES_REWARD_TAKE)
    
    for _ in range(_ACTIONS_SCRIPTED_MAX):
        if terminated:
            break

        action_scripted = _action_scripted(game_state, env.get_legal_actions())
        if action_scripted is None:
            break
        
        game_state, terminated = env.step(action_scripted)
        num_discounts += action_scripted.action_type not in _ACTION_TYPES_REWARD_TAKE
    
    return game_state, terminated, num_discounts
