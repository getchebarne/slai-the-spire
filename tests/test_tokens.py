import warnings
from types import SimpleNamespace

import pytest
import slai
import torch

from src.rl.actions import engine_action
from src.rl.constants import MAP_HEIGHT
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_CARD_REWARD_ROLL
from src.rl.constants import MAX_COMBAT_CARD_REWARD
from src.rl.constants import MAX_SIZE_CARD_PILE_DISCARD
from src.rl.constants import MAX_SIZE_CARD_PILE_DRAW
from src.rl.encoding import encode_batch_game_state
from src.rl.schema.context import ENCODING_DIM_CONTEXT
from src.rl.schema.context import FOCUS
from src.rl.schema.context import focus_view
from src.rl.tokens import get_entities_for_token_kind
from src.rl.tokens import iter_tokens
from src.rl.types import TokenKind


def test_token_order_decodes_card_selections_and_monster_targets():
    env = slai.GameEnv(0, fast_mode=True, neow=False)
    env.reset(9)
    state, done = env.step(env.get_legal_actions()[0])
    assert not done and state.combat is not None
    legal_actions = env.get_legal_actions()
    tokens = list(iter_tokens(state))
    monsters = get_entities_for_token_kind(TokenKind.MONSTERS, state)
    batch = encode_batch_game_state([state], [legal_actions], torch.device("cpu"))
    assert batch.tokens.rows.x_kind.tolist() == [int(kind) for kind, _ in tokens]

    targeted = [
        action
        for action in legal_actions
        if action.action_type == slai.ActionType.CardPlay and len(action.ids) == 2
    ]
    assert targeted
    for action in targeted:
        selection = next(
            row
            for row, (kind, entity) in enumerate(tokens)
            if kind == TokenKind.CARD_PILE_HAND and entity.id == action.ids[0]
        )
        target = next(row for row, monster in enumerate(monsters) if monster.id == action.ids[1])
        assert engine_action(int(action.action_type), selection, target, state) == action
    with pytest.raises(IndexError, match="Monster target rank"):
        engine_action(int(slai.ActionType.CardPlay), selection, len(monsters), state)


def test_card_caps_preserve_draw_top_and_the_wider_first_reward_bundle():
    count = max(MAX_SIZE_CARD_PILE_DRAW, MAX_SIZE_CARD_PILE_DISCARD, MAX_CARD_REWARD_ROLL) + 3
    cards = [SimpleNamespace(id=index) for index in range(count)]
    state = SimpleNamespace(
        combat=SimpleNamespace(card_pile_draw=cards, card_pile_discard=cards),
        reward=SimpleNamespace(cards=[cards, cards]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        draw = get_entities_for_token_kind(TokenKind.CARD_PILE_DRAW, state)
        discard = get_entities_for_token_kind(TokenKind.CARD_PILE_DISCARD, state)
    assert draw == cards[-MAX_SIZE_CARD_PILE_DRAW:]
    assert discard == cards[:MAX_SIZE_CARD_PILE_DISCARD]
    assert (
        get_entities_for_token_kind(TokenKind.CARD_REWARD_0, state) == cards[:MAX_CARD_REWARD_ROLL]
    )
    assert (
        get_entities_for_token_kind(TokenKind.CARD_REWARD_1, state)
        == cards[:MAX_COMBAT_CARD_REWARD]
    )
    assert get_entities_for_token_kind(TokenKind.CARD_REWARD_2, state) == []


def test_room_tokens_follow_column_order_edges_and_wing_boots():
    rooms = [[None] * MAP_WIDTH for _ in range(MAP_HEIGHT)]
    rooms[0][2] = SimpleNamespace(id=0, edges=[4, 1])
    for col in (1, 3, 4):
        rooms[1][col] = SimpleNamespace(id=col, edges=[])
    state = SimpleNamespace(relics=[], map=SimpleNamespace(rooms=rooms, y_current=0, x_current=2))
    assert get_entities_for_token_kind(TokenKind.ROOMS, state) == [rooms[1][1], rooms[1][4]]

    boots = SimpleNamespace(name=slai.RelicName.WingBoots, counter=1)
    state.relics = [boots]
    assert get_entities_for_token_kind(TokenKind.ROOMS, state) == [
        rooms[1][1],
        rooms[1][3],
        rooms[1][4],
    ]
    boots.counter = 0
    assert get_entities_for_token_kind(TokenKind.ROOMS, state) == [rooms[1][1], rooms[1][4]]
    state.map.y_current, state.map.x_current = None, None
    assert get_entities_for_token_kind(TokenKind.ROOMS, state) == [rooms[0][2]]
    state.map.y_current = MAP_HEIGHT - 1
    assert get_entities_for_token_kind(TokenKind.ROOMS, state) == []


def test_focus_view_aliases_only_the_focus_columns():
    x_context = torch.zeros(2, ENCODING_DIM_CONTEXT)
    x_focus = focus_view(x_context)
    x_focus.fill_(1)
    assert bool((x_context[:, FOCUS] == 1).all())
    assert not x_context[:, : FOCUS.start].any()
    assert not x_context[:, FOCUS.stop :].any()
    x_context[1, FOCUS.start] = 3
    assert x_focus[1, 0] == 3
