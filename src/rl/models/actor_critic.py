import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game.const import CARD_REWARD_NUM
from src.game.const import MAP_WIDTH
from src.game.const import MAX_MONSTERS
from src.game.const import MAX_SIZE_DECK
from src.game.const import MAX_SIZE_DISC_PILE
from src.game.const import MAX_SIZE_DRAW_PILE
from src.game.const import MAX_SIZE_HAND
from src.rl.encoding.card import get_encoding_card_dim
from src.rl.encoding.character import get_encoding_character_dim
from src.rl.encoding.energy import get_encoding_energy_dim
from src.rl.encoding.map_ import get_encoding_map_dim
from src.rl.encoding.monster import get_encoding_monster_dim
from src.rl.models.encoder_map import EncoderMap
from src.rl.models.transformer_entity import TransformerEntity


_ENCODING_DIM_CARD = get_encoding_card_dim()
_ENCODING_DIM_CHARACTER = get_encoding_character_dim()
_ENCODING_DIM_ENERGY = get_encoding_energy_dim()
_ENCODING_DIM_MAP = get_encoding_map_dim()
_ENCODING_DIM_MONSTER = get_encoding_monster_dim()


def _calculate_masked_mean_pooling(
    x: torch.Tensor, x_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_len = x_mask.shape

    # Zero out the padding tokens in the input tensor
    x_masked = x * x_mask.view(batch_size, sequence_len, 1)

    # Sum the embeddings of the non-padded tokens
    x_masked_sum = torch.sum(x_masked, dim=1)

    # Divide the sum by the actual sequence length to get the mean
    x_len = torch.sum(x_mask, dim=1, keepdim=True, dtype=torch.float32)
    x_masked_sum_pool = x_masked_sum / (x_len + 1e-9)

    return x_masked_sum_pool, x_len


# TODO: parametrize dimensions better, so they are more readable
class ActorCritic(nn.Module):
    def __init__(
        self,
        embedding_dim_card: int,
        embedding_dim_char: int,
        embedding_dim_monster: int,
        embedding_dim_energy: int,
        num_heads: int,
    ):
        super().__init__()

        self._embedding_dim_card = embedding_dim_card
        self._embedding_dim_char = embedding_dim_char
        self._embedding_dim_monster = embedding_dim_monster
        self._embedding_dim_energy = embedding_dim_energy
        self._num_heads = num_heads

        # Linear projections to map cards and monsters to a space w/ another dimension before
        # inputting them to the transformer
        self._projection_card = nn.Linear(_ENCODING_DIM_CARD, embedding_dim_card)
        self._projection_monster = nn.Linear(_ENCODING_DIM_MONSTER, embedding_dim_monster)

        # Transformers
        self._transformer_card = TransformerEntity(
            embedding_dim_card, embedding_dim_card, num_heads
        )
        self._transformer_monster = TransformerEntity(
            embedding_dim_monster, embedding_dim_monster, num_heads
        )

        # MLPs
        self._mlp_char = nn.Sequential(
            nn.Linear(_ENCODING_DIM_CHARACTER, embedding_dim_char),
            nn.ReLU(),
            nn.Linear(embedding_dim_char, embedding_dim_char),
            nn.ReLU(),
        )
        self._mlp_energy = nn.Sequential(
            nn.Linear(_ENCODING_DIM_ENERGY, embedding_dim_energy),
            nn.ReLU(),
            nn.Linear(embedding_dim_energy, embedding_dim_energy),
            nn.ReLU(),
        )

        # EncoderMap
        self._encoder_map = EncoderMap(
            _ENCODING_DIM_MAP[0],
            _ENCODING_DIM_MAP[1],
            _ENCODING_DIM_MAP[2],
            3,
        )

        # Final MLPs mapping to logits
        self._mlp_reward_select = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._mlp_reward_skip = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._mlp_card_select_or_discard = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self._mlp_monster_select = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._mlp_turn_end = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._mlp_map_select_rest_site_rest = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, MAP_WIDTH + 1),
        )
        self._mlp_rest_site_upgrade = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Critic
        self._mlp_critic = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        x_hand: torch.Tensor,
        x_hand_pad: torch.Tensor,
        x_hand_active: torch.Tensor,
        x_draw: torch.Tensor,
        x_draw_pad: torch.Tensor,
        x_disc: torch.Tensor,
        x_disc_pad: torch.Tensor,
        x_deck: torch.Tensor,
        x_deck_pad: torch.Tensor,
        x_reward: torch.Tensor,
        x_char: torch.Tensor,
        x_monster: torch.Tensor,
        x_monster_pad: torch.Tensor,
        x_energy: torch.Tensor,
        x_map: torch.Tensor,
        x_valid_action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_hand.shape[0]

        # Card encodings. Start with hand
        x_hand_pad = x_hand_pad.to(torch.bool)
        x_hand = self._projection_card(x_hand)
        x_hand = self._transformer_card(x_hand, ~x_hand_pad)

        # Draw pile
        x_draw_pad = x_draw_pad.to(torch.bool)
        x_draw = self._projection_card(x_draw)
        x_draw = self._transformer_card(x_draw, ~x_draw_pad)

        # Discard pile
        x_disc_pad = x_disc_pad.to(torch.bool)
        x_disc = self._projection_card(x_disc)
        x_disc = self._transformer_card(x_disc, ~x_disc_pad)

        # Deck
        x_deck_pad = x_deck_pad.to(torch.bool)
        x_deck = self._projection_card(x_deck)
        x_deck = self._transformer_card(x_deck, ~x_deck_pad)

        # Card rewards
        x_reward_pad = torch.ones(batch_size, CARD_REWARD_NUM, dtype=torch.bool)
        x_reward = self._projection_card(x_reward)
        x_reward = self._transformer_card(x_reward, ~x_reward_pad)

        # Monsters
        x_monster_pad = x_monster_pad.to(torch.bool)
        x_monster = self._projection_monster(x_monster)
        x_monster = self._transformer_monster(x_monster, ~x_monster_pad)

        # Aggregate representations
        x_hand_mean, x_hand_len = _calculate_masked_mean_pooling(x_hand, x_hand_pad)
        x_draw_mean, x_draw_len = _calculate_masked_mean_pooling(x_draw, x_draw_pad)
        x_disc_mean, x_disc_len = _calculate_masked_mean_pooling(x_disc, x_disc_pad)
        x_deck_mean, x_deck_len = _calculate_masked_mean_pooling(x_deck, x_deck_pad)
        x_reward_mean, _ = _calculate_masked_mean_pooling(x_reward, x_reward_pad)
        x_hand_len /= MAX_SIZE_HAND
        x_draw_len /= MAX_SIZE_DRAW_PILE
        x_disc_len /= MAX_SIZE_DISC_PILE
        x_deck_len /= MAX_SIZE_DECK

        x_monster_mean, x_monster_len = _calculate_masked_mean_pooling(x_monster, x_monster_pad)
        x_monster_len /= MAX_MONSTERS

        # Map
        x_map = self._encoder_map(x_map)

        # Character
        x_char = self._mlp_char(x_char)

        # Energy
        x_energy = self._mlp_energy(x_energy)

        # Global aggregate representations
        x_combat = torch.cat(
            [
                x_hand_mean,
                x_draw_mean,
                x_disc_mean,
                x_hand_len,
                x_draw_len,
                x_disc_len,
                x_monster_mean,
                x_monster_len,
                x_char,
                x_energy,
                x_map,
            ],
            dim=1,
        )
        x_map_select_and_rest_site = torch.cat(
            [
                x_char,
                x_map,
                x_deck_mean,
                x_deck_len,
            ],
            dim=1,
        )
        x_reward_select = torch.cat(
            [
                x_char,
                x_map,
                x_deck_mean,
                x_deck_len,
                x_reward_mean,
            ],
            dim=1,
        )

        # Calculate action logits. Start with reward selection
        x_logit_reward_select = torch.cat(
            [
                x_reward,
                x_reward_select.view(batch_size, 1, -1).expand(batch_size, CARD_REWARD_NUM, -1),
            ],
            dim=2,
        )
        x_logit_reward_skip = x_reward_select

        # Combat
        x_logit_card_select_or_discard = torch.cat(
            [
                x_hand,
                x_combat.view(batch_size, 1, -1).expand(batch_size, MAX_SIZE_HAND, -1),
            ],
            dim=2,
        )
        x_hand_active = torch.sum(
            (
                x_hand_active.view(batch_size, MAX_SIZE_HAND, 1).expand(
                    batch_size, MAX_SIZE_HAND, self._embedding_dim_card
                )
                * x_hand
            ),
            dim=1,
        )
        x_logit_monster_select = torch.cat(
            [
                x_monster,
                x_hand_active.view(batch_size, 1, -1).expand(batch_size, MAX_MONSTERS, -1),
                x_combat.view(batch_size, 1, -1).expand(batch_size, MAX_MONSTERS, -1),
            ],
            dim=2,
        )
        x_logit_turn_end = x_combat

        # Rest site
        x_logit_rest_site_upgrade = torch.cat(
            [
                x_deck,
                x_map_select_and_rest_site.view(batch_size, 1, -1).expand(
                    batch_size, MAX_SIZE_DECK, -1
                ),
            ],
            dim=2,
        )

        # Calculate actions
        x_logit_reward_select = self._mlp_reward_select(x_logit_reward_select)
        x_logit_reward_skip = self._mlp_reward_skip(x_logit_reward_skip)
        x_logit_card_select_or_discard = self._mlp_card_select_or_discard(
            x_logit_card_select_or_discard
        )
        x_logit_monster_select = self._mlp_monster_select(x_logit_monster_select)
        x_logit_turn_end = self._mlp_turn_end(x_logit_turn_end)
        x_logit_map_select_rest_site_rest = self._mlp_map_select_rest_site_rest(
            x_map_select_and_rest_site
        )
        x_logit_rest_site_upgrade = self._mlp_rest_site_upgrade(x_logit_rest_site_upgrade)

        # Concatenate actions into a single 1D tensor
        x_logit_all = torch.cat(
            [
                torch.flatten(x_logit_reward_select, start_dim=1),
                x_logit_reward_skip,
                torch.flatten(x_logit_card_select_or_discard.permute(0, 2, 1), start_dim=1),
                torch.flatten(x_logit_monster_select, start_dim=1),
                x_logit_turn_end,
                x_logit_map_select_rest_site_rest,
                torch.flatten(x_logit_rest_site_upgrade, start_dim=1),
            ],
            dim=1,
        )

        # Apply valid action mask and get action probabilities w/ softmax
        x_logit_all_mask = x_logit_all.masked_fill(x_valid_action_mask == 0, float("-inf"))
        x_actor = F.softmax(x_logit_all_mask, dim=-1)

        # Critic
        x_critic = torch.cat(
            [
                x_combat,
                x_deck_mean,
                x_deck_len,
                x_reward_mean,
            ],
            dim=1,
        )
        x_critic = self._mlp_critic(x_critic)

        return x_actor, x_critic
