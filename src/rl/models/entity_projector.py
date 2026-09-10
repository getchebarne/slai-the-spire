import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.encoding.card import ENCODING_DIM_CARD
from src.rl.encoding.character import ENCODING_DIM_CHARACTER
from src.rl.encoding.effect import ENCODING_DIM_EFFECT
from src.rl.encoding.event import ENCODING_DIM_EVENT_OPTION
from src.rl.encoding.health_block import ENCODING_DIM_HEALTH_BLOCK
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIER
from src.rl.encoding.monster import ENCODING_DIM_MONSTER
from src.rl.encoding.potion import ENCODING_DIM_POTION
from src.rl.encoding.relic import ENCODING_DIM_RELIC
from src.rl.types import OWNER_PENDING
from src.rl.types import EntityKind
from src.rl.types import X_Entity
from src.rl.types import X_GameState


class EntityProjector(nn.Module):
    """Factored (AlphaStar-style) entity embedder.

    First layer: each type has a private block. The health/block family the
    monsters and the character share arrives as its own rows
    (`X_GameState.health_block`) and goes through ONE shared block whose output
    is added to the type's private projection (per-block linear-and-sum, as in
    AlphaStar's unit encoder), so both consumers' gradients accumulate into one
    parameter. The second layer + LayerNorm are shared by all token types and
    run on one fused row tensor.
    Rooms come deep-processed from the GNN and join as the seventh block. The
    output is the window as ONE X_Entity — the six blocks then the rooms, the kind
    row cat'd beside the features — plus the pooled pending halt.

    Effects no longer live in the entity row. They arrive as their own rows
    (`X_GameState.effects`), go through `_effects_mlp` as ONE batched GEMM for
    the whole minibatch, and are sum-pooled into their owning token — named by
    (idx_type, offset): the owner's type and its packed position in that type's
    block — before the shared ReLU, so an entity's effects share its
    normalization. The MLP must stay >=2 layers: a linear encoder would make
    pooling additive in the input factors, and `kind x target x amount` binding
    would not survive the sum.
    """

    def __init__(self, dim_embedding: int, dim_map: int) -> None:
        super().__init__()

        self._dim_embedding = dim_embedding
        self._dim_map = dim_map

        # Per-`EntityKind` private blocks
        self._pre_card = nn.Linear(ENCODING_DIM_CARD, dim_embedding)
        self._pre_relic = nn.Linear(ENCODING_DIM_RELIC, dim_embedding)
        self._pre_potion = nn.Linear(ENCODING_DIM_POTION, dim_embedding)
        self._pre_event_option = nn.Linear(ENCODING_DIM_EVENT_OPTION, dim_embedding)
        self._pre_monster = nn.Linear(ENCODING_DIM_MONSTER, dim_embedding)
        self._pre_character = nn.Linear(ENCODING_DIM_CHARACTER, dim_embedding)
        self._pre_room = nn.Linear(
            dim_map, dim_embedding
        )  # Rooms come already-processed by `MapGNN`

        # Health+Block's mapper is shared between the Character and the Monsters
        self._pre_health_block = nn.Linear(ENCODING_DIM_HEALTH_BLOCK, dim_embedding, bias=False)

        # Per-Effect embedder w/ >=2 layers is a must due to OHE-summing semantics
        self._effects_mlp = nn.Sequential(
            nn.Linear(ENCODING_DIM_EFFECT, 2 * dim_embedding),
            nn.ReLU(),
            nn.Linear(2 * dim_embedding, dim_embedding, bias=False),
        )

        # Per-Modifier embedder, same shape / logic as the Effects'
        self._modifiers_mlp = nn.Sequential(
            nn.Linear(ENCODING_DIM_MODIFIER, 2 * dim_embedding),
            nn.ReLU(),
            nn.Linear(2 * dim_embedding, dim_embedding, bias=False),
        )

        # Shared second layer (after ReLU, on the fused all-types row tensor)
        self._post_shared = nn.Linear(dim_embedding, dim_embedding)

        self._layer_norm = nn.LayerNorm(dim_embedding)

    def forward(
        self, x_game_state: X_GameState, x_rooms: X_Entity
    ) -> tuple[X_Entity, torch.Tensor]:
        batch_size = x_game_state.cards.x.shape[0]
        device = x_game_state.cards.x.device

        # Health+Block
        x_batch_hb, x_slot_hb = torch.nonzero(
            x_game_state.health_block.x_entity_kind >= 0, as_tuple=True
        )
        x_hb = self._pre_health_block(x_game_state.health_block.x[x_batch_hb, x_slot_hb])
        x_entity_kind_hb = x_game_state.health_block.x_entity_kind[x_batch_hb, x_slot_hb]
        x_offset_hb = x_game_state.health_block.x_offset[x_batch_hb, x_slot_hb]

        # Entities / Cards
        x_cards = self._project_type(x_game_state.cards, self._pre_card)

        # Entities / Relics
        x_relics = self._project_type(x_game_state.relics, self._pre_relic)

        # Entities / Potions
        x_potions = self._project_type(x_game_state.potions, self._pre_potion)

        # Entities / Monsters
        x_monsters = self._project_type(x_game_state.monsters, self._pre_monster)
        # Health+Block
        x_sel = x_entity_kind_hb == EntityKind.MONSTER
        x_pos = x_batch_hb[x_sel] * x_monsters.shape[1] + x_offset_hb[x_sel]
        x_monsters.view(-1, self._dim_embedding).index_add_(
            0, x_pos, x_hb[x_sel].to(x_monsters.dtype)
        )

        # Entities / Event Options
        x_event_options = self._project_type(x_game_state.event_options, self._pre_event_option)

        # Entities / Character
        x_character = self._project_type(x_game_state.character, self._pre_character)
        # Health+Block
        x_sel = x_entity_kind_hb == EntityKind.CHARACTER
        x_pos = x_batch_hb[x_sel] * x_character.shape[1] + x_offset_hb[x_sel]
        x_character.view(-1, self._dim_embedding).index_add_(
            0, x_pos, x_hb[x_sel].to(x_character.dtype)
        )

        # Entities / Rooms
        x_rooms_x = self._project_type(x_rooms, self._pre_room)

        # Effects/Modifiers: ONE batched GEMM each, then every row is sum-pooled
        # into its owner's slot of the owner TYPE's block, keyed by (idx_type,
        # offset) — no window position exists. Deliberately before the shared
        # ReLU/LayerNorm so an entity's pooled rows share its normalization
        # Effects
        x_batch_eff, x_slot_eff = torch.nonzero(
            x_game_state.effects.x_entity_kind >= 0, as_tuple=True
        )
        x_effs = self._effects_mlp(
            x_game_state.effects.x[x_batch_eff, x_slot_eff]
        )  # (N_eff, ENCODING_DIM_EFFECT)

        # Modifiers
        x_batch_mod, x_slot_mod = torch.nonzero(
            x_game_state.modifiers.x_entity_kind >= 0, as_tuple=True
        )
        x_mods = self._modifiers_mlp(
            x_game_state.modifiers.x[x_batch_mod, x_slot_mod]
        )  # (N_mod, ENCODING_DIM_MODIFIER)

        # Pool Effects and Modifiers and add them to their respective Entities
        x_entity_kind_eff = x_game_state.effects.x_entity_kind[x_batch_eff, x_slot_eff]
        x_offset_eff = x_game_state.effects.x_offset[x_batch_eff, x_slot_eff]
        x_entity_kind_mod = x_game_state.modifiers.x_entity_kind[x_batch_mod, x_slot_mod]
        x_offset_mod = x_game_state.modifiers.x_offset[x_batch_mod, x_slot_mod]

        # The halt owns no entity, so it becomes a global-context input instead
        x_sel = x_entity_kind_eff == OWNER_PENDING
        x_pending = torch.zeros(
            batch_size, self._dim_embedding, dtype=torch.float32, device=device
        )
        x_nb = torch.bincount(x_batch_eff[x_sel], minlength=batch_size)[x_batch_eff[x_sel]]
        x_pending.index_add_(
            0,
            x_batch_eff[x_sel],
            x_effs[x_sel].to(torch.float32) / x_nb.to(torch.float32).sqrt().unsqueeze(1),
        )

        # Cards
        x_sel = x_entity_kind_eff == EntityKind.CARD
        x_pos = x_batch_eff[x_sel] * x_cards.shape[1] + x_offset_eff[x_sel]
        x_n = torch.bincount(x_pos, minlength=x_cards.shape[0] * x_cards.shape[1])[x_pos]
        x_scale = x_n.to(x_effs.dtype).sqrt().unsqueeze(1)
        x_cards.view(-1, self._dim_embedding).index_add_(
            0, x_pos, (x_effs[x_sel] / x_scale).to(x_cards.dtype)
        )

        # Relics
        x_sel = x_entity_kind_eff == EntityKind.RELIC
        x_pos = x_batch_eff[x_sel] * x_relics.shape[1] + x_offset_eff[x_sel]
        x_n = torch.bincount(x_pos, minlength=x_relics.shape[0] * x_relics.shape[1])[x_pos]
        x_scale = x_n.to(x_effs.dtype).sqrt().unsqueeze(1)
        x_relics.view(-1, self._dim_embedding).index_add_(
            0, x_pos, (x_effs[x_sel] / x_scale).to(x_relics.dtype)
        )

        # Potions
        x_sel = x_entity_kind_eff == EntityKind.POTION
        x_pos = x_batch_eff[x_sel] * x_potions.shape[1] + x_offset_eff[x_sel]
        x_n = torch.bincount(x_pos, minlength=x_potions.shape[0] * x_potions.shape[1])[x_pos]
        x_scale = x_n.to(x_effs.dtype).sqrt().unsqueeze(1)
        x_potions.view(-1, self._dim_embedding).index_add_(
            0, x_pos, (x_effs[x_sel] / x_scale).to(x_potions.dtype)
        )

        # Monsters / Effects
        x_sel = x_entity_kind_eff == EntityKind.MONSTER
        x_pos = x_batch_eff[x_sel] * x_monsters.shape[1] + x_offset_eff[x_sel]
        x_n = torch.bincount(x_pos, minlength=x_monsters.shape[0] * x_monsters.shape[1])[x_pos]
        x_scale = x_n.to(x_effs.dtype).sqrt().unsqueeze(1)
        x_monsters.view(-1, self._dim_embedding).index_add_(
            0, x_pos, (x_effs[x_sel] / x_scale).to(x_monsters.dtype)
        )

        # Monsters / Modifiers
        x_sel = x_entity_kind_mod == EntityKind.MONSTER
        x_pos = x_batch_mod[x_sel] * x_monsters.shape[1] + x_offset_mod[x_sel]
        x_n = torch.bincount(x_pos, minlength=x_monsters.shape[0] * x_monsters.shape[1])[x_pos]
        x_scale = x_n.to(x_mods.dtype).sqrt().unsqueeze(1)
        x_monsters.view(-1, self._dim_embedding).index_add_(
            0, x_pos, (x_mods[x_sel] / x_scale).to(x_monsters.dtype)
        )

        # Event options
        x_sel = x_entity_kind_eff == EntityKind.EVENT_OPTION
        x_pos = x_batch_eff[x_sel] * x_event_options.shape[1] + x_offset_eff[x_sel]
        x_n = torch.bincount(x_pos, minlength=x_event_options.shape[0] * x_event_options.shape[1])[
            x_pos
        ]
        x_scale = x_n.to(x_effs.dtype).sqrt().unsqueeze(1)
        x_event_options.view(-1, self._dim_embedding).index_add_(
            0, x_pos, (x_effs[x_sel] / x_scale).to(x_event_options.dtype)
        )

        # Character / Effects
        x_sel = x_entity_kind_eff == EntityKind.CHARACTER
        x_pos = x_batch_eff[x_sel] * x_character.shape[1] + x_offset_eff[x_sel]
        x_n = torch.bincount(x_pos, minlength=x_character.shape[0] * x_character.shape[1])[x_pos]
        x_scale = x_n.to(x_effs.dtype).sqrt().unsqueeze(1)
        x_character.view(-1, self._dim_embedding).index_add_(
            0, x_pos, (x_effs[x_sel] / x_scale).to(x_character.dtype)
        )

        # Character / Modifiers
        x_sel = x_entity_kind_mod == EntityKind.CHARACTER
        x_pos = x_batch_mod[x_sel] * x_character.shape[1] + x_offset_mod[x_sel]
        x_n = torch.bincount(x_pos, minlength=x_character.shape[0] * x_character.shape[1])[x_pos]
        x_scale = x_n.to(x_mods.dtype).sqrt().unsqueeze(1)
        x_character.view(-1, self._dim_embedding).index_add_(
            0, x_pos, (x_mods[x_sel] / x_scale).to(x_character.dtype)
        )

        x_blocks = [
            x_cards,
            x_relics,
            x_potions,
            x_monsters,
            x_event_options,
            x_character,
            x_rooms_x,
        ]
        # EntityKind order: the list above IS the window's entity order, the order
        # every consumer cats the kind rows in. Effects before modifiers — float
        # sums are order-sensitive
        x_grid = torch.cat(x_blocks, dim=1)
        num_tokens = x_grid.shape[1]
        x_grid = x_grid.reshape(batch_size * num_tokens, self._dim_embedding)
        x_idxs = torch.cat(
            [
                x_game_state.cards.x_slice_kind,
                x_game_state.relics.x_slice_kind,
                x_game_state.potions.x_slice_kind,
                x_game_state.monsters.x_slice_kind,
                x_game_state.event_options.x_slice_kind,
                x_game_state.character.x_slice_kind,
                x_rooms.x_slice_kind,
            ],
            dim=1,
        )
        x_mask = (x_idxs >= 0).flatten()

        # Shared second layer + norm, on the valid rows only (never empty: the
        # character mask is all-True). Running it on pads would turn their exact
        # zeros into a bias, and downstream masks multiply (NaN*0=NaN)
        x_row = torch.nonzero(x_mask, as_tuple=True)[0]
        x_hidden = self._layer_norm(self._post_shared(F.relu(x_grid[x_row])))

        # Zeros buffer in the INPUT dtype: the fp32 LayerNorm output lands in an
        # fp32 buffer under bf16 autocast. Explicit upcast: CUDA autocast runs
        # LayerNorm in fp32 but CPU autocast emits bf16, and index_put requires
        # matching dtypes
        x_entities = torch.zeros(
            batch_size * num_tokens,
            self._dim_embedding,
            dtype=x_game_state.cards.x.dtype,
            device=device,
        )
        x_entities[x_row] = x_hidden.to(x_entities.dtype)

        # The window: the six blocks then the rooms, the kind row cat'd beside the
        # features (long: the model indexes with it)
        x_window = X_Entity(
            x_entities.reshape(batch_size, num_tokens, self._dim_embedding), x_idxs.long()
        )
        return x_window, x_pending

    def _project_type(self, x_entity: X_Entity, block: nn.Linear) -> torch.Tensor:
        """One entity type's valid rows through its private block, put back into
        the type's own (B, W, D) block. Pad rows stay exact zeros."""
        batch_size, num_entities, dim = x_entity.x.shape

        # Calculate non-padding tokens
        x_row = torch.nonzero(torch.flatten(x_entity.mask), as_tuple=True)[0]
        x_valid = x_entity.x.reshape(batch_size * num_entities, dim)[x_row]

        # Forward pass
        x_first = block(x_valid)

        x_block = torch.zeros(
            batch_size * num_entities,
            self._dim_embedding,
            dtype=x_first.dtype,
            device=x_first.device,
        )
        x_block[x_row] = x_first
        return x_block.reshape(batch_size, num_entities, self._dim_embedding)
