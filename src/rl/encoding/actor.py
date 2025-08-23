import torch

from src.game.view.actor import ViewModifierType


_VIEW_MODIFIER_TYPE_STACKS_MAX = {
    ViewModifierType.STRENGTH: 20,
    ViewModifierType.WEAK: 5,
    ViewModifierType.MODE_SHIFT: 60,
    ViewModifierType.RITUAL: 20,
    ViewModifierType.SHARP_HIDE: 3,
    ViewModifierType.SPORE_CLOUD: 2,
    ViewModifierType.VULNERABLE: 4,
    ViewModifierType.ACCURACY: 10,  # TODO: fix
    ViewModifierType.NEXT_TURN_BLOCK: 10,  # TODO: fix
    ViewModifierType.NEXT_TURN_ENERGY: 10,  # TODO: fix
    ViewModifierType.BLUR: 10,  # TODO: fix
    ViewModifierType.DEXTERITY: 10,  # TODO: fix
}


def encode_view_actor_modifiers(
    view_actor_modifiers: dict[ViewModifierType, int | None], device: torch.device
) -> torch.Tensor:
    encoding = []
    for view_modifier_type in ViewModifierType:
        # Get current stacks, fallback to 0
        stacks_current = view_actor_modifiers.get(view_modifier_type, 0)

        # For unstackable modifiers, just signal their presence with a 1
        if stacks_current is None:
            stacks_current = 1

        # Normalize and append to list
        stacks_current /= _VIEW_MODIFIER_TYPE_STACKS_MAX[view_modifier_type]
        encoding.append(stacks_current)

    return torch.tensor(encoding, dtype=torch.float32, device=device)
