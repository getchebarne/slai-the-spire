from src.game.view.actor import ViewModifierType


_VIEW_MODIFIER_TYPE_STACKS_MAX = {
    ViewModifierType.STRENGTH: 20,
    ViewModifierType.WEAK: 5,
    ViewModifierType.MODE_SHIFT: 60,
    ViewModifierType.RITUAL: 20,
    ViewModifierType.SHARP_HIDE: 3,
    ViewModifierType.SPORE_CLOUD: 2,
    ViewModifierType.VULNERABLE: 4,
    ViewModifierType.ACCURACY: 16,
    ViewModifierType.NEXT_TURN_BLOCK: 20,
    ViewModifierType.NEXT_TURN_ENERGY: 5,
    ViewModifierType.BLUR: 5,
    ViewModifierType.DEXTERITY: 12,
    ViewModifierType.INFINITE_BLADES: 5,
    ViewModifierType.AFTER_IMAGE: 3,
    ViewModifierType.PHANTASMAL: 2,
    ViewModifierType.DOUBLE_DAMAGE: 1,
    ViewModifierType.THOUSAND_CUTS: 4,
    ViewModifierType.BURST: 4,
}


def encode_view_actor_modifiers(
    view_actor_modifiers: dict[ViewModifierType, int | None],
) -> list[float]:
    encoding = [0] * len(ViewModifierType)
    for idx, view_modifier_type in enumerate(ViewModifierType):
        # Get current stacks, fallback to 0
        stacks_current = view_actor_modifiers.get(view_modifier_type, 0)

        # For unstackable modifiers, just signal their presence with a 1
        if stacks_current is None:
            stacks_current = 1

        # Normalize and append to list TODO: revisit clamp
        stacks_max = _VIEW_MODIFIER_TYPE_STACKS_MAX[view_modifier_type]
        stacks_current = min(stacks_max, stacks_current)
        stacks_current /= _VIEW_MODIFIER_TYPE_STACKS_MAX[view_modifier_type]
        encoding[idx] = stacks_current

    return encoding
