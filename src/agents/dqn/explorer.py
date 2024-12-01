def linear_decay(
    episode_end: int,
    episode_elbow: int,
    value_start: float,
    value_end: float,
) -> list[float]:
    # TODO: make it readable jeez
    return [
        (
            value_start - (value_start - value_end) * min(episode, episode_elbow) / episode_elbow
            if episode < episode_elbow
            else value_end
        )
        for episode in range(episode_end)
    ]
