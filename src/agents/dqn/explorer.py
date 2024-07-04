def linear_decay(
    epoch_end: int,
    epoch_elbow: int,
    value_start: float,
    value_end: float,
) -> list[float]:
    """
    Linear decay from value_start to value_end over epochs until epoch_elbow, then remains
    constant until epoch_end.
    """
    # TODO: make  it readable jeez
    return [
        (
            value_start - (value_start - value_end) * min(epoch, epoch_elbow) / epoch_elbow
            if epoch < epoch_elbow
            else value_end
        )
        for epoch in range(epoch_end)
    ]
