def ramp_weight(
    step_curr: int,
    *,
    n_warmup: int,
    n_rampup: int,
    w_min: float = 0.0,
    w_max: float = 1.0,
) -> float:
    """
    Piece-wise schedule            (steps)
        • warm-up   : 0 … n_warmup-1      → w_min
        • ramp-up   : n_warmup … n_warmup+n_rampup-1
                      linear   w_min → w_max
        • plateau   : ≥ n_warmup+n_rampup → w_max
    """
    if step_curr < n_warmup:                     # ─── warm-up
        return w_min

    ramp_step = step_curr - n_warmup
    if ramp_step < n_rampup:                     # ─── linear ramp
        progress = ramp_step / max(n_rampup - 1, 1)
        return w_min + progress * (w_max - w_min)

    return w_max