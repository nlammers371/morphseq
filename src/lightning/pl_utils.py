import math

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

def cosine_ramp_weight(step_curr: int, 
                       n_warmup: int,
                       n_rampup: int,
                       w_min: float,
                       w_max: float,
                       ) -> float:
    """
    Returns a weight that stays at w_min for n_warmup,
    then smoothly rises from w_min→w_max over the next n_rampup
    following a half‑cosine, and remains at w_max thereafter.
    """
    if step_curr < n_warmup:
        return w_min
    # how far into the ramp we are (0→1)
    t = (step_curr - n_warmup) / float(max(1, n_rampup))
    if t >= 1.0:
        return w_max
    # half‑cosine:  (1 – cos(π t)) / 2
    cos_val = (1 - math.cos(math.pi * t)) / 2
    return w_min + (w_max - w_min) * cos_val