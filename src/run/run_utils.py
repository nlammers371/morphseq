

def deep_merge(a: dict, b: dict) -> dict:
    """
    Return a new dict that is a deep‐merge of `a` and `b`.
    - For keys in both with dict values, recurse.
    - Otherwise b’s value wins.
    """
    merged = a.copy()
    for key, b_val in b.items():
        a_val = merged.get(key)
        if isinstance(a_val, dict) and isinstance(b_val, dict):
            merged[key] = deep_merge(a_val, b_val)
        else:
            merged[key] = b_val
    return merged


def prune_empty(d: dict) -> dict:
    """Return a copy of `d` with any None or '' leaves removed."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            # recurse—but only keep non-empty sub-dicts
            sub = prune_empty(v)
            if sub:
                out[k] = sub
        elif v is None or v == "":
            # skip this key entirely
            continue
        else:
            out[k] = v
    return out