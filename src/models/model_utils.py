from collections import OrderedDict
from typing import Tuple, Any


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

class ModelOutput(OrderedDict):
    """Base ModelOutput class fixing the output type from the models. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())