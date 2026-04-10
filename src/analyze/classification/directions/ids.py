"""Vector ID conventions for classifier direction artifacts.

A vector_id uniquely identifies one direction vector:
  <feature_set>__<comparison_id>__bin_<time_bin_floor>

where time_bin_floor is the integer floor of the time bin (in hpf).

Example: "vae__pbx4_crispant__vs__wik_ab__bin_22"
"""

from __future__ import annotations


_SEP = "__"
_BIN_PREFIX = "bin_"


def make_vector_id(
    *,
    feature_set: str,
    comparison_id: str,
    time_bin: int,
) -> str:
    """Construct a canonical vector_id.

    Parameters
    ----------
    feature_set : str
        Name of the feature set (e.g. "vae").
    comparison_id : str
        The pairwise comparison identifier (e.g. "pbx4_crispant__vs__wik_ab").
    time_bin : int
        Integer floor of the time bin (e.g. 22 for the bin [22, 24) hpf).
    """
    return f"{feature_set}{_SEP}{comparison_id}{_SEP}{_BIN_PREFIX}{int(time_bin)}"


def parse_vector_id(vector_id: str) -> dict[str, object]:
    """Parse a vector_id back into its components.

    Returns
    -------
    dict with keys: feature_set (str), comparison_id (str), time_bin (int).

    Raises
    ------
    ValueError if the string does not match the expected format.
    """
    # Find the last occurrence of __bin_<digits>
    bin_marker = f"{_SEP}{_BIN_PREFIX}"
    idx = vector_id.rfind(bin_marker)
    if idx < 0:
        raise ValueError(
            f"vector_id {vector_id!r} does not contain a '__bin_<int>' suffix. "
            "Expected format: '<feature_set>__<comparison_id>__bin_<time_bin>'."
        )
    suffix = vector_id[idx + len(bin_marker):]
    if not suffix.lstrip("-").isdigit():
        raise ValueError(
            f"vector_id {vector_id!r}: bin suffix {suffix!r} is not an integer."
        )
    time_bin = int(suffix)
    prefix = vector_id[:idx]

    # Split prefix into feature_set and comparison_id at the first __
    first_sep = prefix.find(_SEP)
    if first_sep < 0:
        raise ValueError(
            f"vector_id {vector_id!r}: cannot separate feature_set from comparison_id "
            f"(no '{_SEP}' in prefix {prefix!r})."
        )
    feature_set = prefix[:first_sep]
    comparison_id = prefix[first_sep + len(_SEP):]
    return {
        "feature_set": feature_set,
        "comparison_id": comparison_id,
        "time_bin": time_bin,
    }
