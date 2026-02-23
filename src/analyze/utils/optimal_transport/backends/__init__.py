"""UOT solver backends."""

from analyze.utils.optimal_transport.backends.base import UOTBackend, BackendResult

__all__ = ["UOTBackend", "BackendResult", "POTBackend", "OTTBackend"]


def __getattr__(name: str):
    if name == "POTBackend":
        from analyze.utils.optimal_transport.backends.pot_backend import POTBackend as _POTBackend

        return _POTBackend
    if name == "OTTBackend":
        try:
            from analyze.utils.optimal_transport.backends.ott_backend import OTTBackend as _OTTBackend
        except ImportError:
            return None
        return _OTTBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
