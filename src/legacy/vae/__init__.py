"""Legacy VAE package.

This repo vendors parts of the `pythae`-style API under `src.legacy.vae`.

Important: keep this module *import-light*. Some legacy model configs import
auxiliary scripts that import `src.functions`, which (transitively) import
`src.legacy.vae` symbols. Eagerly importing `AutoModel` here can create a
circular import during embedding generation, especially in the Python 3.9
subprocess used by Build06.

We therefore expose commonly-used symbols via lazy module attribute access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["AutoModel", "ModelOutput", "BaseEncoder", "BaseDecoder"]

if TYPE_CHECKING:
    from .models.auto_model import AutoModel as AutoModel
    from .models.base.base_utils import ModelOutput as ModelOutput
    from .models.nn import BaseDecoder as BaseDecoder
    from .models.nn import BaseEncoder as BaseEncoder


def __getattr__(name: str) -> Any:
    if name == "AutoModel":
        from .models.auto_model import AutoModel as _AutoModel

        return _AutoModel
    if name == "ModelOutput":
        from .models.base.base_utils import ModelOutput as _ModelOutput

        return _ModelOutput
    if name == "BaseEncoder":
        from .models.nn import BaseEncoder as _BaseEncoder

        return _BaseEncoder
    if name == "BaseDecoder":
        from .models.nn import BaseDecoder as _BaseDecoder

        return _BaseDecoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
