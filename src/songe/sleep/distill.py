"""Public compatibility imports retained for pre-backend scripts/integrations.

New code imports songe.backends; existing Backend/train/objective imports remain
supported here. The implementation lives in songe.backends.mlx.
"""

from ..backends import Interrupted
from ..backends.mlx import Backend, train
from ..backends.objective import divergence

__all__ = ["Backend", "Interrupted", "divergence", "train"]
