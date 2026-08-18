"""Visual World Kernel V1.

The package is intentionally isolated from the legacy simulation kernel while its
event-sourced contracts are validated.  The compatibility boundary can later be
moved inward without creating a second source of truth in an active session.
"""

from .kernel import CommandRejected, VisualWorldKernel
from .models import (
    CommandEnvelope,
    CommandKind,
    CommandReceipt,
    DomainEvent,
    WorldDefinition,
    WorldRuntime,
)
from .projection import dev_projection, player_projection
from .service import VisualWorldService
from .storage import ConcurrencyConflict, VisualWorldRepository, VisualWorldSessionNotFound

__all__ = [
    "CommandEnvelope",
    "CommandKind",
    "CommandReceipt",
    "CommandRejected",
    "ConcurrencyConflict",
    "DomainEvent",
    "VisualWorldKernel",
    "VisualWorldRepository",
    "VisualWorldService",
    "VisualWorldSessionNotFound",
    "WorldDefinition",
    "WorldRuntime",
    "dev_projection",
    "player_projection",
]
