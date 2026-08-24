"""Enumerations used by persisted and transported domain models."""

from enum import StrEnum


class TrackState(StrEnum):
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


class EntityState(StrEnum):
    ACTIVE = "active"
    LOST = "lost"
    REACQUIRED = "reacquired"
    CLOSED = "closed"


class EventType(StrEnum):
    ZONE_ENTRY = "zone_entry"
    LOITERING = "loitering"
    DENSITY_SPIKE = "density_spike"


class DistanceSource(StrEnum):
    MONOCULAR = "monocular"
    GEOMETRIC = "geometric"
    FUSED = "fused"
    UNAVAILABLE = "unavailable"


class AssociationOutcome(StrEnum):
    MERGED = "merged"
    REJECTED = "rejected"
    DEFERRED = "deferred"

