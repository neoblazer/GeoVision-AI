"""Model-free resolver for explicitly supplied external detector checkpoints."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from geovision.domain.benchmark import DetectorProfile
from geovision.domain.m1_artifacts import CheckpointFingerprint

HASH_CHUNK_SIZE = 1024 * 1024


class CheckpointFailureCode(StrEnum):
    MISSING = "missing"
    NOT_REGULAR_FILE = "not_regular_file"
    SYMLINK_FORBIDDEN = "symlink_forbidden"
    REPOSITORY_LOCAL = "repository_local"
    FILENAME_MISMATCH = "filename_mismatch"
    SIZE_MISMATCH = "size_mismatch"
    CHANGED_DURING_HASH = "changed_during_hash"
    HASH_MISMATCH = "hash_mismatch"
    INVALID_EXPECTATION = "invalid_expectation"


class CheckpointResolutionError(RuntimeError):
    """Resolver failure with a path-free category and safe explanation."""

    def __init__(self, code: CheckpointFailureCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


@dataclass(frozen=True, slots=True)
class ResolvedCheckpoint:
    path: Path
    fingerprint: CheckpointFingerprint

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValueError("resolved checkpoint path must be absolute")


def _single_path_is_symlink_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _is_symlink_or_reparse(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    return any(
        _single_path_is_symlink_or_reparse(component)
        for component in reversed((absolute, *absolute.parents))
    )


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_dev,
        metadata.st_ino,
    )


def _stream_sha256(path: Path, *, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        while chunk := checkpoint.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_canonical_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_within_repository(checkpoint: Path, repository_root: Path) -> bool:
    normalized_checkpoint = os.path.normcase(os.path.abspath(checkpoint))
    normalized_root = os.path.normcase(os.path.abspath(repository_root))
    try:
        return os.path.commonpath((normalized_checkpoint, normalized_root)) == normalized_root
    except ValueError:
        return False


def _resolve_strict(path: Path) -> Path:
    return path.resolve(strict=True)


def resolve_checkpoint(
    checkpoint_path: str | Path,
    repository_root: str | Path,
    detector_profile: DetectorProfile,
) -> ResolvedCheckpoint:
    """Verify an external checkpoint without importing or loading a model runtime."""

    if not isinstance(detector_profile, DetectorProfile):
        raise TypeError("detector_profile must be a DetectorProfile")
    expected_hash = detector_profile.weights_sha256
    if not _is_canonical_sha256(expected_hash):
        raise CheckpointResolutionError(
            CheckpointFailureCode.INVALID_EXPECTATION,
            "detector profile must contain a canonical lowercase SHA-256",
        )
    if type(detector_profile.weights_size_bytes) is not int:
        raise CheckpointResolutionError(
            CheckpointFailureCode.INVALID_EXPECTATION,
            "detector profile must contain an exact byte size",
        )

    candidate = Path(checkpoint_path)
    if _is_symlink_or_reparse(candidate):
        raise CheckpointResolutionError(
            CheckpointFailureCode.SYMLINK_FORBIDDEN,
            "checkpoint indirection is forbidden",
        )
    if not candidate.exists():
        raise CheckpointResolutionError(
            CheckpointFailureCode.MISSING,
            "checkpoint file is missing",
        )
    if not candidate.is_file():
        raise CheckpointResolutionError(
            CheckpointFailureCode.NOT_REGULAR_FILE,
            "checkpoint must be a regular file",
        )

    try:
        resolved = _resolve_strict(candidate)
        resolved_repository = _resolve_strict(Path(repository_root))
    except FileNotFoundError as exc:
        raise CheckpointResolutionError(
            CheckpointFailureCode.MISSING,
            "checkpoint or repository root is missing",
        ) from exc
    except OSError as exc:
        raise CheckpointResolutionError(
            CheckpointFailureCode.NOT_REGULAR_FILE,
            "checkpoint or repository root could not be resolved",
        ) from exc

    if _is_symlink_or_reparse(resolved):
        raise CheckpointResolutionError(
            CheckpointFailureCode.SYMLINK_FORBIDDEN,
            "checkpoint indirection is forbidden",
        )

    if resolved.name != detector_profile.model:
        raise CheckpointResolutionError(
            CheckpointFailureCode.FILENAME_MISMATCH,
            "checkpoint basename does not match the detector profile",
        )
    if _is_within_repository(resolved, resolved_repository):
        raise CheckpointResolutionError(
            CheckpointFailureCode.REPOSITORY_LOCAL,
            "checkpoint must remain outside the repository",
        )

    try:
        before = resolved.stat()
    except OSError as exc:
        raise CheckpointResolutionError(
            CheckpointFailureCode.NOT_REGULAR_FILE,
            "checkpoint metadata could not be read",
        ) from exc
    if before.st_size != detector_profile.weights_size_bytes:
        raise CheckpointResolutionError(
            CheckpointFailureCode.SIZE_MISMATCH,
            "checkpoint byte size does not match the detector profile",
        )

    try:
        digest = _stream_sha256(resolved, chunk_size=HASH_CHUNK_SIZE)
        after = resolved.stat()
    except FileNotFoundError as exc:
        raise CheckpointResolutionError(
            CheckpointFailureCode.CHANGED_DURING_HASH,
            "checkpoint changed during fingerprinting",
        ) from exc
    except OSError as exc:
        raise CheckpointResolutionError(
            CheckpointFailureCode.NOT_REGULAR_FILE,
            "checkpoint could not be read",
        ) from exc

    if _stat_identity(before) != _stat_identity(after):
        raise CheckpointResolutionError(
            CheckpointFailureCode.CHANGED_DURING_HASH,
            "checkpoint changed during fingerprinting",
        )
    if digest != expected_hash:
        raise CheckpointResolutionError(
            CheckpointFailureCode.HASH_MISMATCH,
            "checkpoint SHA-256 does not match the detector profile",
        )

    fingerprint = CheckpointFingerprint(
        detector_profile_id=detector_profile.profile_id,
        expected_filename=detector_profile.model,
        sha256=digest,
        byte_size=after.st_size,
    )
    return ResolvedCheckpoint(path=resolved, fingerprint=fingerprint)


__all__ = [
    "CheckpointFailureCode",
    "CheckpointResolutionError",
    "ResolvedCheckpoint",
    "resolve_checkpoint",
]
