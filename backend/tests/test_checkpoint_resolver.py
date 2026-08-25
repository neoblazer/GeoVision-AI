from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from pydantic import ValidationError

from geovision.adapters.perception import checkpoints
from geovision.adapters.perception.checkpoints import (
    CheckpointFailureCode,
    CheckpointResolutionError,
    resolve_checkpoint,
)
from geovision.domain.benchmark import DetectorProfile
from geovision.domain.m1_artifacts import CheckpointFingerprint


class CheckpointResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.repository_root = self.temporary_root / "repository"
        self.external_root = self.temporary_root / "external"
        self.repository_root.mkdir()
        self.external_root.mkdir()
        self.checkpoint_bytes = b"small deterministic fake checkpoint"
        self.checkpoint_path = self.external_root / "yolo11n.pt"
        self.checkpoint_path.write_bytes(self.checkpoint_bytes)

    def profile(
        self,
        *,
        sha256: str | None = None,
        size: int | None = None,
    ) -> DetectorProfile:
        return DetectorProfile(
            profile_id="yolo11n",
            backend="ultralytics",
            model="yolo11n.pt",
            weights_sha256=sha256 or hashlib.sha256(self.checkpoint_bytes).hexdigest(),
            weights_size_bytes=len(self.checkpoint_bytes) if size is None else size,
        )

    def test_valid_external_checkpoint_returns_path_free_fingerprint(self) -> None:
        before = self.checkpoint_path.read_bytes()

        resolved = resolve_checkpoint(
            self.checkpoint_path,
            self.repository_root,
            self.profile(),
        )

        self.assertTrue(resolved.path.is_absolute())
        self.assertEqual(resolved.path, self.checkpoint_path.resolve())
        self.assertEqual(resolved.fingerprint.detector_profile_id, "yolo11n")
        self.assertEqual(resolved.fingerprint.expected_filename, "yolo11n.pt")
        self.assertEqual(
            resolved.fingerprint.sha256,
            hashlib.sha256(self.checkpoint_bytes).hexdigest(),
        )
        self.assertEqual(resolved.fingerprint.byte_size, len(self.checkpoint_bytes))
        self.assertNotIn(str(self.temporary_root), resolved.fingerprint.model_dump_json())
        self.assertNotIn("path", resolved.fingerprint.model_dump())
        self.assertEqual(self.checkpoint_path.read_bytes(), before)

    def test_missing_file_is_rejected_with_safe_error(self) -> None:
        missing = self.external_root / "private" / "yolo11n.pt"

        with self.assertRaises(CheckpointResolutionError) as raised:
            resolve_checkpoint(missing, self.repository_root, self.profile())

        self.assertEqual(raised.exception.code, CheckpointFailureCode.MISSING)
        self.assertNotIn(str(self.temporary_root), str(raised.exception))

    def test_directory_is_rejected(self) -> None:
        directory = self.external_root / "directory" / "yolo11n.pt"
        directory.mkdir(parents=True)

        with self.assertRaises(CheckpointResolutionError) as raised:
            resolve_checkpoint(directory, self.repository_root, self.profile())

        self.assertEqual(raised.exception.code, CheckpointFailureCode.NOT_REGULAR_FILE)

    def test_wrong_basename_is_rejected(self) -> None:
        wrong_name = self.external_root / "wrong.pt"
        wrong_name.write_bytes(self.checkpoint_bytes)

        with self.assertRaises(CheckpointResolutionError) as raised:
            resolve_checkpoint(wrong_name, self.repository_root, self.profile())

        self.assertEqual(raised.exception.code, CheckpointFailureCode.FILENAME_MISMATCH)

    def test_repository_local_checkpoint_is_rejected(self) -> None:
        local = self.repository_root / "yolo11n.pt"
        local.write_bytes(self.checkpoint_bytes)

        with self.assertRaises(CheckpointResolutionError) as raised:
            resolve_checkpoint(local, self.repository_root, self.profile())

        self.assertEqual(raised.exception.code, CheckpointFailureCode.REPOSITORY_LOCAL)

    def test_repository_root_equality_is_contained(self) -> None:
        self.assertTrue(
            checkpoints._is_within_repository(self.repository_root, self.repository_root)
        )

    def test_repository_prefix_collision_is_external(self) -> None:
        prefix_collision_root = self.temporary_root / "repository-other"
        prefix_collision_root.mkdir()
        candidate = prefix_collision_root / "yolo11n.pt"
        candidate.write_bytes(self.checkpoint_bytes)

        resolved = resolve_checkpoint(candidate, self.repository_root, self.profile())

        self.assertEqual(resolved.path, candidate.resolve())

    def test_dotdot_resolving_outside_repository_is_allowed(self) -> None:
        outside_root = self.temporary_root / "outside"
        outside_root.mkdir()
        candidate = outside_root / "yolo11n.pt"
        candidate.write_bytes(self.checkpoint_bytes)
        dotdot_path = self.repository_root / ".." / "outside" / "yolo11n.pt"

        resolved = resolve_checkpoint(dotdot_path, self.repository_root, self.profile())

        self.assertEqual(resolved.path, candidate.resolve())

    def test_dotdot_resolving_inside_repository_is_rejected(self) -> None:
        local = self.repository_root / "yolo11n.pt"
        local.write_bytes(self.checkpoint_bytes)
        dotdot_path = self.external_root / ".." / "repository" / "yolo11n.pt"

        with self.assertRaises(CheckpointResolutionError) as raised:
            resolve_checkpoint(dotdot_path, self.repository_root, self.profile())

        self.assertEqual(raised.exception.code, CheckpointFailureCode.REPOSITORY_LOCAL)

    def test_different_drive_commonpath_error_is_treated_as_external(self) -> None:
        with patch.object(checkpoints.os.path, "commonpath", side_effect=ValueError):
            contained = checkpoints._is_within_repository(
                Path("D:/models/yolo11n.pt"),
                Path("C:/repository"),
            )

        self.assertFalse(contained)

    def test_case_normalization_matches_windows_repository_identity(self) -> None:
        candidate = self.repository_root / "models" / "yolo11n.pt"
        differently_cased_root = Path(str(self.repository_root).upper())

        self.assertTrue(
            checkpoints._is_within_repository(candidate, differently_cased_root)
        )

    def test_hash_mismatch_is_rejected(self) -> None:
        with self.assertRaises(CheckpointResolutionError) as raised:
            resolve_checkpoint(
                self.checkpoint_path,
                self.repository_root,
                self.profile(sha256="0" * 64),
            )

        self.assertEqual(raised.exception.code, CheckpointFailureCode.HASH_MISMATCH)

    def test_size_mismatch_is_rejected_before_hashing(self) -> None:
        with patch.object(
            checkpoints,
            "_stream_sha256",
            side_effect=AssertionError("hashing must not begin after a size mismatch"),
        ):
            with self.assertRaises(CheckpointResolutionError) as raised:
                resolve_checkpoint(
                    self.checkpoint_path,
                    self.repository_root,
                    self.profile(size=len(self.checkpoint_bytes) + 1),
                )

        self.assertEqual(raised.exception.code, CheckpointFailureCode.SIZE_MISMATCH)

    def test_uppercase_expected_hash_is_rejected_at_config_and_resolver_boundaries(self) -> None:
        profile_data = self.profile().model_dump(mode="json")
        profile_data["weights_sha256"] = profile_data["weights_sha256"].upper()
        with self.assertRaises(ValidationError):
            DetectorProfile.model_validate(profile_data)

        bypassed_profile = self.profile().model_copy(
            update={"weights_sha256": hashlib.sha256(self.checkpoint_bytes).hexdigest().upper()}
        )
        with self.assertRaises(CheckpointResolutionError) as raised:
            resolve_checkpoint(
                self.checkpoint_path,
                self.repository_root,
                bypassed_profile,
            )
        self.assertEqual(raised.exception.code, CheckpointFailureCode.INVALID_EXPECTATION)

    def test_symlink_is_rejected_when_supported(self) -> None:
        link = self.external_root / "link" / "yolo11n.pt"
        link.parent.mkdir()
        try:
            link.symlink_to(self.checkpoint_path)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")

        with self.assertRaises(CheckpointResolutionError) as raised:
            resolve_checkpoint(link, self.repository_root, self.profile())

        self.assertEqual(raised.exception.code, CheckpointFailureCode.SYMLINK_FORBIDDEN)

    def test_direct_checkpoint_indirection_has_deterministic_coverage(self) -> None:
        candidate = self.checkpoint_path.absolute()

        with patch.object(
            checkpoints,
            "_single_path_is_symlink_or_reparse",
            side_effect=lambda path: path == candidate,
        ):
            with self.assertRaises(CheckpointResolutionError) as raised:
                resolve_checkpoint(self.checkpoint_path, self.repository_root, self.profile())

        self.assertEqual(raised.exception.code, CheckpointFailureCode.SYMLINK_FORBIDDEN)

    def test_reparse_parent_has_deterministic_coverage(self) -> None:
        reparse_component = self.external_root.absolute()

        with patch.object(
            checkpoints,
            "_single_path_is_symlink_or_reparse",
            side_effect=lambda path: path == reparse_component,
        ):
            with self.assertRaises(CheckpointResolutionError) as raised:
                resolve_checkpoint(self.checkpoint_path, self.repository_root, self.profile())

        self.assertEqual(raised.exception.code, CheckpointFailureCode.SYMLINK_FORBIDDEN)

    def test_external_path_resolving_into_repository_is_rejected_by_containment(self) -> None:
        logical_path = self.external_root / "logical-inward" / "yolo11n.pt"
        logical_path.parent.mkdir()
        logical_path.write_bytes(self.checkpoint_bytes)
        repository_target = self.repository_root / "yolo11n.pt"
        repository_target.write_bytes(self.checkpoint_bytes)
        resolved_target = repository_target.resolve()
        resolved_repository = self.repository_root.resolve()

        def controlled_resolve(path: Path) -> Path:
            if path == logical_path:
                return resolved_target
            if path == self.repository_root:
                return resolved_repository
            raise AssertionError(f"unexpected path resolution: {path.name}")

        with (
            patch.object(
                checkpoints,
                "_resolve_strict",
                side_effect=controlled_resolve,
            ),
            patch.object(
                checkpoints,
                "_is_symlink_or_reparse",
                wraps=checkpoints._is_symlink_or_reparse,
            ) as indirection,
            patch.object(
                checkpoints,
                "_is_within_repository",
                wraps=checkpoints._is_within_repository,
            ) as containment,
        ):
            with self.assertRaises(CheckpointResolutionError) as raised:
                resolve_checkpoint(logical_path, self.repository_root, self.profile())

        self.assertEqual(raised.exception.code, CheckpointFailureCode.REPOSITORY_LOCAL)
        self.assertEqual(
            indirection.call_args_list,
            [call(logical_path), call(resolved_target)],
        )
        containment.assert_called_once_with(resolved_target, resolved_repository)

    def test_repository_local_indirection_cannot_resolve_outward(self) -> None:
        logical_path = self.repository_root / "yolo11n.pt"
        logical_path.write_bytes(self.checkpoint_bytes)
        outward_target = self.checkpoint_path.resolve()
        resolved_repository = self.repository_root.resolve()

        def detect_original_indirection(path: Path) -> bool:
            return path == logical_path

        def controlled_resolve(path: Path) -> Path:
            if path == logical_path:
                return outward_target
            if path == self.repository_root:
                return resolved_repository
            raise AssertionError(f"unexpected path resolution: {path.name}")

        self.assertFalse(detect_original_indirection(outward_target))
        with (
            patch.object(
                checkpoints,
                "_is_symlink_or_reparse",
                side_effect=detect_original_indirection,
            ) as indirection,
            patch.object(
                checkpoints,
                "_resolve_strict",
                side_effect=controlled_resolve,
            ) as resolution,
        ):
            with self.assertRaises(CheckpointResolutionError) as raised:
                resolve_checkpoint(logical_path, self.repository_root, self.profile())

        self.assertEqual(raised.exception.code, CheckpointFailureCode.SYMLINK_FORBIDDEN)
        self.assertEqual(indirection.call_args_list, [call(logical_path)])
        resolution.assert_not_called()

    def test_post_resolution_indirection_is_rejected(self) -> None:
        logical_path = self.external_root / "logical-external" / "yolo11n.pt"
        logical_path.parent.mkdir()
        logical_path.write_bytes(self.checkpoint_bytes)
        resolved_target = self.external_root / "resolved-external" / "yolo11n.pt"
        resolved_target.parent.mkdir()
        resolved_target.write_bytes(self.checkpoint_bytes)
        resolved_target = resolved_target.resolve()
        resolved_repository = self.repository_root.resolve()

        def detect_resolved_indirection(path: Path) -> bool:
            return path == resolved_target

        def controlled_resolve(path: Path) -> Path:
            if path == logical_path:
                return resolved_target
            if path == self.repository_root:
                return resolved_repository
            raise AssertionError(f"unexpected path resolution: {path.name}")

        self.assertFalse(detect_resolved_indirection(logical_path))
        self.assertTrue(detect_resolved_indirection(resolved_target))
        with (
            patch.object(
                checkpoints,
                "_is_symlink_or_reparse",
                side_effect=detect_resolved_indirection,
            ) as indirection,
            patch.object(
                checkpoints,
                "_resolve_strict",
                side_effect=controlled_resolve,
            ) as resolution,
        ):
            with self.assertRaises(CheckpointResolutionError) as raised:
                resolve_checkpoint(logical_path, self.repository_root, self.profile())

        self.assertEqual(raised.exception.code, CheckpointFailureCode.SYMLINK_FORBIDDEN)
        self.assertEqual(
            indirection.call_args_list,
            [call(logical_path), call(resolved_target)],
        )
        self.assertEqual(
            resolution.call_args_list,
            [call(logical_path), call(self.repository_root)],
        )

    def test_change_during_hash_is_detected(self) -> None:
        original_stream_sha256 = checkpoints._stream_sha256

        def changing_hash(path: Path, *, chunk_size: int) -> str:
            digest = original_stream_sha256(path, chunk_size=chunk_size)
            with path.open("ab") as checkpoint:
                checkpoint.write(b"changed")
            return digest

        with patch.object(checkpoints, "_stream_sha256", changing_hash):
            with self.assertRaises(CheckpointResolutionError) as raised:
                resolve_checkpoint(
                    self.checkpoint_path,
                    self.repository_root,
                    self.profile(),
                )

        self.assertEqual(raised.exception.code, CheckpointFailureCode.CHANGED_DURING_HASH)

    def test_stable_stat_identity_passes(self) -> None:
        before = self.checkpoint_path.stat()

        resolve_checkpoint(self.checkpoint_path, self.repository_root, self.profile())

        after = self.checkpoint_path.stat()
        self.assertEqual(
            (before.st_size, before.st_mtime_ns, before.st_ctime_ns),
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        )

    def test_module_has_no_model_runtime_imports(self) -> None:
        self.assertNotIn("torch", checkpoints.__dict__)
        self.assertNotIn("ultralytics", checkpoints.__dict__)

    def test_chained_operational_path_is_not_in_public_error_fields(self) -> None:
        private_message = f"checkpoint read failed for {self.checkpoint_path}"

        with patch.object(
            checkpoints,
            "_stream_sha256",
            side_effect=OSError(private_message),
        ):
            with self.assertRaises(CheckpointResolutionError) as raised:
                resolve_checkpoint(
                    self.checkpoint_path,
                    self.repository_root,
                    self.profile(),
                )

        self.assertIsNotNone(raised.exception.__cause__)
        self.assertIn(str(self.checkpoint_path), str(raised.exception.__cause__))
        serialized_public_error = json.dumps(
            {
                "code": raised.exception.code.value,
                "detail": raised.exception.detail,
            }
        )
        self.assertNotIn(str(self.temporary_root), serialized_public_error)

    def test_checkpoint_fingerprint_rejects_paths_and_coercive_fields(self) -> None:
        valid = {
            "detector_profile_id": "yolo11n",
            "expected_filename": "yolo11n.pt",
            "sha256": "a" * 64,
            "byte_size": 1,
        }
        for field_name, invalid_value in (
            ("expected_filename", "models/yolo11n.pt"),
            ("sha256", "A" * 64),
            ("byte_size", True),
            ("byte_size", "1"),
            ("byte_size", -1),
        ):
            with self.subTest(field=field_name, value=invalid_value):
                data = dict(valid)
                data[field_name] = invalid_value
                with self.assertRaises(ValidationError):
                    CheckpointFingerprint.model_validate(data)


if __name__ == "__main__":
    unittest.main()
