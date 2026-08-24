"""Verify the pinned vision runtime without loading models or using the network."""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

EXPECTED_DISTRIBUTIONS = {
    "PyYAML": "6.0.3",
    "lap": "0.5.12",
    "numpy": "2.4.6",
    "opencv-python": "4.14.0.94",
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
    "ultralytics": "8.4.127",
}
EXPECTED_CUDA_RUNTIME = "13.0"
TEMP_CONFIG_ENV = "YOLO_CONFIG_DIR"
AUTOINSTALL_ENV = "YOLO_AUTOINSTALL"


def exact_distribution_versions() -> dict[str, str]:
    """Return exact versions after enforcing the pinned distribution set."""

    versions = {
        distribution: metadata.version(distribution)
        for distribution in EXPECTED_DISTRIBUTIONS
    }
    mismatches = {
        distribution: {"expected": expected, "actual": versions[distribution]}
        for distribution, expected in EXPECTED_DISTRIBUTIONS.items()
        if versions[distribution] != expected
    }
    if mismatches:
        raise RuntimeError(f"distribution version mismatch: {mismatches}")

    try:
        headless_version = metadata.version("opencv-python-headless")
    except metadata.PackageNotFoundError:
        headless_version = None
    if headless_version is not None:
        raise RuntimeError(
            "opencv-python-headless must not coexist with opencv-python "
            f"(found {headless_version})"
        )
    return versions


def tracker_arguments() -> SimpleNamespace:
    """Return the minimal pinned tracker arguments used by the smoke test."""

    return SimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.10,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.80,
        fuse_score=True,
        gmc_method="sparseOptFlow",
        proximity_thresh=0.50,
        appearance_thresh=0.80,
        with_reid=False,
        model="auto",
    )


def synthetic_inputs(cv2: Any, np: Any) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    """Build two deterministic frames and their external N x 6 detections."""

    frame_1 = np.zeros((128, 128, 3), dtype=np.uint8)
    for y_coordinate in range(8, 121, 16):
        for x_coordinate in range(8, 121, 16):
            cv2.circle(
                frame_1,
                (x_coordinate, y_coordinate),
                2,
                (255, 255, 255),
                -1,
            )
    frame_2 = np.zeros_like(frame_1)
    frame_2[1:, 1:] = frame_1[:-1, :-1]

    detections = (
        np.asarray(
            [
                [20.0, 20.0, 42.0, 62.0, 0.95, 0.0],
                [70.0, 25.0, 96.0, 66.0, 0.90, 1.0],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [21.0, 21.0, 43.0, 63.0, 0.94, 0.0],
                [71.0, 26.0, 97.0, 67.0, 0.89, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    return (frame_1, frame_2), detections


def verify_tracker(
    tracker_type: type[Any],
    boxes_type: type[Any],
    frames: tuple[Any, Any],
    detections: tuple[Any, Any],
    np: Any,
) -> dict[str, Any]:
    """Run one tracker over the same two external-detection frames."""

    arguments = tracker_arguments()
    tracker = tracker_type(arguments)
    outputs = []
    for frame, frame_detections in zip(frames, detections, strict=True):
        boxes = boxes_type(frame_detections.copy(), orig_shape=frame.shape[:2])
        outputs.append(tracker.update(boxes, img=frame))

    output_shapes = [list(output.shape) for output in outputs]
    if output_shapes != [[2, 8], [2, 8]]:
        raise RuntimeError(
            f"{tracker_type.__name__} returned unexpected shapes: {output_shapes}"
        )
    track_ids = [np.rint(output[:, 4]).astype(np.int64).tolist() for output in outputs]
    if track_ids[0] != track_ids[1] or len(set(track_ids[0])) != 2:
        raise RuntimeError(
            f"{tracker_type.__name__} did not preserve stable track IDs: {track_ids}"
        )
    return {
        "constructor_signature": str(inspect.signature(tracker_type)),
        "update_signature": str(inspect.signature(tracker_type.update)),
        "input_shapes": [list(item.shape) for item in detections],
        "output_dtypes": [str(output.dtype) for output in outputs],
        "output_shapes": output_shapes,
        "track_ids": track_ids,
    }


def verify_runtime(config_root: Path) -> dict[str, Any]:
    """Run all pinned package, CUDA, tracker, and LAP checks."""

    versions = exact_distribution_versions()

    import cv2
    import numpy as np
    import torch
    import torchvision
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.utils.matching import linear_assignment
    from ultralytics.utils import SETTINGS_FILE

    if str(torch.__version__) != EXPECTED_DISTRIBUTIONS["torch"]:
        raise RuntimeError(f"unexpected torch module version: {torch.__version__}")
    if str(torchvision.__version__) != EXPECTED_DISTRIBUTIONS["torchvision"]:
        raise RuntimeError(f"unexpected torchvision module version: {torchvision.__version__}")
    if torch.version.cuda != EXPECTED_CUDA_RUNTIME:
        raise RuntimeError(f"unexpected CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is unavailable")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("Torch did not expose CUDA device 0")

    settings_file = Path(SETTINGS_FILE).resolve()
    if not settings_file.is_relative_to(config_root.resolve()):
        raise RuntimeError(f"Ultralytics settings escaped the temporary directory: {settings_file}")

    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.empty(1, dtype=torch.float16, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    left = torch.ones((256, 256), dtype=torch.float16, device=device)
    right = torch.ones((256, 256), dtype=torch.float16, device=device)
    product = left @ right
    torch.cuda.synchronize(device)
    if product.dtype != torch.float16 or product.device != device:
        raise RuntimeError(
            f"unexpected FP16 result: dtype={product.dtype}, device={product.device}"
        )
    if product.shape != (256, 256) or product[0, 0].item() != 256.0:
        raise RuntimeError("FP16 matrix multiplication returned an unexpected result")
    peak_memory = torch.cuda.max_memory_allocated(device)

    frames, detections = synthetic_inputs(cv2, np)
    bytetrack_result = verify_tracker(
        BYTETracker,
        Boxes,
        frames,
        detections,
        np,
    )
    botsort_result = verify_tracker(
        BOTSORT,
        Boxes,
        frames,
        detections,
        np,
    )

    bot_arguments = tracker_arguments()
    if bot_arguments.gmc_method != "sparseOptFlow" or bot_arguments.with_reid:
        raise RuntimeError("BoT-SORT policy must use sparseOptFlow with ReID disabled")

    cost = np.asarray([[0.10, 0.90], [0.80, 0.20]], dtype=np.float32)
    matches, unmatched_rows, unmatched_columns = linear_assignment(
        cost,
        thresh=0.50,
        use_lap=True,
    )
    lap_matches = np.asarray(matches).tolist()
    lap_unmatched_rows = np.asarray(unmatched_rows).tolist()
    lap_unmatched_columns = np.asarray(unmatched_columns).tolist()
    if (
        lap_matches != [[0, 0], [1, 1]]
        or lap_unmatched_rows
        or lap_unmatched_columns
    ):
        raise RuntimeError(
            "LAP assignment failed: "
            f"matches={lap_matches}, rows={lap_unmatched_rows}, "
            f"columns={lap_unmatched_columns}"
        )

    return {
        "success": True,
        "versions": versions,
        "opencv_python_headless_installed": False,
        "cuda": {
            "available": True,
            "runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_0": {
                "name": torch.cuda.get_device_name(device),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
            },
        },
        "fp16_matrix": {
            "shape": list(product.shape),
            "dtype": str(product.dtype),
            "device": str(product.device),
            "peak_cuda_memory_allocated_bytes": peak_memory,
        },
        "trackers": {
            "bytetrack": bytetrack_result,
            "botsort": {
                **botsort_result,
                "gmc_method": bot_arguments.gmc_method,
                "with_reid": bot_arguments.with_reid,
            },
        },
        "lap": {
            "matches": lap_matches,
            "unmatched_rows": lap_unmatched_rows,
            "unmatched_columns": lap_unmatched_columns,
        },
    }


def restore_environment(previous: dict[str, str | None]) -> None:
    """Restore the two process variables changed for isolated verification."""

    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def main() -> int:
    """Emit one JSON result and return success only when every check passes."""

    previous_environment = {
        TEMP_CONFIG_ENV: os.environ.get(TEMP_CONFIG_ENV),
        AUTOINSTALL_ENV: os.environ.get(AUTOINSTALL_ENV),
    }
    captured_output = io.StringIO()
    try:
        with tempfile.TemporaryDirectory(prefix="geovision-ultralytics-") as temporary:
            config_root = Path(temporary)
            (config_root / "Ultralytics").mkdir()
            os.environ[TEMP_CONFIG_ENV] = str(config_root)
            os.environ[AUTOINSTALL_ENV] = "false"
            with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(
                captured_output
            ):
                result = verify_runtime(config_root)
    except Exception as error:  # noqa: BLE001 - the script must return structured failure JSON
        result = {
            "success": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    finally:
        restore_environment(previous_environment)

    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
