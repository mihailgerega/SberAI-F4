from typing import Annotated, Any

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas import (
    ViolationAnalysisResponse,
)
from app.services import TrackSegmentationService
from app.services import (
    encode_bgr_to_data_url,
    encode_mask_to_data_url,
)
from app.services import WheelSegmentationService
from app.services import CarSegmentationService

router = APIRouter()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_FRAME_PIXELS = 25_000_000

track_service = TrackSegmentationService()
wheel_service = WheelSegmentationService()
car_service = CarSegmentationService()


class YoloSegDatasetWriter:
    def __init__(self, root: str | Path, split: str = "train") -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images" / split
        self.labels_dir = self.root / "labels" / split
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _points_to_yolo_seg_line(class_id: int, points: object) -> str:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError(f"Invalid polygon shape: {pts.shape}")

        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        return f"{class_id} {coords}"

    def save_sample(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        masks: list[dict[str, object]],
        source_id: str = "frame",
    ) -> tuple[Path, Path]:
        sample_id = f"{source_id}_{frame_index:06d}_{uuid4().hex[:8]}"
        img_path = self.images_dir / f"{sample_id}.jpg"
        label_path = self.labels_dir / f"{sample_id}.txt"

        ok = cv2.imwrite(str(img_path), frame_bgr)
        if not ok:
            raise IOError(f"Failed to write image: {img_path}")

        lines: list[str] = []
        for mask in masks:
            if "class_id" not in mask or "points" not in mask:
                continue
            class_id = int(mask["class_id"])
            lines.append(self._points_to_yolo_seg_line(class_id, mask["points"]))

        label_path.write_text("\n".join(lines), encoding="utf-8")
        return img_path, label_path


dataset_writer = YoloSegDatasetWriter("/data/f1_dataset", split="train")


def _decode_uploaded_frame(frame: UploadFile) -> np.ndarray:
    frame_bytes = frame.file.read()
    if not frame_bytes:
        raise HTTPException(status_code=400, detail="Uploaded frame is empty")
    if len(frame_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded frame is too large")

    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode uploaded data as image frame",
        )

    frame_pixels = int(frame_bgr.shape[0]) * int(frame_bgr.shape[1])
    if frame_pixels > MAX_FRAME_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=(
                "Decoded frame is too large. "
                f"Maximum is {MAX_FRAME_PIXELS} pixels, got {frame_pixels}."
            ),
        )
    return frame_bgr


@router.post("/infer/violation", response_model=ViolationAnalysisResponse)
async def infer_violation(
    frame: UploadFile = File(...),
    frame_index: Annotated[int, Form(ge=0)] = 0,
    offtrack_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.12,
    hard_violation_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.25,
) -> ViolationAnalysisResponse:
    frame_bgr = _decode_uploaded_frame(frame)

    if hard_violation_threshold < offtrack_threshold:
        raise HTTPException(
            status_code=422,
            detail=(
                "hard_violation_threshold must be greater than or equal "
                "to offtrack_threshold"
            ),
        )

    # detector = OfftrackDetector(
    #     offtrack_threshold=offtrack_threshold,
    #     hard_violation_threshold=hard_violation_threshold,
    # )
    

    try:
        track_payloads = track_service.build_track_mask_payload(frame_bgr, frame_index)
        # track_payload = _select_track_payload(track_payloads)
        wheel_payloads = wheel_service.build_wheel_mask_payload(frame_bgr, frame_index)
        car_payloads = car_service.build_car_mask_payload(frame_bgr, frame_index)
        
        all_payloads = [*track_payloads, *wheel_payloads, *car_payloads]

        dataset_writer.save_sample(
            frame_bgr,
            frame_index,
            all_payloads,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model inference failed: {type(exc).__name__}: {exc}",
        ) from exc
    return ViolationAnalysisResponse(
        frame_index=frame_index,
        frame_width=int(frame_bgr.shape[1]),
        frame_height=int(frame_bgr.shape[0]),
        frame_data_url=encode_bgr_to_data_url(frame_bgr),
        annotated_frame_data_url="",
        track_mask_data_url="",
        violation_mask_data_url="",
        violation_detected=False,
        violation_score=0.0,
        reason="",
        offtrack_wheels=None,
        violation_regions=None,
        masks=[
            *track_payloads,
            *wheel_payloads,
            *car_payloads,
        ],
    )
