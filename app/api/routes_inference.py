import base64
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas import (
    FrameInferenceResponse,
    ViolationAnalysisResponse,
    ViolationRegion,
    YoloMask,
)
from app.services import OfftrackDetector, TrackSegmentationService
from app.services import (
    encode_bgr_to_data_url,
    encode_mask_to_data_url,
)
from app.services import WheelSegmentationService

router = APIRouter()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_FRAME_PIXELS = 25_000_000

track_service = TrackSegmentationService()
wheel_service = WheelSegmentationService()


def _encode_frame_to_data_url(frame_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame")

    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_point(x: int, y: int, width: int, height: int) -> list[float]:
    return [
        round(_clip01(x / float(width)), 6),
        round(_clip01(y / float(height)), 6),
    ]


def _to_yolo_segmentation(
    class_id: int,
    points: list[list[float]],
) -> list[float]:
    segmentation = [float(class_id)]
    for x, y in points:
        segmentation.extend([x, y])
    return segmentation


def _polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) * 0.5


def _select_track_payload(track_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not track_payloads:
        raise RuntimeError("Track segmentation did not return any masks")

    return max(
        track_payloads,
        key=lambda payload: _polygon_area(payload.get("points", [])),
    )


def _mask_to_violation_regions(
    violation_mask: np.ndarray,
    violation_score: float,
) -> list[ViolationRegion]:
    height, width = violation_mask.shape[:2]
    mask_u8 = violation_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    regions: list[ViolationRegion] = []
    min_area = max(1.0, width * height * 0.000001)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = max(1.0, 0.01 * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            x2 = x + box_width - 1
            y2 = y + box_height - 1
            points_px = [(x, y), (x2, y), (x2, y2), (x, y2)]
        else:
            points_px = [
                (int(point[0]), int(point[1]))
                for point in approx.reshape(-1, 2)
            ]

        points = [
            _normalize_point(
                min(max(x, 0), width - 1),
                min(max(y, 0), height - 1),
                width,
                height,
            )
            for x, y in points_px
        ]

        x, y, box_width, box_height = cv2.boundingRect(contour)
        x2 = min(width - 1, x + box_width - 1)
        y2 = min(height - 1, y + box_height - 1)

        regions.append(
            ViolationRegion(
                model_name="offtrack_detector",
                class_id=2,
                class_name="track_limit_violation",
                confidence=round(_clip01(float(violation_score)), 6),
                bbox_xyxy=[
                    round(x / float(width), 6),
                    round(y / float(height), 6),
                    round(x2 / float(width), 6),
                    round(y2 / float(height), 6),
                ],
                points=points,
                yolo_segmentation=_to_yolo_segmentation(2, points),
            )
        )

    return regions


@router.post("/infer/frame", response_model=FrameInferenceResponse)
async def infer_frame(
    frame: UploadFile = File(...),
    frame_index: int = Form(0),
) -> FrameInferenceResponse:
    frame_bgr = _decode_uploaded_frame(frame)

    wheel_payloads = wheel_service.build_wheel_mask_payload(frame_bgr, frame_index)
    track_payloads = track_service.build_track_mask_payload(frame_bgr, frame_index)
    masks = [
        YoloMask(**mask) for mask in [*track_payloads, *wheel_payloads]
    ]  # pyright: ignore[reportGeneralTypeIssues, reportArgumentType]

    return FrameInferenceResponse(
        frame_index=frame_index,
        frame_width=int(frame_bgr.shape[1]),
        frame_height=int(frame_bgr.shape[0]),
        frame_data_url=_encode_frame_to_data_url(frame_bgr),
        masks=masks,
    )


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

    detector = OfftrackDetector(
        offtrack_threshold=offtrack_threshold,
        hard_violation_threshold=hard_violation_threshold,
    )

    try:
        track_payloads = track_service.build_track_mask_payload(frame_bgr, frame_index)
        track_payload = _select_track_payload(track_payloads)
        wheel_payloads = wheel_service.build_wheel_mask_payload(frame_bgr, frame_index)
        analysis = detector.analyze(
            frame_bgr=frame_bgr,
            track_points=track_payload["points"],  # pyright: ignore[reportArgumentType]
            wheel_masks=wheel_payloads,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model inference failed: {type(exc).__name__}: {exc}",
        ) from exc

    violation_regions = _mask_to_violation_regions(
        violation_mask=analysis.violation_mask,
        violation_score=analysis.violation_score,
    )

    return ViolationAnalysisResponse(
        frame_index=frame_index,
        frame_width=int(frame_bgr.shape[1]),
        frame_height=int(frame_bgr.shape[0]),
        frame_data_url=encode_bgr_to_data_url(frame_bgr),
        annotated_frame_data_url=encode_bgr_to_data_url(analysis.annotated_frame_bgr),
        track_mask_data_url=encode_mask_to_data_url(analysis.track_mask),
        violation_mask_data_url=encode_mask_to_data_url(analysis.violation_mask),
        violation_detected=analysis.violation_detected,
        violation_score=analysis.violation_score,
        reason=analysis.reason,
        offtrack_wheels=analysis.offtrack_wheels,
        violation_regions=violation_regions,
        masks=[
            *track_payloads,
            *wheel_payloads,
        ],
    )
