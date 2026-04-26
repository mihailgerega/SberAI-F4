import base64

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas import (
    FrameInferenceResponse,
    ViolationAnalysisResponse,
    ViolationAnalysisRequest,
    WheelOfftrackDetail,
    YoloMask,
)
from app.services import OfftrackDetector
from app.services import Sam3TrackBoundaryService
from app.services import (
    encode_bgr_to_data_url,
    encode_mask_to_data_url,
)
from app.services import WheelSegmentationService

router = APIRouter()


track_service = Sam3TrackBoundaryService()
wheel_service = WheelSegmentationService()
detector = OfftrackDetector()


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

    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode uploaded data as image frame",
        )
    return frame_bgr


def _build_analysis_payload(
    frame_bgr: np.ndarray,
    frame_index: int,
    track_prompts: list[str] | None = None,
):
    if track_prompts is not None:
        track_service.prompts = list(track_prompts)

    track_payload = track_service.build_track_mask_payload(frame_bgr, frame_index)
    wheel_payloads = wheel_service.build_wheel_mask_payload(frame_bgr, frame_index)

    analysis = detector.analyze(
        frame_bgr=frame_bgr,
        track_points=track_payload["points"], # pyright: ignore[reportArgumentType]
        wheel_masks=wheel_payloads,
    )

    return track_payload, wheel_payloads, analysis


@router.post("/infer/frame", response_model=FrameInferenceResponse)
async def infer_frame(
    frame: UploadFile = File(...),
    frame_index: int = Form(0),
) -> FrameInferenceResponse:
    frame_bgr = _decode_uploaded_frame(frame)

    wheel_payloads = wheel_service.build_wheel_mask_payload(frame_bgr, frame_index)
    track_payload = track_service.build_track_mask_payload(frame_bgr, frame_index)
    wheel_payloads.append(track_payload)
    # print(wheel_payloads[0])
    masks = [YoloMask(**mask) for mask in wheel_payloads]  # pyright: ignore[reportGeneralTypeIssues, reportArgumentType]

    return FrameInferenceResponse(
        frame_index=frame_index,
        frame_width=int(frame_bgr.shape[1]),
        frame_height=int(frame_bgr.shape[0]),
        frame_data_url=_encode_frame_to_data_url(frame_bgr),
        masks=masks,
    )

# Этот роутер пока что не используется. В моем понимании, где-то здесь должна быть ручка для того, чтобы выделять нарушение
@router.post("/infer/violation", response_model=ViolationAnalysisResponse)
async def infer_violation(
    frame: UploadFile = File(...),
    frame_index: int = Form(0),
    offtrack_threshold: float = Form(0.12),
    hard_violation_threshold: float = Form(0.25),
) -> ViolationAnalysisResponse:
    frame_bgr = _decode_uploaded_frame(frame)

    detector.offtrack_threshold = offtrack_threshold
    detector.hard_violation_threshold = hard_violation_threshold

    track_payload, wheel_payloads, analysis = _build_analysis_payload(
        frame_bgr=frame_bgr,
        frame_index=frame_index,
    )

    return ViolationAnalysisResponse(
        frame_index=frame_index,
        frame_width=int(frame_bgr.shape[1]),
        frame_height=int(frame_bgr.shape[0]),
        frame_data_url=encode_bgr_to_data_url(frame_bgr),
        annotated_frame_data_url=encode_bgr_to_data_url(analysis.annotated_frame_bgr),
        track_mask_data_url=encode_mask_to_data_url(analysis.track_mask),
        violation_detected=analysis.violation_detected,
        violation_score=analysis.violation_score,
        reason=analysis.reason,
        offtrack_wheels=[WheelOfftrackDetail(**d.__dict__) for d in analysis.offtrack_wheels],
        masks=[
            track_payload,
            *wheel_payloads,
        ],
    )
