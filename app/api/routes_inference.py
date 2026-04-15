from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas import FrameInferenceResponse, YoloMask
from app.services import build_stub_masks

router = APIRouter()


def _encode_frame_to_data_url(frame_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame")

    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


@router.post("/infer/frame", response_model=FrameInferenceResponse)
async def infer_frame(
    frame: UploadFile = File(...),
    frame_index: int = Form(0),
) -> FrameInferenceResponse:
    frame_bytes = await frame.read()
    if not frame_bytes:
        raise HTTPException(status_code=400, detail="Uploaded frame is empty")

    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode uploaded data as image frame",
        )

    frame_height, frame_width = frame_bgr.shape[:2]
    masks_payload = build_stub_masks(frame_index=frame_index)
    masks = [YoloMask(**mask) for mask in masks_payload]

    return FrameInferenceResponse(
        frame_index=frame_index,
        frame_width=frame_width,
        frame_height=frame_height,
        frame_data_url=_encode_frame_to_data_url(frame_bgr),
        masks=masks,
    )
