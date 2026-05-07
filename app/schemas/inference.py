from typing import Any
from pydantic import BaseModel, Field
from dataclasses import dataclass
import numpy as np


class YoloMask(BaseModel):
    model_name: str = Field(..., description="Model alias that produced the mask")
    class_id: int = Field(..., description="Class id in YOLO segmentation")
    class_name: str = Field(..., description="Human readable class name")
    # yolo_segmentation: list[float] = Field( # Крайне не рекомендую передавать этот параметр, раздувает конечное изображение на фронтенде
    #     ..., description='YOLO segmentation line: "cls x1 y1 x2 y2 ..."'
    # )
    points: list[list[float]] = Field(
        ..., description="Normalized polygon points as [x, y] pairs"
    )

class FrameInferenceResponse(BaseModel):
    frame_index: int
    frame_width: int
    frame_height: int
    frame_data_url: str = Field(..., description="Frame encoded as data URL")
    masks: list[YoloMask]


class ViolationAnalysisRequest(BaseModel):
    frame_index: int = Field(ge=0)
    offtrack_threshold: float = Field(default=0.12, ge=0.0, le=1.0)
    hard_violation_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    track_prompts: list[str] | None = None

class WheelOfftrackDetail(BaseModel):
    instance_id: int
    outside_ratio: float
    outside_pixels: int
    total_pixels: int

class ViolationAnalysisResponse(BaseModel):
    frame_index: int
    frame_width: int
    frame_height: int
    frame_data_url: str
    annotated_frame_data_url: str
    track_mask_data_url: str
    violation_detected: bool
    violation_score: float
    reason: str
    offtrack_wheels: list[WheelOfftrackDetail]
    masks: list[dict[str, Any]]

@dataclass(slots=True)
class OfftrackAnalysisResult:
    violation_detected: bool
    violation_score: float
    reason: str
    offtrack_wheels: list[WheelOfftrackDetail]
    annotated_frame_bgr: np.ndarray
    track_mask: np.ndarray
    violation_mask: np.ndarray

@dataclass(slots=True)
class TrackSegmentationResult:
    mask: np.ndarray  # bool array, HxW
    polygon_px: list[tuple[int, int]]
    prompt: str
    score: float

@dataclass(slots=True)
class WheelMaskResult:
    masks: list[dict[str, object]]
