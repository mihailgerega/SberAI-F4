from pydantic import BaseModel, Field


class YoloMask(BaseModel):
    model_name: str = Field(..., description="Model alias that produced the mask")
    class_id: int = Field(..., description="Class id in YOLO segmentation")
    class_name: str = Field(..., description="Human readable class name")
    yolo_segmentation: str = Field(
        ..., description='YOLO segmentation line: "cls x1 y1 x2 y2 ..."'
    )
    points: list[list[float]] = Field(
        ..., description="Normalized polygon points as [x, y] pairs"
    )


class FrameInferenceResponse(BaseModel):
    frame_index: int
    frame_width: int
    frame_height: int
    frame_data_url: str = Field(..., description="Frame encoded as data URL")
    masks: list[YoloMask]
