from dataclasses import dataclass
from ultralytics import YOLO
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

class WheelSegmentationService:
    """
    Optional adapter for your trained segmentation .pt model.

    If ULTRALYTICS is available and WHEEL_MODEL_PATH is set, this service loads
    the model and returns wheel masks in the same payload format as your current stubs.
    Otherwise it falls back to a deterministic demo polygon so the rest of the app keeps running.
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        class_name: str = "wheel",
    ) -> None:
        self.model_path = model_path or os.getenv("WHEEL_MODEL_PATH", "")
        self.conf = conf
        self.iou = iou
        self.class_name = class_name
        self._model = None

    def _load_model(self):
        # print(self.model_path)
        if not self.model_path:
            return None
        if YOLO is None:
            return None
        if self._model is None:
            self._model = YOLO(self.model_path)
        # print(self._model)
        return self._model

    @staticmethod
    def _norm_points(points_xy: np.ndarray, width: int, height: int) -> list[list[float]]:
        return [[round(float(x) / float(width), 6), round(float(y) / float(height), 6)] for x, y in points_xy]

    @staticmethod
    def _to_yolo_segmentation(class_id: int, points_xy: np.ndarray, width: int, height: int) -> list[float]:
        seg: list[float] = [float(class_id)]
        for x, y in points_xy:
            seg.extend([round(float(x) / float(width), 6), round(float(y) / float(height), 6)])
        return seg

    @staticmethod
    def _fallback_wheel_polygons(frame_index: int) -> list[np.ndarray]:
        # Deterministic demo positions so the UI works even without the wheel model.
        t = frame_index % 120
        x0 = 0.35 + 0.0025 * t
        y0 = 0.62
        size = 0.045
        left = np.array([
            [x0, y0],
            [x0 + size, y0],
            [x0 + size, y0 + size],
            [x0, y0 + size],
        ], dtype=np.float32)

        x1 = 0.55 + 0.0020 * t
        right = np.array([
            [x1, y0 - 0.01],
            [x1 + size, y0 - 0.01],
            [x1 + size, y0 - 0.01 + size],
            [x1, y0 - 0.01 + size],
        ], dtype=np.float32)
        return [left, right]

    def build_wheel_mask_payload(self, frame_bgr: np.ndarray, frame_index: int) -> list[dict[str, object]]:
        height, width = frame_bgr.shape[:2]
        model = self._load_model()
        # print(self.model_path)
        # print(self._model)

        payload: list[dict[str, object]] = []

        # Раскомментировать, если хочется получать стандартные маски без запуска модели (константные т.е.)
        # if model is None:
        #     for idx, poly_norm in enumerate(self._fallback_wheel_polygons(frame_index)):
        #         poly_px = np.stack([poly_norm[:, 0] * width, poly_norm[:, 1] * height], axis=1)
        #         payload.append(
        #             {
        #                 "model_name": "wheel_segmentation_stub",
        #                 "class_id": 1,
        #                 "class_name": self.class_name,
        #                 # "instance_id": idx,
        #                 "points": [[round(float(x), 6), round(float(y), 6)] for x, y in poly_norm.tolist()],
        #                 "yolo_segmentation": self._to_yolo_segmentation(1, poly_px, width, height),
        #                 # "polygon_px": [[int(x), int(y)] for x, y in poly_px],
        #             }
        #         )
        #     return payload

        # Ultralytics segmentation path.
        results = model.predict(source=frame_bgr, conf=self.conf, iou=self.iou, verbose=False, retina_masks=True) # pyright: ignore[reportOptionalMemberAccess]
        if not results:
            return payload

        result = results[0]
        if result.masks is None or result.boxes is None:
            return payload

        cls_names = getattr(result, "names", {}) or {}
        mask_data = result.masks.data.detach().cpu().numpy()  # pyright: ignore[reportAttributeAccessIssue] # [N, H, W]
        polygons = result.masks.xy  # list of polygons in absolute pixels
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int) # pyright: ignore[reportAttributeAccessIssue]
        scores = result.boxes.conf.detach().cpu().numpy() # pyright: ignore[reportAttributeAccessIssue]

        instance_id = 0
        for i, poly in enumerate(polygons):
            if poly is None or len(poly) < 3:
                continue
            cls_id = int(class_ids[i]) if i < len(class_ids) else 1
            cls_name = str(cls_names.get(cls_id, self.class_name))
            if cls_name != self.class_name and cls_id != 1:
                continue

            poly_arr = np.asarray(poly, dtype=np.float32)
            payload.append(
                {
                    "model_name": "wheels",
                    "class_id": 1,
                    "class_name": self.class_name,
                    # "instance_id": instance_id,
                    # "confidence": float(scores[i]) if i < len(scores) else None,
                    "points": self._norm_points(poly_arr, width, height),
                    "yolo_segmentation": self._to_yolo_segmentation(1, poly_arr, width, height),
                    # "polygon_px": [[int(x), int(y)] for x, y in poly_arr],
                }
            )
            instance_id += 1

        return payload
