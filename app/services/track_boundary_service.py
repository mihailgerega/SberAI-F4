from functools import lru_cache
from typing import Iterable

import cv2
import os
import numpy as np
import torch
from PIL import Image
from app.schemas import TrackSegmentationResult
from transformers import Sam3Model, Sam3Processor
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()


class Sam3TrackBoundaryService:
    """
    Build a drivable-area / track-surface mask using SAM 3 text prompts.

    The idea is:
    1) prompt SAM 3 with several text phrases like "race track" / "track surface";
    2) choose the most plausible large mask;
    3) post-process it into a single connected component;
    4) derive a boundary polygon from the outer contour.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        prompts: Iterable[str] | None = None,
        threshold: float = 0.45,
        mask_threshold: float = 0.5,
        min_area_ratio: float = 0.10,
        max_area_ratio: float = 0.98,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.prompts = list(prompts or (
            "race track",
            "track surface",
            "asphalt track",
            "racetrack asphalt",
            "drivable surface",
        ))
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.token = os.getenv("HF_TOKEN") # Подгружаем токен для доступа к HuggingFace
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _ensure_rgb_pil(frame_bgr: np.ndarray) -> Image.Image:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected BGR image with shape HxWx3, got {frame_bgr.shape}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    @staticmethod
    def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
        mask_u8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num_labels <= 1:
            return mask.astype(bool)
        # skip background label 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas))
        return labels == largest_idx

    @staticmethod
    def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
        mask_u8 = mask.astype(np.uint8) * 255

        kernel = np.ones((9, 9), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

        # Fill internal holes by contour flood fill.
        h, w = mask_u8.shape[:2]
        flood = mask_u8.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        mask_u8 = mask_u8 if mask_u8 is not None else flood_inv

        mask_bool = mask_u8 > 0
        return Sam3TrackBoundaryService._largest_connected_component(mask_bool)

    @staticmethod
    def _contour_to_polygon_px(mask: np.ndarray) -> list[tuple[int, int]]:
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(2.0, 0.01 * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            approx = contour

        pts: list[tuple[int, int]] = []
        for point in approx.reshape(-1, 2):
            pts.append((int(point[0]), int(point[1])))

        # Ensure consistent winding and not too many repeated points.
        deduped: list[tuple[int, int]] = []
        for pt in pts:
            if not deduped or deduped[-1] != pt:
                deduped.append(pt)
        if len(deduped) >= 3 and deduped[0] == deduped[-1]:
            deduped.pop()
        return deduped

    @staticmethod
    def _is_plausible(mask: np.ndarray, h: int, w: int, min_area_ratio: float, max_area_ratio: float) -> bool:
        area_ratio = float(mask.mean())
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            return False

        # Prefer elongated / large surfaces rather than tiny blobs.
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return False
        bbox_w = xs.max() - xs.min() + 1
        bbox_h = ys.max() - ys.min() + 1
        bbox_area_ratio = (bbox_w * bbox_h) / float(h * w)
        return bbox_area_ratio >= min_area_ratio

    @lru_cache(maxsize=1)
    def _load_model(self):
        if Sam3Model is None or Sam3Processor is None:
            raise RuntimeError(
                "transformers Sam3Model/Sam3Processor are not available. "
                "Install the SAM 3-compatible Transformers build and the gated model access."
            )
        login(token=self.token)
        model = Sam3Model.from_pretrained(self.model_name).to(self.device) # pyright: ignore[reportArgumentType]
        processor = Sam3Processor.from_pretrained(self.model_name)
        model.eval()
        return model, processor

    def segment_track(self, frame_bgr: np.ndarray) -> TrackSegmentationResult:
        model, processor = self._load_model()
        image = self._ensure_rgb_pil(frame_bgr)
        h, w = frame_bgr.shape[:2]

        candidates: list[tuple[float, str, np.ndarray]] = []

        for prompt in self.prompts:
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)

            processed = processor.post_process_instance_segmentation(
                outputs,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(), # pyright: ignore[reportOptionalMemberAccess]
            )[0]

            masks = processed.get("masks")
            scores = processed.get("scores")
            if masks is None or len(masks) == 0:
                continue

            masks_np = masks.detach().cpu().numpy().astype(bool)
            scores_np = scores.detach().cpu().numpy() if scores is not None else np.ones((len(masks_np),), dtype=np.float32)

            for mask_np, score in zip(masks_np, scores_np):
                if mask_np.shape[:2] != (h, w):
                    mask_np = cv2.resize(mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if not self._is_plausible(mask_np, h, w, self.min_area_ratio, self.max_area_ratio):
                    continue
                # Score large masks a bit higher because track surface usually dominates the frame.
                composite_score = float(score) * (0.5 + 0.5 * float(mask_np.mean()))
                candidates.append((composite_score, prompt, mask_np))

        if not candidates:
            raise RuntimeError(
                "SAM3 did not return a plausible track mask. "
                "Try changing the prompts, lowering the threshold, or adding a fallback polygon."
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_prompt, best_mask = candidates[0]

        cleaned = self._postprocess_mask(best_mask)
        polygon = self._contour_to_polygon_px(cleaned)
        if len(polygon) < 3:
            raise RuntimeError("Track mask was found, but contour extraction failed.")

        return TrackSegmentationResult(
            mask=cleaned.astype(bool),
            polygon_px=polygon,
            prompt=best_prompt,
            score=float(best_score),
        )

    @staticmethod
    def polygon_to_normalized_points(polygon_px: list[tuple[int, int]], width: int, height: int) -> list[list[float]]:
        points: list[list[float]] = []
        for x, y in polygon_px:
            points.append([round(float(x) / float(width), 6), round(float(y) / float(height), 6)])
        return points

    @staticmethod
    def polygon_to_yolo_segmentation(class_id: int, polygon_px: list[tuple[int, int]], width: int, height: int) -> list[float]:
        seg: list[float] = [float(class_id)]
        for x, y in polygon_px:
            seg.extend([round(float(x) / float(width), 6), round(float(y) / float(height), 6)])
        return seg

    def build_track_mask_payload(self, frame_bgr: np.ndarray, frame_index: int) -> dict[str, object]:
        h, w = frame_bgr.shape[:2]
        result = self.segment_track(frame_bgr)

        return {
            "model_name": "track",
            "class_id": 0,
            "class_name": "track_limits",
            # "frame_index": frame_index,
            # "prompt_used": result.prompt,
            # "score": result.score,
            "points": self.polygon_to_normalized_points(result.polygon_px, w, h),
            "yolo_segmentation": self.polygon_to_yolo_segmentation(0, result.polygon_px, w, h),
            # "polygon_px": [[int(x), int(y)] for x, y in result.polygon_px],
            # "mask_shape": [h, w],
        }
