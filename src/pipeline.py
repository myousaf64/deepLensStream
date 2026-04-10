from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch


@dataclass
class DeviceInfo:
    label: str
    device: str
    half_precision: bool


@dataclass
class VideoStats:
    frames_total: int
    frames_processed: int
    detections: int
    fps: float
    width: int
    height: int
    device: str
    model_name: str
    frame_skip: int
    inference_size: int
    output_path: str


def get_device_info() -> DeviceInfo:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return DeviceInfo(label=f"cuda:0 - {gpu_name}", device="cuda:0", half_precision=True)

    return DeviceInfo(label="cpu", device="cpu", half_precision=False)


def resize_for_inference(frame: np.ndarray, target_long_edge: int) -> np.ndarray:
    height, width = frame.shape[:2]
    current_long_edge = max(height, width)

    if current_long_edge <= target_long_edge:
        return frame

    scale = target_long_edge / current_long_edge
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def annotate_frame(
    model,
    frame: np.ndarray,
    conf_threshold: float,
    inference_size: int,
    device: str,
    half_precision: bool,
) -> tuple[np.ndarray, int]:
    resized_frame = resize_for_inference(frame, inference_size)
    results = model.predict(
        source=resized_frame,
        conf=conf_threshold,
        imgsz=inference_size,
        device=device,
        half=half_precision,
        verbose=False,
    )
    result = results[0]
    annotated_frame = result.plot()

    if annotated_frame.shape[:2] != frame.shape[:2]:
        annotated_frame = cv2.resize(
            annotated_frame,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    return annotated_frame, len(result.boxes)


def process_video(
    model,
    model_name: str,
    input_path: Path,
    output_path: Path,
    conf_threshold: float,
    frame_skip: int,
    inference_size: int,
    device: str,
    half_precision: bool,
    preview_every: int = 15,
    on_preview: Optional[Callable[[np.ndarray, int, int], None]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> VideoStats:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Could not read video dimensions.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError("Could not open the output video writer.")

    frames_processed = 0
    detections = 0
    frame_index = 0
    frame_skip = max(1, frame_skip)
    preview_every = max(1, preview_every)

    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            if frame_index % frame_skip == 0:
                annotated_frame, detected_boxes = annotate_frame(
                    model=model,
                    frame=frame,
                    conf_threshold=conf_threshold,
                    inference_size=inference_size,
                    device=device,
                    half_precision=half_precision,
                )
                output_frame = annotated_frame
                frames_processed += 1
                detections += detected_boxes
            else:
                output_frame = frame

            writer.write(output_frame)

            if on_preview and (frame_index % preview_every == 0 or (frames_total > 0 and frame_index == frames_total - 1)):
                on_preview(output_frame, frame_index + 1, frames_total)

            if on_progress and frames_total > 0:
                on_progress((frame_index + 1) / frames_total)

            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return VideoStats(
        frames_total=frame_index,
        frames_processed=frames_processed,
        detections=detections,
        fps=fps,
        width=width,
        height=height,
        device=device,
        model_name=model_name,
        frame_skip=frame_skip,
        inference_size=inference_size,
        output_path=str(output_path),
    )