## Tech Spec — Real-Time Video Object Detection Web App (Streamlit + YOLO)

### Target Hardware Profile

This spec is optimized for a local laptop with an RTX 3050 Ti-class GPU and 16 GB DDR5 RAM. The implementation should prioritize stable interactive performance over maximum throughput.

Recommended default assumptions:

* GPU-first inference when CUDA is available
* Conservative VRAM usage to remain stable on laptop GPUs
* Batch size of 1 for video inference
* Input resolution capped to 640 px on the long edge, with a lower inference size when needed
* Model default: `yolov8n.pt`

### 1. Overview

Build a Python-based web application that performs object detection on uploaded videos (and later webcam feed) using a deep learning model, and displays annotated results in real time via a browser UI.

---

### 2. Objectives

* Upload video file and run object detection frame-by-frame
* Display annotated video in a web UI
* Support near real-time processing on a local laptop GPU
* Allow future extension to live webcam streaming

---

### 3. Tech Stack

* **Frontend/UI:** Streamlit (Python web UI)
* **ML Model:** YOLOv8 (Ultralytics)
* **Video Processing:** OpenCV
* **Backend Logic:** Python 3.9+
* **Optional Acceleration:** CUDA (GPU support)
* **Precision:** FP16 on GPU, FP32 on CPU fallback

---

### 4. Core Components

#### 4.1 UI Layer

* File uploader (mp4, avi, mov)
* Video display panel (frame updates)
* Status indicators (processing/loading)

#### 4.2 Processing Pipeline

1. Load uploaded video file
2. Decode frames using OpenCV
3. Resize/preprocess frames before inference to reduce VRAM and latency
4. Run YOLO object detection per frame or on a configurable frame interval
5. Draw bounding boxes + labels
6. Stream frames back to UI

#### 4.3 ML Model

* Pretrained YOLOv8 (`yolov8n.pt` for speed)
* Input: RGB/BGR frames
* Output: bounding boxes, class labels, confidence scores
* Use `yolov8s.pt` only if the GPU memory budget remains stable during testing

---

### 5. Data Flow

```
User Upload → Streamlit UI → OpenCV Video Decode → YOLO Inference → Annotated Frames → Streamlit Display
```

---

### 6. Performance Strategy

* Cache model loading (`@st.cache_resource`)
* Process every Nth frame for speed optimization, with `N` configurable from the UI
* Resize frames dynamically, targeting 640 px on the long edge and dropping to 512 or 416 when FPS is too low
* Use GPU half precision when CUDA is available
* Keep inference batch size at 1
* Avoid storing all processed frames in memory; stream results incrementally
* Prefer generator-based frame iteration so Streamlit does not rerun expensive work on every UI update
* Optional GPU acceleration (CUDA)

Recommended tuning order for local testing:

1. Lower frame size
2. Increase frame skipping
3. Switch to `yolov8n.pt` if not already using it
4. Enable FP16 on GPU
5. Reduce video source resolution before upload if possible

---

### 7. Key Features (MVP)

* Video upload support
* Frame-by-frame object detection
* Bounding box visualization
* Browser-based playback
* Configurable inference speed controls: frame skip, confidence threshold, and output resolution

---

### 8. Future Enhancements

* Real-time webcam detection
* FPS counter + performance metrics
* Object tracking (e.g., ByteTrack)
* Download processed video output
* REST API backend using FastAPI
* Multi-stream support

---

### 9. Limitations (MVP)

* Not true streaming pipeline; updates are chunked or generator-driven rather than low-latency websocket video streaming
* Limited scalability for large videos
* CPU inference may be slow and should be treated as fallback only
* Higher-resolution videos may need automatic downscaling to stay within the laptop GPU memory budget

---

### 10. Deployment

* Local: `streamlit run app.py`
* Cloud options:

  * Streamlit Community Cloud
  * Docker + VPS
  * AWS / GCP with GPU instance

### 11. Acceptance Criteria

The MVP is acceptable when it can:

* Load a local video file without exhausting GPU memory
* Detect and annotate objects at interactive speed on the target laptop for short-to-medium clips
* Fall back cleanly to CPU if CUDA is unavailable
* Keep the UI responsive during inference

### 12. Implementation Notes

* Use a single cached model instance for the full session
* Prefer temporary file-backed video handling over loading the whole file into RAM
* Keep Streamlit state minimal to avoid rerun overhead
* Surface current device, model size, frame skip, input resolution, and estimated FPS in the UI

### 13. Project Todo List

#### Phase 1: Project Setup

* Create the Streamlit app skeleton
* Add `requirements.txt` with Streamlit, Ultralytics, OpenCV, and supporting packages
* Verify Python, CUDA, and GPU driver compatibility on the laptop
* Confirm the app can launch locally with a placeholder UI

#### Phase 2: Video Ingestion

* Implement video upload in the UI
* Validate supported file types and basic size limits
* Add temporary file handling for uploaded videos
* Confirm OpenCV can decode the uploaded video reliably

#### Phase 3: Detection Pipeline

* Load `yolov8n.pt` through a cached model loader
* Run inference on video frames one at a time
* Add GPU-first execution with CPU fallback
* Draw bounding boxes, labels, and confidence values on frames

#### Phase 4: Performance Controls

* Add frame skipping control to the UI
* Add confidence threshold control to the UI
* Add output resize control to the UI
* Enable FP16 on GPU and verify it remains stable
* Measure FPS and confirm memory usage stays within laptop limits

#### Phase 5: UI Integration

* Display annotated preview frames in the browser
* Show current device, model, FPS, and processing status
* Prevent expensive reruns from resetting inference progress
* Keep the interface responsive during long video processing jobs

#### Phase 6: Testing

* Test with a short low-resolution sample video
* Test with a longer medium-resolution sample video
* Test GPU and CPU fallback paths separately
* Test edge cases such as empty uploads, unsupported formats, and large files
* Verify outputs remain stable when frame skipping is increased

#### Phase 7: Hardening

* Reduce memory usage if uploads cause VRAM pressure
* Improve error messages for failed decode or inference steps
* Cache only what is needed and avoid storing full video output in RAM
* Confirm app behavior after repeated uploads in the same session

#### Phase 8: Final MVP

* Deliver a working video upload and detection app
* Keep the UI smooth on the RTX 3050 Ti laptop target
* Support annotated playback with configurable speed controls
* Document setup, launch steps, and local test expectations

---

here's a sample code:

```python
from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="coco8.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model

```
