# DeepCamera Streamlit MVP

Local video object detection app built with Streamlit, OpenCV, and YOLOv8.

## Features

* Video upload for mp4, avi, mov, and mkv files
* YOLOv8 inference with GPU-first execution when CUDA is available
* Frame skipping, confidence threshold, and inference size controls
* Annotated video preview and downloadable result
* CPU fallback when a GPU is not available

## Setup

1. Create and activate the virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. If you want CUDA acceleration, install a CUDA-enabled PyTorch build that matches your system before installing the rest of the requirements.

## Run

```bash
streamlit run app.py
```

## Notes for the target laptop

This MVP is tuned for a laptop RTX 3050 Ti with 16 GB RAM. Start with `yolov8n.pt`, 640 px inference size, and frame skipping set to 2 or 3 for smoother playback.

## TODO

- [ ] Resolve or remove the TODO and FIXME markers in 1 file:
      `teschSpec.md`
