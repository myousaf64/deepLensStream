from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

from src.pipeline import DeviceInfo, get_device_info, process_video


APP_TITLE = "DeepCamera"
DEFAULT_MODEL = "yolo26n.pt"
VALID_EXTENSIONS = ["mp4", "avi", "mov", "mkv"]
MODELS_DIR = Path("models")
FALLBACK_MODELS = ["yolov8n.pt", "yolov8s.pt"]


st.set_page_config(page_title=APP_TITLE, page_icon="🎥", layout="wide")


def apply_dynamic_style(compact_layout: bool, video_size: str) -> None:
    max_height = {"small": 260, "medium": 380, "large": 520}[video_size]
    top_padding = "0.8rem" if compact_layout else "1.5rem"
    bottom_padding = "1rem" if compact_layout else "2rem"

    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(35, 92, 147, 0.18), transparent 28%),
                    radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 22%),
                    linear-gradient(180deg, #07111f 0%, #0b1726 48%, #08101b 100%);
                color: #e5eefb;
            }}
            .block-container {{
                padding-top: {top_padding};
                padding-bottom: {bottom_padding};
            }}
            .panel {{
                background: rgba(8, 17, 28, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 18px 60px rgba(0, 0, 0, 0.25);
            }}
            .eyebrow {{
                text-transform: uppercase;
                letter-spacing: 0.18em;
                font-size: 0.75rem;
                color: #93c5fd;
                margin-bottom: 0.4rem;
            }}
            .hero {{
                font-size: 2.3rem;
                line-height: 1.05;
                font-weight: 700;
                margin-bottom: 0.4rem;
            }}
            .subtle {{
                color: rgba(226, 232, 240, 0.78);
                font-size: 0.98rem;
            }}
            div[data-testid='stVideo'] video {{
                max-height: {max_height}px;
                object-fit: contain;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    return YOLO(model_name)


def discover_models() -> list[str]:
    local_models: list[str] = []

    if MODELS_DIR.exists():
        local_models = sorted([str(path.as_posix()) for path in MODELS_DIR.glob("*.pt")])

    if local_models:
        return local_models

    return FALLBACK_MODELS


def save_upload_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def create_output_path() -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        return Path(temp_file.name)


def frame_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_header(device_info: DeviceInfo, debug_mode: bool) -> None:
    st.markdown('<div class="eyebrow">Local AI video inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero">DeepCamera MVP</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">Upload a video, run YOLOv8 detection locally, and keep the pipeline tuned for an RTX 3050 Ti laptop.</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Mode", "Debug" if debug_mode else "Clean")
    col_b.metric("Default model", DEFAULT_MODEL)
    col_c.metric("Upload formats", ", ".join(VALID_EXTENSIONS).upper())

    if debug_mode:
        st.caption(f"Runtime device: {device_info.label}")


def render_presentation_controls() -> tuple[bool, bool, bool, bool, str]:
    if "debug_mode" not in st.session_state:
        st.session_state["debug_mode"] = False
    if "compact_layout" not in st.session_state:
        st.session_state["compact_layout"] = False
    if "compare_view" not in st.session_state:
        st.session_state["compare_view"] = True
    if "video_size" not in st.session_state:
        st.session_state["video_size"] = "medium"

    with st.container():
        st.caption("Presentation Controls")
        col1, col2, col3, col4 = st.columns(4)
        col1.toggle("Debug", key="debug_mode")
        col2.toggle("Compact", key="compact_layout")
        col3.toggle("Compare", key="compare_view")
        col4.selectbox(
            "Video size",
            options=["small", "medium", "large"],
            key="video_size",
        )

    return (
        bool(st.session_state["debug_mode"]),
        bool(st.session_state["compact_layout"]),
        bool(st.session_state["compare_view"]),
        str(st.session_state["video_size"]),
    )


def main() -> None:
    device_info = get_device_info()
    model_choices = discover_models()
    debug_mode, compact_layout, compare_view, video_size = render_presentation_controls()
    apply_dynamic_style(compact_layout=compact_layout, video_size=video_size)
    render_header(device_info, debug_mode)

    with st.sidebar:
        st.subheader("Inference Controls")
        default_index = 0
        for index, item in enumerate(model_choices):
            if item.endswith(DEFAULT_MODEL):
                default_index = index
                break

        model_name = st.selectbox(
            "Model",
            model_choices,
            index=default_index,
            help="Automatically lists local .pt files from the models folder. Smaller models are faster on this laptop.",
        )
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        frame_skip = st.slider("Frame skip", 1, 8, 2, 1)
        inference_size = st.select_slider("Inference long edge", options=[320, 416, 512, 640], value=640)
        preview_every = st.slider("Preview refresh interval", 1, 60, 15, 1)
        write_preview = st.checkbox("Show live preview while processing", value=True)
        st.caption("For the RTX 3050 Ti target, start with a small model, 640 px, and frame skip set to 2 or 3.")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a video", type=VALID_EXTENSIONS)
    process_button = st.button("Run detection", type="primary", use_container_width=True, disabled=uploaded_file is None)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("Upload a local video to start.")
        return

    preview_col = None
    if compare_view:
        left, right = st.columns(2)
        with left:
            st.subheader("Original")
            st.video(uploaded_file)
        with right:
            st.subheader("Annotated")
            preview_col = st.container()
            preview_col.info("Run detection to populate this panel.")
    else:
        st.video(uploaded_file)

    if not process_button:
        st.caption("Adjust controls in the sidebar, then run detection.")
        return

    input_path = save_upload_to_temp(uploaded_file)
    output_path = create_output_path()

    try:
        model = load_model(model_name)
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        try:
            input_path.unlink(missing_ok=True)
        except TypeError:
            if input_path.exists():
                input_path.unlink()
        return

    preview_placeholder = preview_col.empty() if preview_col is not None else st.empty()
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()
    started_at = time.perf_counter()

    def on_preview(frame, current_frame: int, total_frames: int) -> None:
        if not write_preview:
            return
        preview_placeholder.image(
            frame_to_rgb(frame),
            caption=f"Preview frame {current_frame}/{total_frames or '?'}",
            use_container_width=True,
        )

    def on_progress(value: float) -> None:
        progress_bar.progress(min(max(value, 0.0), 1.0))

    status_placeholder.write("Processing video...")

    try:
        stats = process_video(
            model=model,
            model_name=model_name,
            input_path=input_path,
            output_path=output_path,
            conf_threshold=confidence,
            frame_skip=frame_skip,
            inference_size=inference_size,
            device=device_info.device,
            half_precision=device_info.half_precision,
            preview_every=preview_every,
            on_preview=on_preview,
            on_progress=on_progress,
        )
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        return
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except TypeError:
            if input_path.exists():
                input_path.unlink()

    progress_bar.progress(1.0)
    status_placeholder.success("Processing complete.")
    elapsed_seconds = round(time.perf_counter() - started_at, 2)

    output_bytes = output_path.read_bytes()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames", f"{stats.frames_total}")
    col2.metric("Inference frames", f"{stats.frames_processed}")
    col3.metric("Detections", f"{stats.detections}")
    col4.metric("Device", stats.device)

    if compare_view and preview_col is not None:
        with preview_col:
            st.video(output_bytes)
    else:
        st.subheader("Annotated output")
        st.video(output_bytes)

    st.download_button(
        label="Download processed video",
        data=output_bytes,
        file_name=f"{Path(uploaded_file.name).stem}_annotated.mp4",
        mime="video/mp4",
        use_container_width=True,
    )

    with st.expander("Run summary", expanded=False):
        st.write(
            {
                "video_resolution": f"{stats.width}x{stats.height}",
                "fps": round(stats.fps, 2),
                "model": model_name,
                "frame_skip": stats.frame_skip,
                "inference_size": stats.inference_size,
                "output_path": stats.output_path,
                "elapsed_seconds": elapsed_seconds,
            }
        )

    if debug_mode:
        with st.expander("Debug Details", expanded=True):
            st.write(
                {
                    "runtime_device": device_info.label,
                    "selected_model": model_name,
                    "half_precision": device_info.half_precision,
                    "compare_view": compare_view,
                    "compact_layout": compact_layout,
                    "video_size": video_size,
                }
            )


if __name__ == "__main__":
    main()