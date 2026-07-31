
"""Streamlit interface for running video frames through the detection API.

Supports two modes:
  1. Upload Video   - process a pre-recorded video file frame by frame.
  2. Live Webcam    - realtime detection on the browser's webcam feed via
                       streamlit-webrtc, sending sampled frames to the API.
"""

import queue
import threading
import time
from collections import Counter
from pathlib import Path
import tempfile

import av
import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

DEFAULT_URL = "https://perkiness-aftermost-diffusive.ngrok-free.dev/predict"

# Same size as YOLO training
MODEL_SIZE = (640, 640)


def annotate_frame(frame, detections):
    """Draw API detections onto an OpenCV BGR frame."""

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])

        label = f'{detection["class"]} {detection["confidence"]:.2f}'

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return frame


def request_detections(session, url, frame, timeout=30):
    """Send one frame to the FastAPI endpoint."""

    success, encoded = cv2.imencode(".jpg", frame)

    if not success:
        raise ValueError("Could not encode frame.")

    response = session.post(
        url,
        files={
            "file": (
                "frame.jpg",
                encoded.tobytes(),
                "image/jpeg",
            )
        },
        timeout=timeout,
    )

    response.raise_for_status()

    payload = response.json()

    return payload.get("detections", [])


def check_api(api_url):
    """
    Check whether the API server is reachable.

    Returns:
        (True, message)
        (False, message)
    """

    health_url = api_url.replace("/predict", "/health")

    try:
        response = requests.get(health_url, timeout=5)

        if response.status_code == 200:
            return True, "API server is online."

        elif response.status_code == 502:
            return False, "API server is offline."

        elif response.status_code == 503:
            return False, "Server is running but the YOLO model failed to load."

        else:
            return False, f"Server returned HTTP {response.status_code}"

    except requests.exceptions.ConnectTimeout:
        return False, "Connection timed out."

    except requests.exceptions.ReadTimeout:
        return False, "Server is taking too long to respond."

    except requests.exceptions.ConnectionError:
        return False, "Unable to connect to the server."

    except requests.exceptions.InvalidURL:
        return False, "Invalid API URL."

    except Exception as e:
        return False, str(e)


# =============================================================================
# Realtime webcam video processor
# =============================================================================
class DetectionVideoProcessor:
    """
    streamlit-webrtc video processor.

    Every incoming browser frame is drawn immediately (so playback stays
    smooth), but only every `frame_step`-th frame is actually sent to the
    API. Detection calls run in a background thread so a slow/laggy API
    response never blocks the video pipeline - the processor just keeps
    reusing the most recent detections until a new result arrives.
    """

    def __init__(self, api_url, frame_step, stats_queue):
        self.api_url = api_url
        self.frame_step = max(1, frame_step)
        self.stats_queue = stats_queue

        self.session = requests.Session()
        self.frame_count = 0
        self.last_detections = []

        self._lock = threading.Lock()
        self._inflight = False

    def _run_detection(self, frame):
        try:
            detections = request_detections(
                self.session,
                self.api_url,
                frame,
                timeout=10,
            )
            with self._lock:
                self.last_detections = detections

            # Report stats back to the main Streamlit thread.
            try:
                self.stats_queue.put_nowait(
                    {
                        "detections": detections,
                        "timestamp": time.time(),
                    }
                )
            except queue.Full:
                pass

        except Exception as exc:
            try:
                self.stats_queue.put_nowait({"error": str(exc)})
            except queue.Full:
                pass
        finally:
            with self._lock:
                self._inflight = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        self.frame_count += 1

        resized = cv2.resize(img, MODEL_SIZE, interpolation=cv2.INTER_AREA)

        should_dispatch = (
            self.frame_count % self.frame_step == 0 and not self._inflight
        )

        if should_dispatch:
            with self._lock:
                self._inflight = True
            threading.Thread(
                target=self._run_detection,
                args=(resized.copy(),),
                daemon=True,
            ).start()

        with self._lock:
            detections = self.last_detections

        # Scale detection boxes (in MODEL_SIZE space) back up to the
        # original frame resolution before drawing.
        scale_x = img.shape[1] / MODEL_SIZE[0]
        scale_y = img.shape[0] / MODEL_SIZE[1]

        scaled_detections = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            scaled_detections.append(
                {
                    "class": detection["class"],
                    "confidence": detection["confidence"],
                    "bbox": [
                        x1 * scale_x,
                        y1 * scale_y,
                        x2 * scale_x,
                        y2 * scale_y,
                    ],
                }
            )

        annotated = annotate_frame(img, scaled_detections)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# =============================================================================
# Page setup
# =============================================================================
st.set_page_config(
    page_title="Video Emotion Detection",
    layout="wide",
)

st.title("Video Emotion Detection")
st.write("Run YOLO emotion detection on an uploaded video, or live from your webcam.")

# -----------------------------------------------------------------------------
with st.sidebar:

    st.header("Settings")

    api_url = st.text_input(
        "Prediction API URL",
        value=DEFAULT_URL,
    )

    frame_step = st.slider(
        "Process every Nth frame",
        min_value=1,
        max_value=20,
        value=3,
    )

    minimum_fps = st.number_input(
        "Minimum FPS Benchmark",
        min_value=1.0,
        max_value=60.0,
        value=10.0,
        step=0.5,
    )
    st.divider()
    st.header("Status")

    health_label, refresh_button = st.columns([5, 1])
    health_label.caption("API server status")
    retry_health_check = refresh_button.button(
        "↻",
        key="retry_api_health",
        help="Check the API server again",
        width="content",
    )

    if retry_health_check:
        st.toast("Checking API server…")

    healthy, message = check_api(api_url)

    if healthy:
        st.success(f"🟢 {message}")
    else:
        st.error(f"🔴 {message}")
        st.stop()

# -----------------------------------------------------------------------------
label_col, toggle_col = st.columns([3, 1])
is_realtime = st.session_state.get("realtime_mode", False)
label_col.subheader("📷 Live Webcam (Realtime)" if is_realtime else "📤 Upload Video")
realtime_mode = toggle_col.toggle(
    "Realtime Webcam",
    key="realtime_mode",
    help="Off = process an uploaded video file. On = live detection from your browser's webcam.",
)

st.divider()

# =============================================================================
# MODE 1: Upload Video (original behaviour, unchanged)
# =============================================================================
if not realtime_mode:

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
    )

    if uploaded_video is None:
        st.info("Upload a video to begin.")
        st.stop()

    # Original uploaded video
    st.video(uploaded_video, width=400)

    if not st.button("Run Detection", type="primary"):
        st.stop()

    suffix = Path(uploaded_video.name).suffix or ".mp4"

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(uploaded_video.getbuffer())
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)

        if not capture.isOpened():
            st.error("Could not open uploaded video.")
            st.stop()

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)

        st.caption(f"Frames: {total_frames:,} | FPS: {fps:.2f}")

        progress = st.progress(0, text="Starting...")

        preview = st.empty()
        status = st.empty()

        class_counts = Counter()
        detection_rows = []

        session = requests.Session()

        frame_number = 0
        processed_frames = 0

        start_time = time.perf_counter()

        while True:

            success, frame = capture.read()

            if not success:
                break

            frame_number += 1

            if frame_number % frame_step != 0:
                continue

            # Resize to the SAME size used during training
            frame = cv2.resize(
                frame,
                MODEL_SIZE,
                interpolation=cv2.INTER_AREA,
            )

            try:
                detections = request_detections(session, api_url, frame)
            except Exception as e:
                st.error(f"Frame {frame_number}: {e}")
                break

            processed_frames += 1

            for detection in detections:

                class_counts[detection["class"]] += 1

                detection_rows.append(
                    {
                        "Frame": frame_number,
                        "Emotion": detection["class"],
                        "Confidence": round(float(detection["confidence"]), 3),
                    }
                )

            annotated = annotate_frame(frame.copy(), detections)

            # Show preview without stretching
            preview.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption=f"Frame {frame_number}",
                width=600,
            )

            status.write(f"Processed Frames: {processed_frames}")

            if total_frames:
                progress.progress(
                    min(frame_number / total_frames, 1.0),
                    text=f"Processing frame {frame_number}/{total_frames}",
                )

        capture.release()
        session.close()

        elapsed = time.perf_counter() - start_time

        detection_fps = processed_frames / elapsed if elapsed > 0 else 0

        progress.progress(1.0, text="Detection Complete")

        # ---------------------------------------------------------------
        st.subheader("Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Frames Processed", processed_frames)
        col2.metric("Total Detections", sum(class_counts.values()))
        col3.metric("Detection FPS", f"{detection_fps:.2f}")

        st.subheader("Performance")

        if detection_fps >= minimum_fps:
            st.success(f"Realtime benchmark met ({detection_fps:.2f} FPS)")
        else:
            st.warning(f"Realtime benchmark NOT met ({detection_fps:.2f} FPS)")

        # ---------------------------------------------------------------
        if class_counts:

            st.subheader("Emotion Counts")

            st.bar_chart(
                pd.Series(class_counts).sort_values(ascending=False)
            )

            st.subheader("Detection Details")

            st.dataframe(
                pd.DataFrame(detection_rows),
                width="stretch",
            )

        else:
            st.info("No detections found.")

    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


# =============================================================================
# MODE 2: Live Webcam (Realtime) - new
# =============================================================================
else:

    st.caption(
        "Uses your browser's webcam. Frames are streamed live; every "
        f"{frame_step}th frame is sent to the API for detection, and the "
        "most recent result is drawn on the live video continuously."
    )

    if "stats_queue" not in st.session_state:
        st.session_state.stats_queue = queue.Queue(maxsize=100)

    if "class_counts" not in st.session_state:
        st.session_state.class_counts = Counter()

    if "last_fps_ts" not in st.session_state:
        st.session_state.last_fps_ts = []

    # IMPORTANT: video_processor_factory can be invoked by streamlit-webrtc
    # from a background worker thread that has no Streamlit script context,
    # so st.session_state is NOT safely accessible inside it. Pull the
    # queue (and any other settings) into plain local variables here, in
    # the main script thread, and let the factory close over those objects
    # directly instead of doing session_state lookups at call time.
    _stats_queue = st.session_state.stats_queue
    _api_url = api_url
    _frame_step = frame_step

    def make_processor():
        return DetectionVideoProcessor(
            api_url=_api_url,
            frame_step=_frame_step,
            stats_queue=_stats_queue,
        )

    webrtc_ctx = webrtc_streamer(
        key="realtime-emotion-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=make_processor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()

    if webrtc_ctx.state.playing:

        # Drain any new stats that arrived from the background detection
        # thread since the last Streamlit rerun.
        drained_error = None
        new_results = 0

        while True:
            try:
                item = st.session_state.stats_queue.get_nowait()
            except queue.Empty:
                break

            if "error" in item:
                drained_error = item["error"]
                continue

            new_results += 1
            st.session_state.last_fps_ts.append(item["timestamp"])
            for detection in item["detections"]:
                st.session_state.class_counts[detection["class"]] += 1

        # Keep only the last 30s of timestamps for a rolling FPS estimate.
        cutoff = time.time() - 30
        st.session_state.last_fps_ts = [
            t for t in st.session_state.last_fps_ts if t >= cutoff
        ]

        if drained_error:
            status_placeholder.error(f"Detection error: {drained_error}")
        else:
            status_placeholder.success("🟢 Streaming live")

        detection_fps = (
            len(st.session_state.last_fps_ts) / 30
            if st.session_state.last_fps_ts
            else 0
        )

        col1, col2 = metrics_placeholder.columns(2)
        col1.metric("Total Detections (session)", sum(st.session_state.class_counts.values()))
        col2.metric("Detection Calls / sec (rolling avg)", f"{detection_fps:.2f}")

        if st.session_state.class_counts:
            chart_placeholder.bar_chart(
                pd.Series(st.session_state.class_counts).sort_values(ascending=False)
            )

        if st.button("Reset session stats"):
            st.session_state.class_counts = Counter()
            st.session_state.last_fps_ts = []
            st.rerun()

    else:
        status_placeholder.info("Click **Start** above to begin the live webcam feed.")

    