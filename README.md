# Video Emotion Detection — YOLOv8

Real-time and offline facial emotion detection. A fine-tuned YOLOv8 model runs
as a REST API on a free GPU (Google Colab), tunneled to the internet with
ngrok, and a Streamlit web app sends it video frames (either an uploaded file
or your live webcam) and displays annotated results.

- **Server**: `task3_server.ipynb` — runs in Colab, loads the trained model
  (`best.pt`), serves it via FastAPI, and exposes it publicly through ngrok.
- **Client**: `app.py` — a Streamlit app you run locally, which uploads
  frames to the server and displays bounding boxes + emotion labels.

---

## 1. Setup

### Step A — Start the detection server (Google Colab)

1. Go to [Google Colab](https://colab.research.google.com) and upload
   `task3_server.ipynb` (**File → Upload notebook**).
2. Set the runtime to use a GPU: **Runtime → Change runtime type → T4 GPU**.
3. Upload your trained weights file, **`best.pt`**, into the Colab session's
   `/content/` folder (drag it into the file browser on the left, or use the
   folder icon → upload). This is required — the server loads the model from
   `/content/best.pt`.
4. Get a free ngrok auth token from [ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken)
   (sign up, then copy the token from your dashboard).
5. In the notebook cell containing:
   ```python
   from pyngrok import ngrok
   ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")
   ```
   replace `"YOUR_NGROK_TOKEN_HERE"` with **your own** token.
   > ⚠️Never commit or share a notebook with a real token pasted in — treat
   > it like a password. If one is ever exposed, revoke it and generate a
   > new one from the ngrok dashboard.
6. Run all cells top to bottom (**Runtime → Run all**). This will:
   - install dependencies (`ultralytics`, `fastapi`, `uvicorn`, `pyngrok`, etc.)
   - write out `app.py` (the FastAPI server) via `%%writefile`
   - load `best.pt` onto the GPU
   - start the API on port `8000`
   - open an ngrok tunnel and print a public URL, e.g.:
     ```
     NgrokTunnel: "https://xxxxxxx.ngrok-free.dev" -> "http://localhost:8000"
     ```
   Copy that `https://...ngrok-free.dev` URL — you'll paste it into the
   Streamlit app in Step B.
7. Keep the Colab tab open. The tunnel and server stay alive only while the
   notebook is running; closing the tab or letting the session time out
   stops the API.

### Step B — Run the Streamlit client (your computer)

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # macOS / Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   streamlit run app.py
   ```
4. In the browser tab that opens, find **Prediction API URL** in the
   sidebar and paste the ngrok URL from Step A, followed by `/predict`,
   e.g.:
   ```
   https://xxxxxxx.ngrok-free.dev/predict
   ```
5. The sidebar shows a live **API server status** check (online / offline /
   model not loaded) so you can confirm the connection before running
   detection.

---

## 2. Using the app

The app has two modes, toggled in the sidebar:

- **Upload Video** (default) — upload an `.mp4`/`.avi`/`.mov`/`.mkv`/`.webm`
  file, click **Run Detection**, and watch frame-by-frame annotated preview,
  a live progress bar, and a summary once finished: frames processed, total
  detections, achieved FPS vs. your configured minimum FPS benchmark, an
  emotion-count bar chart, and a full per-frame detection table.
- **Live Webcam** — streams your browser's webcam through
  `streamlit-webrtc`. Every *N*th frame (configurable via **Process every
  Nth frame**) is sent to the API in a background thread so the video stays
  smooth even if a request is slow; the most recent detections are drawn
  continuously. Running totals and a rolling detections/sec estimate are
  shown live, with a **Reset session stats** button.

Sidebar settings:
| Setting | Purpose |
|---|---|
| Prediction API URL | The `/predict` endpoint of your Colab/ngrok server |
| Process every Nth frame | Trade-off between speed and thoroughness |
| Minimum FPS Benchmark | Threshold used to report pass/fail on the FPS metric |

---

## 3. Steps to close everything

### Stop the Streamlit client
- In the terminal where it's running, press **Ctrl + C**.


### Stop the Colab server
1. In `task3_server.ipynb`, run the cell under **"Kill the server"**:
   ```python
   if 'server' in locals() and server.poll() is None:
     os.system("fuser -k 8000/tcp")
   ```
   This stops the FastAPI process on port 8000.

2. Fully release the GPU by disconnecting the runtime:
   **Runtime → Disconnect and delete runtime** (or just close the Colab
   tab). This frees up your Colab GPU quota for next time.

---

## 4. Training metrics

Evaluated on the **AffectNet** face test set:

| Metric | Value |
|---|---|
| mAP50 | 0.7480 |
| mAP50-95 | 0.7456 |
| Precision | 0.6392 |
| Recall | 0.7374 |
| F1 Score | 0.6848 |

- **mAP50 / mAP50-95** measure detection quality (how well predicted boxes
  match ground truth) at IoU 0.5 and averaged across IoU 0.5–0.95,
  respectively — both are strong here, indicating well-localized face
  detections.
- **Precision (0.64)** — of all predicted emotion detections, ~64% were
  correct; the remaining ~36% were false positives.
- **Recall (0.74)** — the model correctly found ~74% of all actual emotion
  instances in the test set, missing ~26%.
- **F1 Score (0.68)** — the harmonic mean of precision and recall, giving a
  single balanced measure of classification performance.

_FPS / real-time performance benchmark results to be added._

---

## 5. Project structure

```
.
├── app.py                            # Streamlit client 
├── requirements.txt                  # Client dependencies
├── task3_server.ipynb                # Colab notebook: Serves YOLOv8 API
├── task3_model_training_code.ipynb   # Yolov8 model finetuning code
├── test.mp4                          # Video to test the code
└── README.md
```
