# Intelligent Face Tracker with Auto-Registration

## Objective

Build a computer-vision application that detects faces in video, tracks them
across frames, generates face embeddings, registers identities automatically,
records entry and exit events, and counts registered identities.

The current prototype has been run against the supplied sample video.
Live RTSP operation and real-world visitor-count accuracy require further
validation.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and integration |
| OpenCV | Video input, frame processing, and image saving |
| YOLO face detector | Face detection using trained face-detection weights |
| InsightFace / ArcFace embeddings | Face representation for identity matching |
| ByteTrack | Associating detections across video frames |
| SQLite | Persistent identity and event storage |
| JSON | Application configuration |
| Python logging | Operational and event logs |
| pytest | Automated testing |

## Features

- Face detection using `models/yolov8n-face.pt`
- Face embedding generation using InsightFace `buffalo_l`
- Embedding-based identity matching and automatic registration
- Tracking with YOLO's ByteTrack integration
- Entry and exit event recording
- Timestamped entry and exit image crops
- SQLite persistence
- Operational logging in `logs/events.log`
- Configurable video source, model paths, recognition threshold,
  and tracking parameters
- Preliminary support for opening RTSP sources through OpenCV

## Project Structure

```text
katomaran-face-tracker/
├── main.py
├── config.json
├── requirements.txt
├── requirements-lock.txt
├── data/
│   ├── sample_video.mp4
│   └── visitors.db
├── models/
│   ├── yolov8n-face.pt
│   └── model_info.txt
├── src/
│   ├── database.py
│   ├── detector.py
│   ├── event_manager.py
│   ├── identity_manager.py
│   ├── logger_setup.py
│   ├── main.py
│   ├── recognizer.py
│   ├── tracker.py
│   └── video_source.py
├── logs/
│   ├── events.log
│   ├── entries/
│   └── exits/
├── docs/
│   ├── AI_PLANNING.md
│   └── architecture.mmd
├── sample_output/
├── outputs/
└── tests/
    ├── test_events.py
    └── test_main.py
```

## Requirements

- Python. Development and execution have been observed on Python 3.14.0.
- Dependencies listed in `requirements.txt`.
- Face detector weights at `models/yolov8n-face.pt`.
- InsightFace model files for `buffalo_l`.
- Sufficient CPU and RAM for the selected models.
- OpenCV video support compatible with the chosen input source.

Python compatibility with other versions has not been independently verified.
Use the project lock file when recreating the tested environment.

## Setup

Open PowerShell in the project root.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use the Python executable
directly at `.venv\Scripts\python.exe` or follow your organization's
PowerShell execution policy.

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Check the model files

Confirm that these files exist:

- `models/yolov8n-face.pt`
- `models/model_info.txt`

The application uses InsightFace `buffalo_l`. Its model files must be
available locally or downloaded by the supported InsightFace initialization
process. The first initialization may take longer.

Review the upstream model repository's license and redistribution terms
before distributing model weights.

### 4. Configure the application

Edit `config.json` for your environment.

### 5. Run the application

```powershell
python main.py
```

To run automated tests:

```powershell
pytest -v
```

## Configuration

The current configuration uses these settings:

| Key | Purpose |
|---|---|
| `video_source` | Local video path, camera index, or authorized RTSP URL |
| `face_model_path` | Face-trained YOLO weights |
| `database_path` | SQLite database file |
| `detection_skip_frames` | Configured recognition scheduling interval |
| `track_max_lost_frames` | Grace period before a missing track is expired |
| `recognition_threshold` | Similarity threshold for identity matching |
| `face_detector_confidence` | Minimum detector confidence |
| `log_file` | Operational log path |
| `entry_image_dir` | Entry snapshot directory |
| `exit_image_dir` | Exit snapshot directory |
| `show_video` | Optional video preview; set to `true` to display it |

Example local-video configuration:

```json
{
  "video_source": "data/sample_video.mp4",
  "face_model_path": "models/yolov8n-face.pt",
  "database_path": "data/visitors.db",
  "detection_skip_frames": 2,
  "track_max_lost_frames": 30,
  "recognition_threshold": 0.45,
  "face_detector_confidence": 0.5,
  "log_file": "logs/events.log",
  "entry_image_dir": "logs/entries",
  "exit_image_dir": "logs/exits",
  "show_video": true
}
```

The recognition threshold is provisional and has not been calibrated against
a labeled dataset. The detection-skip setting should be validated against the
actual main-loop implementation before claiming a particular performance
benefit.

### RTSP input

For an authorized camera, configure `video_source` with the camera's RTSP URL:

```json
"video_source": "rtsp://CAMERA_HOST:554/STREAM_PATH"
```

Replace the placeholders with the actual URL supplied by the camera
administrator. Never commit passwords, tokens, or other camera credentials
to a public repository.

RTSP connectivity and complete application-level reconnection behavior
remain unverified until tested with a real stream.

## Architecture

See [`docs/architecture.mmd`](docs/architecture.mmd) for the Mermaid
architecture diagram and [`docs/AI_PLANNING.md`](docs/AI_PLANNING.md)
for technology choices, assumptions, testing, and limitations.

The main processing path is:

1. Read a frame using OpenCV.
2. Detect faces and obtain tracking IDs using YOLO and ByteTrack.
3. Extract a face crop and generate an InsightFace embedding.
4. Compare the embedding with registered identities.
5. Reuse a matching identity or register a new one.
6. Record entry and exit events.
7. Save timestamped images and persist event information in SQLite.
8. Write operational messages to the log file.

Tracking IDs represent tracks, not permanent identities. One person can be
associated with multiple tracks if tracking is lost and the person is
reacquired.

## Database

The application uses SQLite at `data/visitors.db`.

### `faces`

Stores registered identities and their first-seen timestamp and embedding.

### `events`

Stores event records associated with an identity and tracking ID, including
event type, timestamp, and image path.

The current event types are `entry` and `exit`. Event consistency checks
have been run against the generated database. A consistent event history
does not by itself prove correct identity matching or real-world visitor
counts.

## Logs and Sample Output

Operational logs are written to:

```text
logs/events.log
```

Inspect the most recent messages:

```powershell
Get-Content logs/events.log -Tail 50
```

Entry and exit image crops are stored under date-based directories within:

```text
logs/entries/
logs/exits/
```

The sample video has previously been processed for 240 frames. Database
inspections observed 31 registered identities in one run and 123 entry plus
123 exit events in a later database inspection. These observations may
include records from multiple application runs and are not ground-truth
counts of distinct people.

Before submission, include representative, non-sensitive sample images,
sanitized logs, and database query results in `sample_output/`. Do not include
camera credentials, private video, or unnecessary personal data.

## Testing and Validation

Observed checks include:

- Python syntax compilation completed without errors.
- The face detector produced detections on the sample video.
- InsightFace produced a finite 512-dimensional embedding in a sample test.
- The integrated application processed 240 frames.
- A database check reported 123 entry and 123 exit events.
- The event consistency query reported no tracks with missing or duplicate
  entry/exit events in that inspected database.
- The image validation query reported zero missing event images.

These are observed results, not a claim that all automated tests pass or that
recognition is accurate. Run `pytest -v` and record its actual output before
submission.

### Manual scenarios still requiring validation

- Same person remains visible without repeated entry events.
- A person returns after a tracking interruption and retains the same identity.
- Different people are distinguished correctly.
- Track IDs are handled safely after tracker resets or reconnections.
- Video end and early termination finalize events exactly once.
- Interrupted processing can resume without inconsistent event state.
- Live RTSP input works and reconnects safely.

## Known Limitations

- Live RTSP operation has not been verified against a real camera.
- Reconnection support in `src/video_source.py` is not yet verified as part
  of the complete application pipeline.
- Identity matching can produce false matches or duplicate identities.
- Track IDs can change when a face disappears and is reacquired.
- Entry and exit events are inferred from tracking lifecycle rather than
  physical crossing of a defined doorway.
- The recognition threshold has not been calibrated on a labeled dataset.
- Crash recovery and durable track lifecycle recovery are not verified.
- CPU/GPU utilization, peak RAM usage, and average FPS have not yet been
  measured.
- Registered identity totals should not be presented as verified distinct-
  visitor counts without manual ground-truth evaluation.

## Compute Load

Record measurements from the actual machine used for the final demonstration.

| Metric | Result |
|---|---|
| Operating system | Windows |
| Python version | 3.14.0 |
| Face detector | YOLOv8n face weights |
| Face recognition model | InsightFace `buffalo_l` |
| Execution provider | CPUExecutionProvider |
| Video resolution | Not measured here |
| CPU model and utilization | Not measured |
| Installed RAM and peak usage | Not measured |
| GPU model and utilization | Not measured |
| Average processing FPS | Not measured |

Do not substitute estimated values for actual measurements.

## Demonstration

**Project Demo:** [Watch the Katomaran Face Tracker Demo](https://youtu.be/njDCHThPNoM)

The demonstration video is hosted on YouTube. The application uses YOLO face detection, InsightFace embeddings, ByteTrack tracking, and SQLite event logging.

**Demo link:** Pending recording.

The demonstration should show application startup, face detection, identity
registration, event logs, saved entry/exit images, and the database results.
Clearly disclose that live RTSP behavior is unverified if no real stream
has been tested.

## AI Planning

See [`docs/AI_PLANNING.md`](docs/AI_PLANNING.md) for the development workflow,
technology decisions, assumptions, test results, and known limitations.

## Architecture Diagram

See [`docs/architecture.mmd`](docs/architecture.mmd).

This project is a part of a hackathon run by https://katomaran.com
