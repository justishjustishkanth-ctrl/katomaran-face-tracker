# AI Planning Document

## 1. Problem

Build an intelligent face tracker that detects faces in video, associates
detections across frames, generates face embeddings, registers identities,
and records visitor entry and exit events.

The current prototype processes a supplied video file. Live RTSP operation
requires validation against an actual authorized camera stream.

## 2. Functional Requirements

- Detect faces using a face-trained YOLO model.
- Generate face embeddings using InsightFace with the buffalo_l model.
- Match embeddings against previously registered identities.
- Track faces across frames using YOLO tracking with ByteTrack.
- Record entry and exit events for recognized tracks.
- Save timestamped face crops for entry and exit events.
- Store identities and event records in SQLite.
- Write operational messages to logs/events.log.
- Count registered identities separately from tracking events.
- Configure the detection skip interval and tracking grace period.
- Accept local video files, with RTSP support prepared in the video-source module.
- Validate event consistency and saved image paths.

## 3. Architecture

Video input
    -> YOLO face detection and tracking
    -> Face crop extraction
    -> InsightFace embedding generation
    -> Identity matching and registration
    -> Entry/exit event management
    -> SQLite database, timestamped images, and operational logs

### Module responsibilities

- main.py: coordinates the application and video-processing loop.
- src/detector.py: loads the face-trained YOLO model and returns tracked faces.
- src/recognizer.py: generates normalized face embeddings using InsightFace.
- src/identity_manager.py: compares embeddings and registers identities.
- src/database.py: persists identities and event records in SQLite.
- src/event_manager.py: records events and saves face crops.
- src/logger_setup.py: configures application logging.
- src/video_source.py: opens local video, camera-index, or RTSP sources.
- config.json: stores configurable paths and processing parameters.

## 4. Technology Choices

### Python
Python supports rapid development and has libraries for computer vision,
machine learning, database access, and automated testing.

### OpenCV
OpenCV reads video frames, extracts image crops, draws bounding boxes,
and saves event images. VideoCapture can open local files and supported
camera streams.

### YOLO face detector
The project uses models/yolov8n-face.pt, a face-trained YOLO model,
rather than relying on a general object detector to identify faces.
The model is lightweight relative to larger detector variants.

Model source and license information are recorded in models/model_info.txt.
Review the upstream license and redistribution terms before submission.

### InsightFace and ArcFace embeddings
InsightFace's buffalo_l model generates face embeddings. The application
uses normalized embeddings to compare a newly observed face against
stored identities using cosine similarity.

The configured recognition threshold is 0.45. This is a provisional
parameter, not a validated accuracy guarantee. It must be evaluated on
representative video before drawing conclusions about identity accuracy.

### ByteTrack
The YOLO tracking API uses ByteTrack to associate face detections across
frames. Tracking IDs represent tracks, not permanent human identities.
An identity can be associated with multiple tracking IDs if tracking is lost
and the person is reacquired.

### SQLite
SQLite stores face identities and event history in a local database file.
It requires no separate database server and is convenient for a standalone
hackathon prototype.

### JSON configuration
config.json separates paths and processing parameters from the Python code.
This makes the sample video, model path, database path, and thresholds easier
to change without editing the processing loop.

### Python logging
The logging module writes operational messages and event information to
logs/events.log. These logs support debugging and evaluation.

### pytest
pytest runs automated tests for database behavior and event consistency.
Passing database tests does not establish face recognition accuracy.

## 5. Assumptions

- The input video is readable by the installed OpenCV build.
- The supplied sample video is available at data/sample_video.mp4.
- The face model exists at models/yolov8n-face.pt.
- The buffalo_l InsightFace model is installed and can run with the
  configured CPU execution provider.
- Face crops must contain sufficient facial detail for embedding generation.
- Poor lighting, occlusion, small faces, and unusual viewing angles can
  reduce detection and recognition quality.
- The current recognition threshold is 0.45 and has not been calibrated
  against a labeled evaluation dataset.
- Track disappearance after the configured grace period is treated as an
  exit. This is a software rule, not proof that a person physically left
  the monitored area.
- A person can have multiple track IDs during one video.
- A registered identity count is not automatically equivalent to a
  ground-truth count of distinct people.
- Live RTSP camera behavior depends on camera configuration, network
  connectivity, and OpenCV/FFmpeg support.

## 6. AI Development Workflow

The project was developed incrementally with AI assistance and manual
execution in PowerShell.

### Recorded development steps

1. Created the Python project structure and configuration file.
2. Installed OpenCV, Ultralytics, InsightFace, ONNX Runtime, NumPy,
   pytest, and the tracking dependency used by the installed stack.
3. Obtained the face-trained YOLO model and recorded its source information.
4. Implemented video input, detection, embedding generation, identity
   management, SQLite storage, event recording, and logging modules.
5. Tested imports and Python syntax.
6. Tested face detection and embedding generation on a sample frame.
7. Tested ByteTrack on successive frames.
8. Ran the integrated application against the supplied sample video.
9. Inspected SQLite event totals, event consistency, and image paths.
10. Prepared automated database tests and RTSP input support.

### AI prompt and review record

The development requests included:
- Explain the project architecture and module responsibilities.
- Create a face detector using a face-trained YOLO model.
- Generate face embeddings using InsightFace.
- Implement identity registration and matching.
- Store visitor and event data in SQLite.
- Save timestamped entry and exit images.
- Add operational logging and automated tests.
- Prepare configurable video input and RTSP support.
- Review results and identify limitations before submission.

AI-generated code was tested and adjusted as errors were encountered.
The developer must review and understand the final implementation and
verify all reported results against the actual source code.

### Corrections and open work

- Adjusted database tests to use the actual add_face() and add_event()
  method names rather than assumed register_face() and save_event() names.
- Added an event consistency check and image-path validation.
- Added preliminary RTSP input handling in src/video_source.py.
- Full application-level reconnection recovery and crash recovery
  remain unverified.
- Recognition accuracy and distinct-person counts still require evaluation.

## 7. Compute Requirements

Record the actual environment used for testing.

- Operating system: Windows (PowerShell and VS Code).
- Python: 3.14.0.
- Environment: project-local virtual environment (.venv).
- Face detector: models/yolov8n-face.pt.
- Face recognition model: InsightFace buffalo_l.
- Execution provider configured: CPUExecutionProvider.
- Input: data/sample_video.mp4.
- Video resolution: not yet recorded here; measure it from the supplied video.
- CPU model: not yet recorded.
- Installed RAM: not yet recorded.
- GPU model and usage: not yet recorded.
- Measured average FPS: not yet recorded.
- Peak memory usage: not yet recorded.

Do not invent benchmark values. Measure the hardware and runtime before
completing this section.

## 8. Known Limitations

- Live RTSP operation has not been verified against an actual camera.
- Reconnection logic in the video-source module is not yet integrated
  into the complete main application.
- A reconnect requires careful track-state and event-state handling.
- ByteTrack IDs may be lost, reassigned, or reused.
- Multiple tracks can correspond to one registered identity.
- Face embeddings may produce false matches or fail to match the same
  person under different conditions.
- The recognition threshold has not been validated against labeled data.
- Entry and exit are inferred from track lifecycle, not physical doorway
  crossing or direction-of-travel analysis.
- SQLite event checks do not prove real-world visitor-count accuracy.
- A process interruption can leave an incomplete visit without a final
  exit event; durable recovery is not yet verified.
- A completed video run produces exit events for remaining recognized
  tracks, but this behavior needs further lifecycle testing.
- Detection skip configuration is not yet a validated performance-versus-
  accuracy control for the full application.
- Model inference speed and memory consumption have not been benchmarked.

## 9. Testing

### Automated testing

Database tests are maintained in tests/test_events.py. They should cover:
- Registering identities and counting them.
- Recording entry and exit events.
- Preventing duplicate events for the same track and event type.

Run the tests using:

    pytest -v

Record the actual test result at submission time. Do not claim tests passed
unless the current test run confirms it.

### Manual testing

Recorded observations from the sample-video run:
- Python syntax compilation completed without errors.
- The application processed 240 frames.
- A previous database inspection reported 31 registered identities.
- A later event inspection reported 123 entry events and 123 exit events.
- The event consistency query reported zero tracks with missing or duplicate
  entry/exit events.
- The image validation query reported zero missing event images.

These are observations from the inspected database, not a verified
ground-truth visitor count. Database totals may include multiple runs.

### Remaining manual scenarios

- One new face appears: verify one identity and one entry.
- A person remains visible: verify no repeated entry for the same track.
- A person disappears: verify one exit after the configured grace period.
- The same person returns: verify identity reuse and new visit events.
- A different person appears: verify a distinct identity when the embedding
  is sufficiently different.
- The video ends: verify active recognized tracks are finalized once.
- The application is interrupted: verify database readability and document
  incomplete-event recovery behavior.
- The RTSP stream disconnects: verify reconnection and track-state recovery
  once an authorized test stream is available.

### Result reporting

Separate automated test results, manual observations, and unverified
assumptions in the final submission. Do not present expected outcomes as
completed test results.
