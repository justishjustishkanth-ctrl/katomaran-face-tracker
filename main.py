import json
import logging
import uuid
from pathlib import Path

import cv2

from src.database import Database
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.identity_manager import IdentityManager
from src.event_manager import EventManager
from src.logger_setup import setup_logger
import time


def load_config():
    with open("config.json", "r", encoding="utf-8-sig") as file:
        return json.load(file)


def main():
    # PERFORMANCE_TIMING_ENABLED
    startup_start = time.perf_counter()
    config = load_config()
    logger = setup_logger(config["log_file"])
    database = Database(config["database_path"])

    detector = FaceDetector(
        config["face_model_path"],
        confidence=config["face_detector_confidence"],
    )
    recognizer = FaceRecognizer()
    identity_manager = IdentityManager(
        database,
        threshold=config["recognition_threshold"],
    )
    event_manager = EventManager(database, logger, config)
    initialization_seconds = time.perf_counter() - startup_start
    print(f'Startup and model initialization: {initialization_seconds:.2f} seconds')
    frame_times = []

    # A different prefix on each run reduces track_id collisions in SQLite.
    session_prefix = int(uuid.uuid4().hex[:8], 16) * 1_000_000

    # State is keyed by (session ID, ByteTrack ID), not ByteTrack ID alone.
    tracks = {}
    frame_number = 0
    skip_frames = max(1, int(config["detection_skip_frames"]))
    max_lost = max(1, int(config["track_max_lost_frames"]))
    show_video = bool(config.get("show_video", False))

    source = config["video_source"]
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        database.close()
        raise RuntimeError(f"Could not open video source: {source}")

    logger.info("Video processing started: %s", source)

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_number += 1
            frame_start = time.perf_counter()

            # Update ByteTrack on every frame to preserve motion continuity.
            detections = detector.track(frame)
            visible_keys = set()

            for track_id, x1, y1, x2, y2, confidence in detections:
                key = ("current-session", int(track_id))
                visible_keys.add(key)

                state = tracks.get(key)
                if state is None:
                    state = {
                        "db_track_id": session_prefix + int(track_id),
                        "face_id": None,
                        "entry_recorded": False,
                        "last_seen": frame_number,
                        "last_frame": frame.copy(),
                        "last_bbox": (x1, y1, x2, y2),
                    }
                    tracks[key] = state

                state["last_seen"] = frame_number
                state["last_frame"] = frame.copy()
                state["last_bbox"] = (x1, y1, x2, y2)

                # Retry recognition until an identity is assigned, then keep it
                # for this track. Tracking itself continues on every frame.
                should_recognize = (
                    frame_number % skip_frames == 0
                    or state["face_id"] is None
                )

                if should_recognize and state["face_id"] is None:
                    height, width = frame.shape[:2]
                    left = max(0, min(width, x1))
                    top = max(0, min(height, y1))
                    right = max(0, min(width, x2))
                    bottom = max(0, min(height, y2))

                    if right > left and bottom > top:
                        face_crop = frame[top:bottom, left:right]
                        embedding = recognizer.get_embedding(face_crop)

                        if embedding is not None:
                            face_id, is_new = identity_manager.identify(
                                embedding
                            )
                            if face_id is not None:
                                state["face_id"] = face_id
                                logger.info(
                                    "Identity assigned: face_id=%s "
                                    "track_id=%s new=%s",
                                    face_id,
                                    state["db_track_id"],
                                    is_new,
                                )

                # Record exactly one entry per database track in this run.
                if (
                    state["face_id"] is not None
                    and not state["entry_recorded"]
                ):
                    result = event_manager.record(
                        face_id=state["face_id"],
                        track_id=state["db_track_id"],
                        event_type="entry",
                        frame=state["last_frame"],
                        bbox=state["last_bbox"],
                    )
                    if result:
                        state["entry_recorded"] = True

                if show_video:
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    label = (
                        f"Visitor {state['face_id']}"
                        if state["face_id"] is not None
                        else f"Track {track_id}"
                    )
                    cv2.putText(
                        frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

            # A track is considered lost only after the configured grace
            # period. Use its last valid frame and bounding box for the exit.
            lost_keys = [
                key for key, state in tracks.items()
                if frame_number - state["last_seen"] > max_lost
            ]

            for key in lost_keys:
                state = tracks.pop(key)
                if state["face_id"] is not None and state["entry_recorded"]:
                    event_manager.record(
                        face_id=state["face_id"],
                        track_id=state["db_track_id"],
                        event_type="exit",
                        frame=state["last_frame"],
                        bbox=state["last_bbox"],
                    )

            if show_video:
                cv2.putText(
                    frame,
                    f"Unique visitors: {database.unique_visitor_count()}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Katomaran Face Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Video processing stopped by user")
                    break

            frame_times.append(time.perf_counter() - frame_start)

            if frame_number % 100 == 0:
                logger.info(
                    "Processed %s frames; unique visitors=%s",
                    frame_number,
                    database.unique_visitor_count(),
                )

    finally:
        # Close active, recognized tracks using their last valid image.
        for state in tracks.values():
            if state["face_id"] is not None and state["entry_recorded"]:
                try:
                    event_manager.record(
                        face_id=state["face_id"],
                        track_id=state["db_track_id"],
                        event_type="exit",
                        frame=state["last_frame"],
                        bbox=state["last_bbox"],
                    )
                except Exception:
                    logger.exception("Failed to record final exit event")

        capture.release()
        if show_video:
            cv2.destroyAllWindows()
        database.close()

    if frame_times:
        average_seconds = sum(frame_times) / len(frame_times)
        print(f"Frames timed: {len(frame_times)}")
        print(f"Average processing time: {average_seconds * 1000:.2f} ms/frame")
        print(f"Approximate processing FPS: {1 / average_seconds:.2f}" if average_seconds > 0 else "Approximate processing FPS: unavailable")
    print(f"Frames processed: {frame_number}")
    print("Video processing finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Face tracker failed")
        raise
