
from datetime import datetime, timezone
from pathlib import Path

import cv2


def utc_now():
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class EventManager:
    """Save timestamped face crops and entry/exit metadata."""

    def __init__(self, database, logger, config):
        self.database = database
        self.logger = logger
        self.config = config

    def record(self, face_id, track_id, event_type, frame, bbox):
        """Save one entry/exit event and its cropped face image."""

        # Validate the event.
        if event_type not in ("entry", "exit"):
            raise ValueError("Event type must be entry or exit")

        if face_id is None or track_id is None:
            raise ValueError("face_id and track_id are required")

        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or invalid")

        # Validate and clamp the face bounding box.
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]

        x1 = max(0, min(int(x1), width))
        x2 = max(0, min(int(x2), width))
        y1 = max(0, min(int(y1), height))
        y2 = max(0, min(int(y2), height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid face bounding box")

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            raise ValueError("Face crop is empty")

        # Generate a UTC timestamp and dated output directory.
        timestamp = utc_now()
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        folder_key = (
            "entry_image_dir"
            if event_type == "entry"
            else "exit_image_dir"
        )

        folder = Path(self.config[folder_key]) / day
        folder.mkdir(parents=True, exist_ok=True)

        safe_timestamp = timestamp.replace(":", "-")
        image_path = folder / (
            f"face_{face_id}_track_{track_id}_{safe_timestamp}.jpg"
        )

        # Save the cropped face.
        if not cv2.imwrite(str(image_path), face_crop):
            raise RuntimeError(
                f"Could not save face image: {image_path}"
            )

        try:
            # Save event metadata to SQLite.
            event_id = self.database.add_event(
                face_id=face_id,
                track_id=track_id,
                event_type=event_type,
                timestamp=timestamp,
                image_path=str(image_path),
            )

            # The database returns None when a duplicate event is rejected.
            if event_id is None:
                image_path.unlink(missing_ok=True)

                self.logger.info(
                    "Duplicate %s event skipped: face_id=%s track_id=%s",
                    event_type,
                    face_id,
                    track_id,
                )
                return None

        except Exception:
            # Remove the image if the database operation fails.
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                self.logger.exception(
                    "Could not remove orphaned image: %s",
                    image_path,
                )

            self.logger.exception(
                "Failed to save %s event", event_type
            )
            raise

        # Log only events successfully inserted into the database.
        self.logger.info(
            "%s logged: face_id=%s track_id=%s image=%s",
            event_type.capitalize(),
            face_id,
            track_id,
            image_path,
        )

        return {
            "event_id": event_id,
            "face_id": face_id,
            "track_id": track_id,
            "event_type": event_type,
            "timestamp": timestamp,
            "image_path": str(image_path),
        }
