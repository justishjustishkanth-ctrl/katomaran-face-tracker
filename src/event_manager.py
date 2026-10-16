from datetime import datetime, timezone
from pathlib import Path

import cv2


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class EventManager:
    """Save timestamped face crops and entry/exit metadata."""

    def __init__(self, database, logger, config):
        self.database = database
        self.logger = logger
        self.config = config

    def record(self, face_id, track_id, event_type, frame, bbox):
        if event_type not in ("entry", "exit"):
            raise ValueError("Event type must be entry or exit")

        if face_id is None or track_id is None:
            raise ValueError("face_id and track_id are required")

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

        timestamp = utc_now()
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        folder_key = (
            "entry_image_dir" if event_type == "entry"
            else "exit_image_dir"
        )
        folder = Path(self.config[folder_key]) / day
        folder.mkdir(parents=True, exist_ok=True)

        safe_timestamp = timestamp.replace(":", "-")
        image_path = folder / (
            f"face_{face_id}_track_{track_id}_{safe_timestamp}.jpg"
        )

        if not cv2.imwrite(str(image_path), face_crop):
            raise RuntimeError(f"Could not save face image: {image_path}")

        try:
            self.database.add_event(
                face_id=face_id,
                track_id=track_id,
                event_type=event_type,
                timestamp=timestamp,
                image_path=str(image_path),
            )
        except Exception:
            # Avoid leaving an image without a corresponding database row.
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                self.logger.exception(
                    "Could not remove orphaned image: %s", image_path
                )
            self.logger.exception("Failed to save %s event", event_type)
            raise

        self.logger.info(
            "%s logged: face_id=%s track_id=%s image=%s",
            event_type.capitalize(),
            face_id,
            track_id,
            image_path,
        )

        return {
            "face_id": face_id,
            "track_id": track_id,
            "event_type": event_type,
            "timestamp": timestamp,
            "image_path": str(image_path),
        }
