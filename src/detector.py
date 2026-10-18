
import logging

from ultralytics import YOLO


logger = logging.getLogger("face_tracker")


class FaceDetector:
    """Detect and track faces with a face-trained YOLO model."""

    def __init__(self, model_path, confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        logger.info("Face detector initialized: %s", model_path)

    def detect(self, frame):
        """Return boxes as (x1, y1, x2, y2, confidence)."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )

        faces = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = (
                    int(value) for value in box.xyxy[0].tolist()
                )
                score = float(box.conf[0])
                faces.append((x1, y1, x2, y2, score))

        logger.debug("Detection completed: faces=%s", len(faces))
        return faces

    def track(self, frame):
        """Return (track_id, x1, y1, x2, y2, confidence) tuples."""
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            verbose=False,
        )

        tracked_faces = []

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()

            for box, track_id, score in zip(boxes, ids, scores):
                x1, y1, x2, y2 = map(int, box)
                tracked_faces.append(
                    (track_id, x1, y1, x2, y2, score)
                )

        # DEBUG avoids writing one message for every video frame.
        logger.debug(
            "Tracking completed: active detections=%s",
            len(tracked_faces),
        )

        return tracked_faces
