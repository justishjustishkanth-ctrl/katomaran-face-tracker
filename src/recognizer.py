
import logging

import numpy as np
from insightface.app import FaceAnalysis


logger = logging.getLogger("face_tracker")


class FaceRecognizer:
    """Generate normalized facial embeddings using InsightFace."""

    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=(320, 320))
        logger.info("Face recognizer initialized: InsightFace buffalo_l")

    def get_embedding(self, face_crop):
        """Return a normalized embedding, or None for an invalid crop."""
        if face_crop is None or face_crop.size == 0:
            logger.warning("Embedding skipped: empty face crop")
            return None

        detected_faces = self.app.get(face_crop)

        if not detected_faces:
            logger.debug("Embedding unavailable: no face found in crop")
            return None

        # Select the largest face found within the crop.
        face = max(
            detected_faces,
            key=lambda item: (
                item.bbox[2] - item.bbox[0]
            ) * (
                item.bbox[3] - item.bbox[1]
            ),
        )

        embedding = face.normed_embedding

        if embedding is None:
            logger.warning("Embedding unavailable: model returned None")
            return None

        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
            logger.warning("Embedding rejected: invalid values or shape")
            return None

        norm = np.linalg.norm(embedding)

        if norm == 0:
            logger.warning("Embedding rejected: zero-length vector")
            return None

        normalized_embedding = embedding / norm
        logger.info(
            "Embedding generated successfully: dimensions=%s",
            normalized_embedding.size,
        )
        return normalized_embedding
