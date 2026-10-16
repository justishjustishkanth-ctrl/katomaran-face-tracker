import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    """Generate normalized facial embeddings using InsightFace."""

    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=(320, 320))

    def get_embedding(self, face_crop):
        """Return a normalized embedding, or None for an invalid crop."""
        if face_crop is None or face_crop.size == 0:
            return None

        detected_faces = self.app.get(face_crop)

        if not detected_faces:
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
            return None

        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
            return None

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return None

        return embedding / norm
