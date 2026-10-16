from datetime import datetime, timezone

import numpy as np


def utc_now():
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def cosine_similarity(first, second):
    """Calculate cosine similarity for two embedding vectors."""
    first = np.asarray(first, dtype=np.float32).reshape(-1)
    second = np.asarray(second, dtype=np.float32).reshape(-1)

    if first.shape != second.shape or first.size == 0:
        return -1.0

    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        return -1.0

    denominator = np.linalg.norm(first) * np.linalg.norm(second)

    if denominator == 0:
        return -1.0

    return float(np.dot(first, second) / denominator)


class IdentityManager:
    """Match embeddings or register new identities."""

    def __init__(self, database, threshold=0.45):
        self.database = database
        self.threshold = threshold

    def identify(self, embedding):
        """Return (face_id, is_new_identity)."""
        if embedding is None:
            return None, False

        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

        if (
            embedding.size == 0
            or not np.all(np.isfinite(embedding))
            or np.linalg.norm(embedding) == 0
        ):
            return None, False

        embedding = embedding / np.linalg.norm(embedding)

        best_id = None
        best_score = -1.0

        for face_id, embedding_bytes in self.database.get_faces():
            known_embedding = np.frombuffer(
                embedding_bytes, dtype=np.float32
            )
            score = cosine_similarity(embedding, known_embedding)

            if score > best_score:
                best_score = score
                best_id = face_id

        if best_id is not None and best_score >= self.threshold:
            return best_id, False

        new_id = self.database.add_face(
            utc_now(),
            embedding.tobytes(),
        )
        return new_id, True
