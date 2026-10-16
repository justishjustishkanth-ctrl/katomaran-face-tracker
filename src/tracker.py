from dataclasses import dataclass
from typing import Optional


def calculate_iou(box_a, box_b):
    """Calculate intersection-over-union for two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: tuple
    missed_frames: int = 0
    face_id: Optional[int] = None
    embedding: object = None
    entry_recorded: bool = False


class Tracker:
    """Associate face detections across frames using bounding-box overlap."""

    def __init__(self, max_lost_frames=30, iou_threshold=0.3):
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_track_id = 1

    def update(self, detections):
        """
        Update tracks from detections.

        detections: iterable of (x1, y1, x2, y2, confidence)
        Returns: active tracks matched or created in this frame.
        """
        detections = list(detections)
        unmatched_tracks = set(self.tracks)
        matched = []

        # Greedily match the highest-overlap detection/track pairs.
        candidates = []
        for index, detection in enumerate(detections):
            box = tuple(map(int, detection[:4]))
            for track_id in unmatched_tracks:
                track = self.tracks[track_id]
                score = calculate_iou(box, track.bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, index, track_id, box))

        candidates.sort(reverse=True, key=lambda item: item[0])
        used_detections = set()

        for score, index, track_id, box in candidates:
            if index in used_detections or track_id not in unmatched_tracks:
                continue

            track = self.tracks[track_id]
            track.bbox = box
            track.missed_frames = 0

            used_detections.add(index)
            unmatched_tracks.remove(track_id)
            matched.append(track)

        # Create a new track for each unmatched detection.
        for index, detection in enumerate(detections):
            if index in used_detections:
                continue

            box = tuple(map(int, detection[:4]))
            track = Track(
                track_id=self.next_track_id,
                bbox=box,
            )
            self.tracks[track.track_id] = track
            self.next_track_id += 1
            matched.append(track)

        # Retain unmatched tracks temporarily to tolerate missed detections.
        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            track.missed_frames += 1

            if track.missed_frames > self.max_lost_frames:
                del self.tracks[track_id]

        return matched

    def get_active_tracks(self):
        """Return tracks not yet expired."""
        return list(self.tracks.values())
