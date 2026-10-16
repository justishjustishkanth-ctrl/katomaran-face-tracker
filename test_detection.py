import json
import cv2
from src.video_source import open_video
from src.detector import FaceDetector


def main():
    with open("config.json", encoding="utf-8-sig") as file:
        config = json.load(file)

    detector = FaceDetector(
        config["face_model_path"],
        confidence=config["face_detector_confidence"],
    )

    capture = open_video(config["video_source"])
    frame_number = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                print("End of video.")
                break

            frame_number += 1
            faces = detector.detect(frame)

            for x1, y1, x2, y2, score in faces:
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"Face {score:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            if frame_number == 1 or frame_number % 30 == 0:
                print(f"Frame {frame_number}: {len(faces)} faces detected")

            cv2.imshow("Face Detection Test - Press Q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
