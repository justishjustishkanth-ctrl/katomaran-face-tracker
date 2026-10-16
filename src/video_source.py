import time
import cv2


def open_video(source, timeout_ms=10000):
    """Open a local file, camera index, or authorized RTSP stream."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if isinstance(source, str) and source.lower().startswith("rtsp://"):
        capture = cv2.VideoCapture()
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
        ]
        try:
            opened = capture.open(source, cv2.CAP_FFMPEG, params)
        except (TypeError, cv2.error):
            # Some OpenCV builds don't support this parameter overload.
            opened = capture.open(source, cv2.CAP_FFMPEG)

        if not opened:
            capture.release()
            raise RuntimeError("Could not open RTSP stream.")
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video source: {source}")

    return capture


def read_with_reconnect(source, capture, retries=3, delay_seconds=2):
    """
    Read one frame. If reading fails, try reopening the source.
    Returns (capture, frame), or (capture, None) if reconnection fails.
    """
    success, frame = capture.read()
    if success and frame is not None:
        return capture, frame

    capture.release()

    for attempt in range(1, retries + 1):
        print(f"Video read failed; reconnect attempt {attempt}/{retries}")
        time.sleep(delay_seconds)
        try:
            capture = open_video(source)
            success, frame = capture.read()
            if success and frame is not None:
                print("Video source reconnected.")
                return capture, frame
            capture.release()
        except (RuntimeError, cv2.error) as exc:
            print(f"Reconnect failed: {exc}")

    return None, None


def process_video(source):
    capture = open_video(source)
    frame_number = 0

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break

            frame_number += 1
            print(f"Frame {frame_number}: {frame.shape[1]}x{frame.shape[0]}")
            cv2.imshow("Video Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import json

    with open("config.json", "r", encoding="utf-8-sig") as file:
        config = json.load(file)

    process_video(config["video_source"])
