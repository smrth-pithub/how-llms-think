"""Find the webcam: tries camera numbers 0-4 and reports which ones work.

    check_camera.bat        (or: .venv\\Scripts\\python -m tools.check_camera)
"""
import os
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402


def main():
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    found = []
    black = []
    for i in range(5):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            continue
        t0, n = time.time(), 0
        while time.time() - t0 < 2 and n < 120:
            ok, frame = cap.read()
            n += ok
        fps = n / (time.time() - t0)
        h, w = frame.shape[:2]
        brightness = float(frame[::8, ::8].mean())
        cap.release()
        if brightness < 5:
            print(f"  camera {i}: opens, but the picture is black (a virtual camera, a closed privacy shutter,"
                  f" or the camera is in use by another app)")
            black.append(i)
            continue
        print(f"  camera {i}: works, {w}x{h}, about {fps:.0f} fps")
        found.append(i)
    if not found:
        print("  No working camera with a picture. Plug the webcam in (try another USB port),"
              " close Zoom/Teams/Camera app, and run this again.")
    else:
        print(f"\nStart the demo with:  start.bat --camera {found[-1]}"
              + ("   (the lower number is probably the laptop's own camera)" if len(found) > 1 else ""))


if __name__ == "__main__":
    main()
