import cv2
import sys

print(f"OpenCV Version: {cv2.__version__}")
try:
    tracker = cv2.TrackerCSRT_create()
    print("SUCCESS: TrackerCSRT_create() is available.")
except AttributeError:
    print("ERROR: TrackerCSRT_create() not found.")
    # Attempt legacy compat
    try:
        tracker = cv2.legacy.TrackerCSRT_create()
        print("SUCCESS: found as cv2.legacy.TrackerCSRT_create()")
    except AttributeError:
        print("FATAL: TrackerCSRT not found in any standard location.")
except Exception as e:
    print(f"ERROR: {e}")
