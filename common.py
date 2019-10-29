import time

import cv2
from imutils.video import VideoStream

DATABASE_LOC = "./dataset/faces.pickle"


def display_frame(frame):
    """Displays the frame to the user."""
    cv2.imshow("Frame", frame)


def start_video_stream(camera: int):
    """Starts the video stream and returns the created stream.
    Also waits for the video stream to open before returning it."""
    video_stream = VideoStream(src=camera).start()
    time.sleep(2.0)
    return video_stream
