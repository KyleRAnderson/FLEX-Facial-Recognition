import cv2

DATABASE_LOC = "./dataset/faces.pickle"


def display_frame(frame):
    """Displays the frame to the user."""
    cv2.imshow("Frame", frame)


def start_video_stream(camera: int):
    """Starts the video stream and returns the created stream.
    Also waits for the video stream to open before returning it."""
    video_stream = cv2.VideoCapture(0)
    return video_stream
