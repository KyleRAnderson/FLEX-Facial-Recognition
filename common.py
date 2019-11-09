import os

import cv2

DATA_DIR = "data"
DATASET_DIR = "dataset"
DATABASE_LOC = os.path.join(DATA_DIR, "database.pickle")

RES_DIRECTORY = "./res"
# Directory for the face detection model.
FACE_DETECTION_MODEL_DIR = os.path.join(RES_DIRECTORY, "face_detection_model")
EMBEDDINGS_PROCESSOR_LOC = os.path.join(RES_DIRECTORY, "openface_nn4.small2.v1.t7")


def display_frame(frame):
    """Displays the frame to the user."""
    cv2.imshow("Frame", frame)


def start_video_stream(camera: int):
    """Starts the video stream and returns the created stream.
    Also waits for the video stream to open before returning it."""
    video_stream = cv2.VideoCapture(0)
    return video_stream


def load_cascade(cascade_loc: str) -> cv2.CascadeClassifier:
    """
    Opens the cascade classifier at the given path.
    :param cascade_loc: The file location of the cascade.
    :return:The CascadeClassifier class.
    """
    return cv2.CascadeClassifier(cascade_loc)


def load_detector(proto_path: str, model_path: str):
    """
    Loads the caffe detector with the given proto text file and the model file.
    :param proto_path: The path location of the prototext file.
    :param model_path: The path to the caffe model.
    :return: The detector.
    """
    return cv2.dnn.readNetFromCaffe(proto_path, model_path)


def load_embedding_model(model_path: str):
    """
    Loads the torch embedding model at the given location.
    :param model_path: The path to the model.
    :return: The embedding model
    """
    return cv2.dnn.readNetFromTorch(model_path)


CAFFE_MODEL_NAME = "res10_300x300_ssd_iter_140000.caffemodel"
PROTOTXT_NAME = "deploy.prototxt"


def load_detector_from_dir(detector_dir: str):
    prototxt: str = os.path.join(detector_dir, PROTOTXT_NAME)
    caffe_model: str = os.path.join(detector_dir, CAFFE_MODEL_NAME)
    return load_detector(prototxt, caffe_model)
