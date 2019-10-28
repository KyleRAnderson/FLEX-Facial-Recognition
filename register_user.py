"""
Creates a facial recognition profile for a new user.
"""
from common import USER_IDS_KEY, start_video_stream
import face_recognition
import cv2


def process_image(image, encoding_model: str = "hog"):
    """
    Processes a single image, returning the encoded face object
    :param image: The image containing the face to be processed.
    :param encoding_model: The encoding model, either CNN or HOG
    :return: The processed facial recognition profile encoding.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert between the image formats

    # Detect the coordinates of the boxes corresponding to faces in the input image
    boxes = face_recognition.face_locations(image_rgb, model=encoding_model)
    # Actually make the encodings for the face.
    return face_recognition.face_encodings(image_rgb, boxes)


def register_user(user_id: str, encoding_model="hog", image_flip: int = None, video_source=None):
    """
    Function for registering a new user using the given video source. If video source isn't provided, then the camera
    on id 0 is used.
    :param user_id: The user id for the user that is being registered.
    :param encoding_model: The type of encoding model. Must be either "hog" or "cnn". HOG is faster, CNN is more thorough.
    :param image_flip: The integer by which this image should be flipped. 0 for reflect about x-axis, 1 for reflect on y-axis
    :return: Encoded face that was detected, or None if no face was detected or if there was another error.
    """
    if video_source is None:
        video_source = start_video_stream(0)
