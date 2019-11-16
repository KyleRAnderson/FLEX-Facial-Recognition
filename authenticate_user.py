"""
Methods for authenticating a user.
"""

import time

import cv2
import face_recognition
import imutils

import common
import data_handler

# How long to wait before timing out and saying failed authentication.
TIMEOUT: float = 30.0
# Minimum number of frames in which a user must be recognized in order to be authenticated.
MIN_USER_RECOGNITION_COUNT = 10
USER_IDS_KEY: str = "user_ids"
DEFAULT_CAMERA: int = 0
DEFAULT_WIDTH = 375


def load_encodings(file_location: str):
    """Loads the encodings for faces from the given file location."""
    return data_handler.load_database(file_location)


def determine_identity(face_encoding, known_faces) -> str:
    """
    "Determines the most likely identity of a single face. Returns the user id.
    :param face_encoding: The encoding which needs identification.
    :param known_faces: The database of known faces to use for searching.
    :return: The string user_id of the recognized user.
    """
    recognized_users = {}
    for (user_id, user_encodings) in known_faces.items():
        matches = face_recognition.compare_faces(user_encodings, face_encoding)
        count = matches.count(True)
        if count > 0:
            # Count the number of occurrences of true.
            recognized_users[user_id] = count

    matched_user = ""
    if len(recognized_users) > 0:
        matched_user: str = max(recognized_users,
                                key=recognized_users.get)
    return matched_user


def check_recognized_users(recognized_user_counts):
    """Determines if there are recognized users in the dictionary,
    and if so returns the list of their IDs"""
    recognized_users = []
    for user_id, count in recognized_user_counts.items():
        if count >= MIN_USER_RECOGNITION_COUNT:
            recognized_users.append(user_id)
    return recognized_users


def draw_rectangles_and_user_ids(image_frame, conversion: float, box_user_id_map: dict):
    """Draws the rectangles and user_ids onto the video stream so anyone viewing the stream could see them."""
    if box_user_id_map and len(box_user_id_map) > 0:
        for ((top, right, bottom, left), user_id) in box_user_id_map.items():
            top = round(top * conversion)
            right = round(right * conversion)
            bottom = round(bottom * conversion)
            left = round(left * conversion)

            # Draw the rectangle onto the face we've identified
            cv2.rectangle(image_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Find the top so we can put the text there.
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image_frame, user_id, (left, y), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 255, 0), 2)
    common.display_frame(image_frame)


def run_face_recognition(frame, known_faces: dict, encoding_model: str = "hog", draw_rectangles: bool = False,
                         image_width: int = DEFAULT_WIDTH) -> list:
    """

    :param frame: The frame to use in recognition.
    :param known_faces: The known faces data.
    :param encoding_model: The type of encoding model to use. CNN is more reliable but much slower, HOG is faster.
    :param draw_rectangles: True to draw the rectangles and user ids around identified faces, false otherwise.
    :param image_width: The width to which images should be resized in order to speed up processing.
    Lower quality means faster but less reliable recognition.
    :return: The list of recognized user ids.
    """
    recognized_user_ids: list = []
    original_frame = frame

    # Convert input from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize image to speed up processing.
    rgb_image = imutils.resize(frame, width=image_width)
    r = frame.shape[1] / float(rgb_image.shape[1])

    # Detect the location of each face and determine the boxes in which they lie
    boxes = face_recognition.face_locations(
        rgb_image, model=encoding_model)
    # Compute the facial embeddings (the encoding) at
    # each of the locations found in the previous line.
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    box_user_id_mapping = {}
    for (i, encoding) in enumerate(encodings):
        user_id: str = determine_identity(encoding, known_faces)
        if user_id:
            box_user_id_mapping[boxes[i]] = user_id
            recognized_user_ids.append(user_id)

    if draw_rectangles:
        draw_rectangles_and_user_ids(original_frame, r, box_user_id_mapping)

    return recognized_user_ids


def recognize_user(known_faces: dict, encoding_model: str = "hog", image_flip: int = None,
                   draw_rectangles: bool = False, camera: int = DEFAULT_CAMERA, image_width: int = DEFAULT_WIDTH):
    """Attempts to recognize a user.
    Returns the ID of the user if identified, or None if no users are identified."""
    recognized_users_count = {}
    recognized_user = None
    video_stream = common.start_video_stream(camera)

    # Determine the time at which we will time out. Equal to current time + timeout.
    timeout_time: float = time.time() + TIMEOUT
    while time.time() < timeout_time:
        # Read a image_frame from the video stream.
        ret, image_frame = video_stream.read()
        if image_flip is not None:
            image_frame = cv2.flip(image_frame, image_flip)

        recognized_user_ids = run_face_recognition(image_frame, known_faces, encoding_model=encoding_model,
                                                   draw_rectangles=draw_rectangles, image_width=image_width)

        for user_id in recognized_user_ids:
            if user_id not in recognized_users_count:
                recognized_users_count[user_id] = 0
            recognized_users_count[user_id] += 1

        # Now check if we have already positively identified a user enough times
        recognized_users = check_recognized_users(recognized_users_count)
        if len(recognized_users) > 0:
            break
        cv2.waitKey(20)  # Required or else video stream doesn't really render.

    if recognized_users_count:
        recognized_user = max(recognized_users_count,
                              key=recognized_users_count.get)
        if recognized_users_count[recognized_user] < MIN_USER_RECOGNITION_COUNT:
            recognized_user = None
    return recognized_user


def recognize_user_from_database(database_loc: str = common.DATABASE_LOC, encoding_model: str = "hog",
                                 image_flip: int = None, draw_rectangles: bool = False, camera: int = DEFAULT_CAMERA,
                                 image_width: int = DEFAULT_WIDTH):
    """
    Recognizes a user
    :param database_loc: The database containing the face encodings for users.
    :param encoding_model: The encoding model to be used for recognition.
    :param image_flip: The type of image flip to be applied to the image, if it will be upside-down or horizontally inverted.
    :param draw_rectangles: True to draw the rectangles to the screen, false otherwise.
    :param camera: Which camera to use
    :param image_width: The width to which images should be resized in order to speed up processing.
    Lower quality means faster but less reliable recognition.
    :return: The recognized user's id, or None if no user was recognized.
    """
    return recognize_user(data_handler.load_database(database_loc), encoding_model=encoding_model,
                          image_flip=image_flip,
                          draw_rectangles=draw_rectangles, camera=camera, image_width=image_width)


# If this program is the main program, authenticate the user.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Facial Identification Options")
    parser.add_argument("--encodings", "-e", type=str, help="File location of facial encodings.", required=False,
                        default=None)
    parser.add_argument("--model", "-m", type=str, help="Type of encoding method, either \"hog\" or \"cnn\". HOG is "
                                                        "faster, CNN is more accurate.", required=False,
                        default=None, choices=["cnn", "hog"])
    parser.add_argument("--flip", "-f", type=int,
                        help="Whether or not to flip the image vertically or horizontally. 0 to flip horizontally, "
                             "1 to flip vertically.",
                        required=False, default=None, choices=[0, 1])
    parser.add_argument("--show", "-s", action="store_true",
                        help="Include this argument to have the image shown to you.", default=False)
    parser.add_argument("--camera", "-c", type=int, required=False,
                        help="Which camera to be used during authentication.", default=DEFAULT_CAMERA)
    parser.add_argument("--width", "-w", type=int, required=False,
                        help=f"The image width to use, in pixels. Default is {DEFAULT_WIDTH} pixels.")
    args = parser.parse_args()

    args_dict = {}
    if args.encodings is not None:
        args_dict["database_loc"] = args.encodings
    if args.model is not None:
        args_dict["encoding_model"] = args.model

    user = recognize_user_from_database(**args_dict, image_flip=args.flip,
                                        draw_rectangles=args.show, camera=args.camera, image_width=args.width)
    if user:
        print(f"Recognized user {user}.")
