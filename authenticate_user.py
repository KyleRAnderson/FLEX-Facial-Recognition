"""
Methods for authenticating a user.
"""

from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2

# How long to wait before timing out and saying failed authentication.
TIMEOUT: float = 30.0
# Minimum number of frames in which a user must be recognized in order to be authenticated.
MIN_USER_RECOGNITION_COUNT = 10
image_writer = None
USER_IDS_KEY: str = "names" # TODO change


def load_encodings(file_location: str):
    """Loads the encodings for faces from the given file location."""
    with open(file_location, "rb") as encodings_file:
        encodings = pickle.loads(encodings_file.read())
    return encodings


def start_video_stream(camera: int):
    """Starts the video stream and returns the created stream. 
    Also waits for the video stream to open before returning it."""
    video_stream = VideoStream(src=camera).start()
    time.sleep(2.0)
    return video_stream


def determine_identity(face_encoding, known_faces):
    """Determines the most likely identity of a single face. Returns the user id."""
    matches = face_recognition.compare_faces(
        known_faces["encodings"], face_encoding)
    matched_user = ''
    matched_user_id_count = {}

    # If there is at least one match to a face in the database, figure out which one it is.
    if True in matches:
        matched_users = [user_index for (
            user_index, is_match) in enumerate(matches) if is_match]

        for i in matched_users:
            user_id: str = known_faces[USER_IDS_KEY][i]
            matched_user_id_count[user_id] = matched_user_id_count.get(user_id, 0) + 1

    matched_user: str = max(matched_user_id_count,
                            key=matched_user_id_count.get)
    return matched_user


def check_recognized_users(recognized_user_counts):
    """Determines if there are recognized users in the dictionary,
    and if so returns the list of their IDs"""
    recognized_users = []
    for user_id, count in recognized_user_counts.items():
        if count >= MIN_USER_RECOGNITION_COUNT:
            recognized_users.append(user_id)
    return recognized_users


def draw_rectanges_and_user_ids(image_frame, conversion: float, boxes, user_ids: list):
    """Draws the rectangles and user_ids onto the video stream so anyone viewing the stream could see them."""
    if boxes and user_ids and len(user_ids) > 0:
        for ((top, right, bottom, left), user_id) in zip(boxes, user_ids):
            top = round(top * conversion)
            right = round(right * conversion)
            bottom = round(bottom * conversion)
            left = round(left * conversion)

            # Draw the rectangle onto the face we've identified
            cv2.rectangle(image_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Find the top so we can put the text there.
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image_frame, user_id, (left, y), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 255, 0), 2)
    display_frame(image_frame)


def display_frame(frame):
    """Displays the frame to the user."""
    cv2.imshow("Frame", frame)


def recognize_user(encodings_location: str = "./encodings.pickle", encoding_model: str = "hog", image_flip: int = None,
                   draw_rectangles=False):
    """Attempts to recognize a user.
    Returns the ID of the user if identified, or None if no users are identified.
    Dictionary of the form { "user_id": #frames_recognized } to keep
    track of how many times each user was recognized."""
    recognized_users_count = {}
    recognized_user = None
    video_stream = start_video_stream(0)
    known_faces = load_encodings(encodings_location)

    # Determine the time at which we will time out. Equal to current time + timeout.
    timeout_time: float = time.time() + TIMEOUT
    while time.time() < timeout_time:
        # Read a image_frame from the video stream.
        image_frame = video_stream.read()
        if image_flip is not None:
            image_frame = cv2.flip(image_frame, image_flip)

        # Convert input from BGR to RGB
        cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
        # Resize image to width of 750 PX to speed up processing.
        rgb_image = imutils.resize(image_frame, width=750)
        r = image_frame.shape[1] / float(rgb_image.shape[1])

        # Detect the location of each face and determine the boxes in which they lie
        boxes = face_recognition.face_locations(
            rgb_image, model=encoding_model)
        # Computer the facial embeddings (the encoding) at
        # each of the locations found in the previous line.
        encodings = face_recognition.face_encodings(rgb_image, boxes)

        for encoding in encodings:
            user_id: str = determine_identity(encoding, known_faces)
            if user_id:
                if user_id not in recognized_users_count:
                    recognized_users_count[user_id] = 0
                recognized_users_count[user_id] += 1

        if draw_rectangles:
            draw_rectanges_and_user_ids(image_frame, r, boxes, list(known_faces[USER_IDS_KEY]))

        # Now check if we have already positively identified a user enough times
        recognized_users = check_recognized_users(recognized_users_count)
        if len(recognized_users) > 0:
            break
        cv2.waitKey(1) # Required or else video stream doesn't really render.

    if recognized_users_count:
        recognized_user = max(recognized_users_count,
                              key=recognized_users_count.get)
        if recognized_users_count[recognized_user] < MIN_USER_RECOGNITION_COUNT:
            recognized_user = None
    return recognized_user


# If this program is the main program, authenticate the user.
if __name__ == "__main__":
    import argparse;

    parser = argparse.ArgumentParser(description="Facial Identification Options")
    parser.add_argument("--encodings", "-e", type=str, help="File location of facial encodings.", required=False,
                        default="./encodings.pickle")
    parser.add_argument("--model", "-m", type=str, help="Type of encoding method, either \"hog\" or \"cnn\". HOG is "
                                                        "faster, CNN is more accurate.", required=False,
                        default="hog", choices=["cnn", "hog"])
    parser.add_argument("--flip", "-f", type=int,
                        help="Whether or not to flip the image vertically or horizontally. 0 to flip horizontally, 1 to flip vertically.",
                        required=False, default=None, choices=[0, 1])
    parser.add_argument("--show", "-s", action="store_true",
                        help="Include this argument to have the image shown to you.", default=False)
    args = parser.parse_args()
    user = recognize_user(encoding_model=args.model, encodings_location=args.encodings, image_flip=args.flip,
                          draw_rectangles=args.show)
    if user:
        print(f"Recognized user {user}.")
