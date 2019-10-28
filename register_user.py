"""
Creates a facial recognition profile for a new user.
"""
import common
import data_handler
from common import USER_IDS_KEY, start_video_stream
import face_recognition
import cv2
import os


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


def register_user(user_id: str, dataset_dir: str, encoding_model="hog", database_loc: str = common.DATABASE_LOC,
                  show_output: bool = False):
    """
    Function for registering a new user using the given video source. If video source isn't provided, then the camera
    on id 0 is used.
    :param user_id: The user id for the user that is being registered.
    :param dataset_dir: The directory location of pictures for the user.
    :param encoding_model: The type of encoding model. Must be either "hog" or "cnn". HOG is faster, CNN is more thorough.
    :param database_loc: Location of the pickle file database.
    :return: Encoded face that was detected, or None if no face was detected or if there was another error.
    """
    processed_images = []
    for (i, filename) in enumerate(os.listdir(dataset_dir)):
        full_path = os.path.join(dataset_dir, filename)
        # Might want to check file validity here at some point, but won't for now.
        image = cv2.imread(full_path)
        if image is not None:
            processed = process_image(image, encoding_model=encoding_model)
            if show_output:
                print(f"Processing image {i + 1}")
            if processed:
                processed_images.append(processed)

    if len(processed_images) > 0:  # Only do things if we actually have stuff to add.
        user_info = data_handler.load_database(database_loc)
        if user_id not in user_info.keys():
            user_info[user_id] = []
        user_info[user_id].extend(processed_images)
        data_handler.write_database(database_loc, user_info)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Facial Registration Options")
    parser.add_argument("user_id", type=str, help="User ID to be used")
    parser.add_argument("--encodings", "-e", type=str, help="File location to output encodings.", required=True,
                        default="./encodings.pickle")
    parser.add_argument("--dataset", "-d", type=str, help="Directory location of the dataset images.", required=True)
    parser.add_argument("--model", "-m", type=str, help="Type of encoding method, either \"hog\" or \"cnn\". HOG is "
                                                        "faster, CNN is more accurate.", required=False,
                        default="hog", choices=["cnn", "hog"])
    args = parser.parse_args()
    register_user(user_id=args.user_id, dataset_dir=args.dataset, encoding_model=args.model,
                  database_loc=args.encodings, show_output=True)
