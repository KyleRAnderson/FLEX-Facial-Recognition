"""
Creates a facial recognition profile for a new user.
"""
import os

import cv2
import face_recognition
from imutils import paths as impaths

import common
import data_handler


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
    # Only want the first recognized face
    return face_recognition.face_encodings(image_rgb, [boxes[0]]) if boxes and len(boxes) > 0 else []


def delete_file(file_path: str) -> None:
    """
    Deletes the file at the given location.
    :param file_path: The path to the file.
    :return: None
    """
    os.unlink(file_path)


def register_user(user_id: str, dataset_dir: str, encoding_model="hog",
                  show_output: bool = False, delete_on_processed: bool = False):
    """
    Function for registering a new user using the given video source. If video source isn't provided, then the camera
    on id 0 is used.
    :param user_id: The user id for the user that is being registered.
    :param dataset_dir: The directory location of pictures for the user.
    :param encoding_model: The type of encoding model. Must be either "hog" or "cnn". HOG is faster, CNN is more thorough.
    :param show_output: True to print console output for progress, false otherwise.
    :param delete_on_processed: True to delete the image file after processing it, false otherwise.
    :return: Encoded face that was detected, or None if no face was detected or if there was another error.
    """
    processed_images = []
    for (i, filename) in enumerate(impaths.list_images(dataset_dir)):
        # Might want to check file validity here at some point, but won't for now.
        image = cv2.imread(filename)
        if image is not None:
            if show_output:
                print(f"Processing image {i + 1} for user {user_id}")
            processed = process_image(image, encoding_model=encoding_model)
            if processed:
                processed_images.extend(processed)

                # Delete after we're done if we're supposed to.
                if delete_on_processed:
                    delete_file(filename)

    return {user_id: processed_images} if len(processed_images) > 0 else None


def register_users_in_dir(directory_location: str, encoding_model: str = "hog", delete_images_on_complete: bool = False,
                          show_output: bool = False):
    """
    Registers all the users in a directory.
    :param directory_location:
    :param encoding_model: The type of encoding model to use.
    :param delete_images_on_complete: True to delete the images after processing them, false otherwise.
    :param show_output: True to print progress output, false otherwise.
    :return: The dictionary of registered users in the given directory.
    """
    total_dict = {}
    for directory in next(os.walk(directory_location))[1]:
        total_directory = os.path.join(directory_location, directory)
        # Using the directory name as the user_id as well.
        user_dict = register_user(directory, total_directory, encoding_model=encoding_model, show_output=show_output,
                                  delete_on_processed=delete_images_on_complete)
        if user_dict is not None:
            total_dict.update(user_dict)

    return total_dict if len(total_dict) > 0 else None


def register_users_and_save(directory_location: str = common.DATASET_DIR,
                            database_location: str = common.DATABASE_LOC, encoding_model="hog",
                            delete_images_on_complete: bool = True, show_output: bool = False,
                            overwrite_data: bool = False):
    processed_users = register_users_in_dir(directory_location, encoding_model=encoding_model,
                                            delete_images_on_complete=delete_images_on_complete,
                                            show_output=show_output)
    database = data_handler.load_database(database_location) if not overwrite_data else {}
    if processed_users is not None:
        for user_id, encodings in processed_users.items():
            if user_id not in database:
                database[user_id] = []
            database[user_id].extend(encodings)
    data_handler.write_database(database_location, database)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Facial Registration Options")
    parser.add_argument("--encodings", "-e", type=str, help="File location to output encodings.", default=None)
    parser.add_argument("--dataset", "-d", type=str, help="Directory location of the dataset images.", default=None)
    parser.add_argument("--model", "-m", type=str, help="Type of encoding method, either \"hog\" or \"cnn\". HOG is "
                                                        "faster, CNN is more accurate.", required=False, default=None,
                        choices=["cnn", "hog"])
    parser.add_argument("--overwrite", default=False, action="store_true",
                        help="Include this flag to overwrite the database, replacing its content.", required=False)
    args = parser.parse_args()

    args_dict = {}
    if args.encodings is not None:
        args_dict["database_location"] = args.encodings
    if args.dataset is not None:
        args_dict["directory_location"] = args.dataset
    if args.model is not None:
        args_dict["encoding_model"] = args.model

    register_users_and_save(**args_dict, show_output=True, delete_images_on_complete=False,
                            overwrite_data=args.overwrite)
