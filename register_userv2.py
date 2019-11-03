import os

import cv2
import imutils
import numpy
from imutils import paths as impaths

import common
import data_handler

IMAGE_RESIZE_WIDTH = 600


def detect_faces(face_detector, image):
    """
    Detects faces in the provided image using the provided face_detector.
    :param face_detector: The face_detector.
    :param image: The image to be processed.
    :return: The detected faces in the image.
    """
    image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), scalefactor=1.0, size=(300, 300),
                                       mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_detector.setInput(image_blob)
    return face_detector.forward()


def find_best_match(faces, image, min_confidence: float = 0.5):
    """
    Finds the single face in the given list of faces with the best match
    :param image: The image from which the face was detected.
    :param faces: The list of faces to go through.
    :param min_confidence: The minimum percentage confidence in order for a face to be considered recognized.
    :return: The best matched face, or None if none were matched or none were matched with a large enough confidence.
    """
    best_match = None

    if len(faces) > 0:
        # Assume that each image has only one face, so take the bounding box with the largest probability of being a face.
        i = numpy.argmax(faces[0, 0, :, 2])
        confidence = faces[0, 0, i, 2]

        # Only continue if the confidence is enough
        if confidence > min_confidence:
            img_height, img_width = image.shape[0:2]
            # Determine the bounding box for the face
            box = faces[0, 0, i, 3:7] * numpy.array([img_width, img_height, img_width, img_height])
            # Get start and end positions for the box.
            startx, starty, endx, endy = box.astype("int")

            # Extract face ROI and get dimensions for it
            face = image[starty: endy, startx:endx]
            face_height, face_width = face.shape[0:2]

            # Don't match the face if it's too small.
            if face_width >= 20 and face_height >= 20:
                best_match = face

    return best_match


def extract_face_embeddings(face, embedding_cnn):
    """
    Extracts the facial embeddings for the given face
    :param face: The face for which embeddings should be created
    :param embedding_cnn: The embedding cnn to be used.
    :return: The embeddings for the face
    """
    # Construct a blob and pass it to the embedder to obtain a 128-d quantification for the face.
    face_blob = cv2.dnn.blobFromImage(face, scalefactor=1.0 / 255, size=(96, 96), mean=(0, 0, 0), swapRB=True,
                                      crop=False)
    embedding_cnn.setInput(face_blob)
    vec = embedding_cnn.forward()

    return vec


def process_dataset(directory_location: str, detector_dir: str = common.FACE_DETECTION_MODEL_DIR,
                    embedding_model_path: str = common.EMBEDDINGS_PROCESSOR_LOC,
                    show_output: bool = False, file_output: str = None) -> dict:
    """
    Processes the images in the given directory for facial identification.
    :param directory_location: The path to a directory full of a dataset of images for the same person.
    Note that each subdirectory within this directory should be named the same as the user_id for the user.
    E.g:
    dataset
      --- some_user_id
        -- image1.png
        -- image2.png
      --- some_other_user_id
        -- image1.png
        -- image2.png
    :param detector_dir: String location of the detection file directory.
    :param embedding_model_path: The path to the embedding model.
    :param show_output: True to print progress, False otherwise.
    :param file_output: The pickle file to which the embeddings should be outputted. None means it won't be saved.
    :return: The processed dataset dictionary, with format { "user_id" : [encoding1, encoding2, ...] , ... }
    """
    # Dictionary with results.
    result_database = {}

    image_paths = list(impaths.list_images(directory_location))
    face_detector = common.load_detector_from_dir(detector_dir)
    embedding_cnn = common.load_embedding_model(embedding_model_path)

    for (i, image_path) in enumerate(image_paths):
        current_user_id: str = image_path.split(os.path.sep)[-2]
        if show_output:
            print(f"Processing image {i + 1} for user {current_user_id}.")
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=IMAGE_RESIZE_WIDTH)

        faces = detect_faces(face_detector, image)
        face = find_best_match(faces, image)
        if face is not None:
            facial_embeddings = extract_face_embeddings(face, embedding_cnn)

            if facial_embeddings is not None and len(facial_embeddings) > 0:
                if current_user_id not in result_database:
                    result_database[current_user_id] = []
                result_database[current_user_id].append(facial_embeddings)

    if file_output is not None:
        data_handler.write_database(file_output, result_database)

    return result_database


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Registers users' facial encodings from a dataset of images containing their face.")
    parser.add_argument("dataset", type=str, help="Location of the dataset which should be processed.")
    parser.add_argument("output", type=str,
                        help="Location of the output pickle database file to which the encodings should be written.")
    args = parser.parse_args()
    process_dataset(args.dataset, show_output=True, file_output=args.output)
