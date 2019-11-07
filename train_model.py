"""
Responsible for training the machine learning model for recognizing faces.
"""


from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import common
import data_handler

def train_and_save(facial_embeddings_database: str= common.EMBEDDINGS_LOC, output_file: str = common.RECOGNITION_DATABASE_LOC) -> None:
    """
    Trains the database using the given facial embeddings database and outputs the results to file.
    :param facial_embeddings_database: The facial embedding database location.
    :param output_file: The file location for the output of the database.
    :return: None
    """
    database = data_handler.load_database(facial_embeddings_database)
    data_handler.write_database(output_file, train_model(database))


def train_model(facial_embeddings: dict) -> SVC:
    """
    Trains the model for the given database
    :param facial_embeddings_database: The location of the pickle database.
    :param output_file: File location where to output the pickle database of facial recognitions.
    :return:
    """
    label_encoder = LabelEncoder()
    user_id_repeat_list = []
    for user_id, encodings in facial_embeddings.items():
        user_id_repeat_list.extend([user_id for x in range(len(encodings))])

    # The facial_embeddings
    labels = label_encoder.fit_transform(user_id_repeat_list)

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    # TODO not too sure this line does what is intended.
    recognizer.fit(data_handler.get_encodings_in_database(facial_embeddings), labels)

    return recognizer

if __name__ == "__main__":
    train_and_save()