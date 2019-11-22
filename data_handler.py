"""
General IO for pickle database operations.
"""

import os
import pickle


def get_user_ids_in_database(database: dict) -> list:
    """
    Gets all the user_ids in the given database.
    :param database: The database to look through.
    :return: All the user_ids in the database.
    """
    return list(database.keys())


def get_encodings_in_database(database: dict):
    """
    Gets a list of all encodings in the given database.
    :param database: The database dictionary.
    :return: All the encodings
    """
    result = []
    for encodings in database.values():
        result.extend(encodings)
    return result


def load_database(file_location: str):
    """
    Attempts to load the pickle database at the given file location
    :param file_location: String location of file to be loaded.
    :return: The loaded pickle database.
    """
    file_content = {}
    try:
        with open(file_location, "rb") as database:
            file_content = pickle.load(database)
    except (FileNotFoundError, EOFError):
        file_content = {}
    return file_content


def write_database(output_file: str, database_content) -> None:
    """
    Writes the dictionary database to the given file location
    :param output_file: The location of the file to be outputted on.
    :param database_content: The database content to be written to the file.
    :return: None
    """
    if output_file and database_content and database_content is not None:
        directory: str = os.path.dirname(output_file)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        with open(output_file, "wb") as output:
            pickle.dump(database_content, output)
