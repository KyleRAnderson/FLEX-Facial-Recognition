"""
General IO for pickle database operations.
"""

import pickle


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


def write_database(output_file: str, database_content: dict) -> None:
    """
    Writes the dictionary database to the given file location
    :param output_file: The location of the file to be outputted on.
    :param database_content: The database content to be written to the file.
    :return: None
    """
    if output_file and database_content and database_content is not None:
        with open(output_file, "wb") as output:
            pickle.dump(database_content, output)
