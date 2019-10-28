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
    with open(file_location, "rb") as database:
        file_content = pickle.load(database)
    return file_content


def write_database(output_file: str, database_content: dict) -> None:
    """
    Writes the dictionary database to the given file location
    :param output_file: The location of the file to be outputted on.
    :param database_content: The database content to be written to the file.
    :return: None
    """
    with open(output_file, "wb") as output:
        output.write(pickle.dump(database_content, output))
