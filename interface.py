"""
API Interface to the modules in this project. To be used by the UI application.
"""

import os

import authenticate_user as authenticate
import common
import register_user as register


def authenticate_user(base_dir: str = os.getcwd()) -> str:
    """
    Interface for authenticating a user with the given base directory location.
    :param base_dir: The base directory, from which all default paths are relative. Defaults to current working directory.
    :return: The user_id of the recognized user, or empty string if none was recognized.
    """
    recognized = authenticate.recognize_user_from_database(database_loc=os.path.join(base_dir, common.DATA_DIR))
    return recognized if recognized is not None else ""


def register_user(base_dir: str = os.getcwd()) -> None:
    """
    Registers new added users in the database.
    :param base_dir: The base directory, from which all default paths are relative. Defaults to current working directory.
    :return: None
    """
    register.register_users_and_save(directory_location=os.path.join(base_dir, common.DATASET_DIR),
                                     delete_images_on_complete=True, overwrite_data=False)
