# ---   IMPORTS   --- #
# ------------------- #
import os


# ---   CLASS   --- #
# ----------------- #
class IOUtils:
    """
    :author: Alberto M. Esmoris Pena
    Class with util static methods for general input/ouput operations.
    """
    # ---   VALIDATION   --- #
    # ---------------------- #
    @staticmethod
    def validate_path_to_file(
        path,
        msg='Cannot find file at given path:'
    ):
        """
        Validate the given path points to an accessible file.
        :param path: Path to a file that must be validated.
        :param msg: Message for the exception.
        :return: Nothing but an exception will be raised if the path is not
            valid.
        """
        # Validate input path as file
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{msg}\n"{path}"')

    @staticmethod
    def validate_path_to_directory(
        path,
        msg='Given path does not point to an accessible directory:'
    ):
        """
        Validate the given path points to an accessible directory.
        :param path: Path to a directory that must be validated.
        :param msg: Message for the exception.
        :return: Nothing but an exception will be raised if the path is not
            valid.
        """
        # Validate input path as directory
        if not os.path.isdir(path):
            raise NotADirectoryError(f'{msg}\n"{path}"')
