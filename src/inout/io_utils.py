# ---   IMPORTS   --- #
# ------------------- #
import src.main.main_logger as LOGGING
import os


# ---   CLASS   --- #
# ----------------- #
class IOUtils:
    """
    :author: Alberto M. Esmoris Pena

    Class with util static methods for general input/output operations.
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
        msg='Given path does not point to an accessible directory:',
        try_to_make=True
    ):
        """
        Validate the given path points to an accessible directory.

        :param path: Path to a directory that must be validated.
        :type path: str
        :param msg: Message for the exception.
        :type msg: str
        :param try_to_make: Whether try to make the given directory (True) or
            just validate it exists (False).
        :type try_to_make: bool
        :return: Nothing but an exception will be raised if the path is not
            valid.
        """
        # Validate input path as directory
        if not os.path.isdir(path):
            if try_to_make:  # Try to make the directory
                try:
                    os.makedirs(path, mode=0o754, exist_ok=True)
                    return
                except Exception as ex:
                    LOGGING.LOGGER.debug(
                        f'Failed to make directory "{path}":\n{ex}'
                    )
            raise NotADirectoryError(f'{msg}\n"{path}"')
