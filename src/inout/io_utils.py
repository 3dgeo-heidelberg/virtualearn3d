# ---   IMPORTS   --- #
# ------------------- #
import src.main.main_logger as LOGGING
import urllib.request as urlreq
import urllib.error as urlerr
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
        msg='Cannot find file at given path:',
        accept_url=False
    ):
        """
        Validate the given path points to an accessible file.

        :param path: Path to a file that must be validated.
        :type path: str
        :param msg: Message for the exception.
        :type msg: str
        :param accept_url: If True, when the given path is a valid and
            accessible URL it will be accepted as a valid path. If False,
            URLs will not be accepted.
        :type accept_url: bool
        :return: Nothing but an exception will be raised if the path is not
            valid.
        """
        # Check input path as URL
        if IOUtils.is_url(path):
            IOUtils.validate_url_to_file(path, msg)
            return
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

    @staticmethod
    def validate_url_to_file(
        url,
        msg='Given URL is not accessible:'
    ):
        """
        Validate the given URL can be accessed.

        :param url: The URL to be checked.
        :type url: str
        :param msg: Message for the exception.
        :type msg: str
        :return: Nothing but an exception will be raised if the URL is not
            accessible.
        """
        # Validate
        open_url = urlreq.urlopen(url)
        is_readable = open_url.readable()
        open_url.close()
        # On not valid
        if not is_readable:
            raise urlerr.URLError(f'{msg}\n"{url}"')

    # ---   CHECKS   --- #
    # ------------------ #
    @staticmethod
    def is_url(s):
        """
        Check whether the given string represents a URL.

        :param s: The string to be checked.
        :return: True if the given string represents a URL, false otherwise.
        """
        return s[:7] == 'http://' or s[:8] == 'https://'
