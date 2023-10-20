# ---   IMPORTS   --- #
# ------------------- #
from src.main.vl3d_exception import VL3DException
import src.main.main_logger as LOGGING
import traceback


# ---   EXCEPTIONS   --- #
# ---------------------- #
class VL3DTestException(VL3DException):
    """
    :author: Alberto M. Esmoris Pena

    Class for exceptions related to tests.
    See :class:`.VL3DException`.
    """
    def __init__(self, message=''):
        # Call parent VL3DException
        super().__init__(message)


# ---   CLASS   --- #
# ----------------- #
class VL3DTest:
    """
    :author: Alberto M. Esmoris Pena

    Base implementation for any VL3D test. It must be extended by classes
    aiming to provide runnable tests. Each derived class must overload the run
    method to implement the test's logic.

    :ivar name: The name of the test.
    :vartype name: str
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, name='VL3D test'):
        """
        Basic configuration for any VL3D test.

        :param name: Test name
        :type name: str
        """
        self.name = name

    # ---  TEST INTERFACE  --- #
    # ------------------------ #
    def run(self):
        """
        Run the test.

        :return: True if test is successfully passed, False otherwise.
        :rtype: bool
        """
        raise VL3DTestException(
            "VL3DTest cannot be run. Please, override run method properly."
        )

    # ---  COMMON TEST BEHAVIOR  --- #
    # ------------------------------ #
    def start(self):
        """
        Run the test and report its final status.

        :return: True when the test has been passed, Fase otherwise.
        :rtype: bool
        """
        # Run test
        try:
            status = self.run()
        except Exception as ex:
            LOGGING.LOGGER.warning(
                f'{self.name} raised and exception: {ex}\n\n'
                f'{traceback.format_exc()}'
            )
            status = False
            raise ex  # TODO Remove
        # Report status
        if status:
            LOGGING.LOGGER.info(
                '\033[1m{name:64}    \033[92m[PASSED]\033[0m'
                .format(name=self.name)
            )
        else:
            LOGGING.LOGGER.info(
                '\033[1m{name:64}    \033[91m[FAILED]\033[0m'
                .format(name=self.name)
            )
        # Return
        return status
