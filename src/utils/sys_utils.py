# ---   IMPORTS   --- #
# ------------------- #
import src.main.main_logger as LOGGING
import psutil
import multiprocessing
import inspect


# ---   CLASS   --- #
# ----------------- #
class SysUtils:
    """
    :author: Alberto M: Esmoris Pena

    Class with util static methods related to the system where the VL3D
    framework is run.
    """
    # ---   METHODS   --- #
    # ------------------- #
    @staticmethod
    def get_sys_mem():
        """
        Obtain the system memory, in bytes.

        :return: System memory, in bytes.
        :rtype: int
        """
        return psutil.virtual_memory().total

    @staticmethod
    def get_sys_threads():
        """
        Obtain the maximum number of parallel threads.

        :return: Maximum number of parallel threads.
        :rtype: int
        """
        return multiprocessing.cpu_count()

    @staticmethod
    def validate_requested_threads(
        nthreads, warning=True, raise_exception=False, caller='CALLER'
    ):
        """
        Validate that the number of threads (nthreads) is not greater than
        the number of available threads.

        :param nthreads: The number of threads.
        :type nthreads: int
        :param warning: Flag to control whether to emit a warning message
            through the logging system (True) or not (False).
        :type warning: bool
        :param raise_exception: Flag to control whether to raise a exception
            when
        :type raise_exception: bool
        :param caller: The method's caller. It can be either the name or
            the class.
        :type caller: str or class
        """
        sys_threads = SysUtils.get_sys_threads()
        valid = nthreads <= sys_threads
        if valid:  # Leave the function as requested threads are acceptable
            return
        # From here on, the number of threads might be problematic
        if inspect.isclass(caller):  # If caller given as class
            caller = caller.__name__  # take the class name as caller
        if warning:
            LOGGING.LOGGER.warning(
                f'{caller} requested {nthreads} threads but the system '
                f'is not expected to support more than {sys_threads} '
                'parallel threads.'
            )
        if raise_exception:
            raise ValueError(
                f'{caller} requested {nthreads} threads but the system '
                f'supports up to {sys_threads} parallel threads.'
            )
