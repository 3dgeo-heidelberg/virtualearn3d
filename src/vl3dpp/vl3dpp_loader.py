import src.main.main_logger as LOGGING
import sys
import os


def vl3dpp_load(logging=True, warning=True):
    """
    Loads the VL3DPP backend.

    :param logging: True to enable logging messages, False otherwise. Note
        warning messages are not disabled with this flag.
    :param warning: Flag to specifically enable (True) or disable (False)
        warning messages.
    :return: Nothing at all, but the backend will be accessible after calling
        this method.
    """
    # Prepare paths
    sys_path = sys.path
    cwd = os.getcwd()
    dir_release = os.path.join(cwd, 'cpp/build')
    dir_debug = os.path.join(cwd, 'cpp/build-debug')
    # First, try for release
    release_loaded = dir_release in sys_path
    if release_loaded:  # Already loaded
        if logging:
            LOGGING.LOGGER.debug(
                'VL3D++ was already loaded in RELEASE mode.'
            )
        return
    elif os.path.exists(dir_release):
        sys.path.append(dir_release)
        if logging:
            LOGGING.LOGGER.debug(
                'VL3D++ was loaded in RELEASE mode from '
                f'"{dir_release}".'
            )
        return
    # Second, try for debug
    debug_loaded = dir_debug in sys_path
    if debug_loaded:  # Already loaded
        if logging:
            LOGGING.LOGGER.debug(
                'VL3D++ was already loaded in DEBUG mode.'
            )
        return
    elif os.path.exists(dir_debug):
        sys.path.append(dir_debug)
        if logging:
            LOGGING.LOGGER.debug(
                'VL3D++ was loaded in DEBUG mode from '
                f'"{dir_release}".'
            )
        return
    # Warning, could not load VL3D++ backend
    if warning:
        LOGGING.LOGGER.warning(
            'VL3D++ could not be loaded.'
        )
