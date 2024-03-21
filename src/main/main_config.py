# ---   IMPORTS   --- #
# ------------------- #
import src.main.main_logger as LOGGING
import yaml
import sys
import os


# ---   CONFIG   --- #
# ------------------ #
# Global variable for the global config dictionary
VL3DCFG = {
    'IO': None,
    'EVAL': None,
    'MINING': None,
    'MODEL': None,
    'TEST': None
}

def main_config_init(rootdir=''):
    """
    Initialize the main config (global variable VL3DCFG).
    The main config defines the default values for many components in the
    framework. Some values are just defaults that can be overridden by
    JSON specifications (e.g., pipelines). Other values define the internal
    mechanics of the framework (e.g., what ratio of system memory must be
    occupied for the automatic mem-to-file proxy to be triggered).

    :param rootdir: Path to the directory where the vl3d.py script is located.
    :type rootdir: str
    :return: Nothing.
    """
    # Load IO config
    main_config_subdict_init(rootdir, 'config/io.yml', 'IO')
    # Load evaluator/evaluation config
    main_config_subdict_init(rootdir, 'config/eval.yml', 'EVAL')
    # Load data mining config
    main_config_subdict_init(rootdir, 'config/mining.yml', 'MINING')
    # Load model config
    main_config_subdict_init(rootdir, 'config/model.yml', 'MODEL')
    # Load test config
    main_config_subdict_init(rootdir, 'config/test.yml', 'TEST')
    # Report successful initialization
    LOGGING.LOGGER.debug('VL3DCFG was successfully loaded!')


def main_config_subdict_init(dirpath, subpath, subkey):
    """
    Assist the :meth:`main_config.main_config_init` method to load each
    subdictionary.

    :param dirpath: The path where the vl3d.py script is located.
    :type dirpath: str
    :param subpath: The path, relative to the vl3d.py script path, where the
        YAML config file is located.
    :type subpath: str
    :param subkey: The key of the dictionary inside the main VL3DCFG
        dictionary.
    :type subkey: str
    :return: Nothing, but the VL3DCFG global dictionary is updated.
    """
    global VL3DCFG
    with open(os.path.join(dirpath, subpath), "r") as ymlf:
        yml = yaml.safe_load(ymlf)
        VL3DCFG[subkey] = yml
