# ---   IMPORTS   --- #
# ------------------- #
import src.main.main_logger as LOGGING
from src.main.main_args import ArgsParser
from src.inout.json_io import JsonIO
from src.main.main_mine import MainMine
from src.main.main_train import MainTrain
from src.main.main_predict import MainPredict
from src.main.main_eval import MainEval
from src.main.main_pipeline import MainPipeline
import sys


# ---   MAIN   --- #
# ---------------- #
def main():
    """
    The main entry point that governs the main branch that must be called.
    """
    # Configure logging
    LOGGING.main_logger_init()
    # Parse input arguments
    main_type, main_subtype = ArgsParser.parse_main_type(sys.argv)
    # Redirect through corresponding main branch
    if main_type == 'vl3d':
        main_vl3d(main_subtype)
    elif main_type == 'test':
        main_test()
    # Raise exception on unexpected main type
    else:
        raise ValueError(
            f'Unexpected main type: "{main_type}"'
        )


# ---  MAIN VL3D  --- #
# ------------------- #
def main_vl3d(subtype):
    """
    Execute the logic of the virtualearn3d application covering the entire
    training loop and the application of a trained model.

    :param str subtype: The string specifying the type of VL3D branch, i.e.,
        mine, train, predict, eval, or pipeline.
    """
    # Obtain JSON specification
    try:
        spec = JsonIO.read(sys.argv[2])
    except IndexError as iex:
        LOGGING.LOGGER.error('Missing JSON specification argument')
        raise iex
    # Check subtype
    if subtype == 'mine':
        MainMine.main(spec)
    elif subtype == 'train':
        MainTrain.main(spec)
    elif subtype == 'predict':
        MainPredict.main(spec)
    elif subtype == 'eval':
        MainEval.main(spec)
    elif subtype == 'pipeline':
        MainPipeline.main(spec)
    # Raise exception on unexpected subtype
    else:
        raise ValueError(
            f'The main VL3D branch found and unexpected subtype: "{subtype}"'
        )


# ---  MAIN TEST   --- #
# -------------------- #
def main_test():
    """
    Execute the tests.
    """
    # TODO Rethink : Implement
    LOGGING.LOGGER.warning('main_test TODO yet ...')