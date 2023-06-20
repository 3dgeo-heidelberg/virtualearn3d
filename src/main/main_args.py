# ---   CLASS   --- #
# ----------------- #
class ArgsParser:
    """
    :author: Alberto M. Esmoris Pena
    Class with util static methods to parse the input arguments often given
        by command line.
    """
    @staticmethod
    def parse_main_type(argv):
        """
        Find the main type of the requested execution branch.
        :param argv: The input arguments.
        :return: ("vl3d", x) for the virtualearn3d branch, ("test", None) for
            the test branch. In the case of the virtualearn3d branch, the x
            element is another string specifying the type of branch, i.e.,
            mine, train, predict, eval, or pipeline.
        :rtype: tuple
        """
        arg = argv[1]  # First input argument (0 is the script's name)
        # If first input argument is as expected, return corresponding output
        if arg == "--mine":
            return "vl3d", "mine"
        elif arg == "--train":
            return "vl3d", "train"
        elif arg == "--predict":
            return "vl3d", "predict"
        elif arg == "--eval":
            return "vl3d", "eval"
        elif arg == "--pipeline":
            return "vl3d", "pipeline"
        elif arg == '--test':
            return "test", None
        # On unexpected first input argument
        raise ValueError(
            f'Unexpected first input argument: "{arg}"'
        )