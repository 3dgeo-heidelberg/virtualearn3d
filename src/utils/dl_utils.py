# ---   CLASS   --- #
# ----------------- #
class DLUtils:
    """
    :author: Alberto M. Esmoris Pena

    Class with utils static methods to work with deep learning.
    """
    # ---   METHODS   --- #
    # ------------------- #
    @staticmethod
    def is_using_binary_crossentropy(comp_args, default=False):
        """
        Check whether the compilation arguments dictionary uses a binary
        cross-entropy loss function (True) or not (False).

        :param comp_args: The compilation arguments.
        :type comp_args: dict
        :param default: Default value to be assummed when the loss function
            cannot be explicitly checked.
        :type default: bool
        :return: True if a binary cross-entropy is used, False otherwise.
        :rtype: bool
        """
        # If no compilation arguments are given
        if comp_args is None:
            return default
        # Explore given compilation arguments
        loss_args = comp_args.get('loss', None)
        # If loss function arguments are not explicitly defined
        if loss_args is None:
            return default
        # Explore given loss function specification
        fun_name = loss_args.get('function', '').lower()
        return (
            fun_name == 'binary_crossentropy' or
            fun_name == 'class_weighted_binary_crossentropy'
        )
