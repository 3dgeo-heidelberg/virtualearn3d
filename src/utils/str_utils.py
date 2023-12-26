# ---   CLASS   --- #
# ----------------- #
class StrUtils:
    """
    :author: Alberto M. Esmoris Pena

    Class with util static methods to work with strings.
    """
    # ---   METHODS   --- #
    # ------------------- #
    @staticmethod
    def to_numpy_expr(expr):
        """
        Receive an evaluable expression and replace any call to a standard math
        function to use numpy aliased as np instead.

        :param expr: The expression to be numpyfied.
        :type expr: str
        :return: The numpyfied expression.
        """
        # Remove any available numpy prefix to prevent double numpyfication
        expr = expr.replace('np.', '')
        # Numpyfy common maths
        expr = expr.replace('abs', 'np.abs') \
            .replace('sqrt', 'np.sqrt') \
            .replace('power', 'np.power') \
            .replace('exp', 'np.exp') \
            .replace('log', 'np.log') \
            .replace('pi', 'np.pi') \
            .replace('square', 'np.square')
        # Return numpyfied expression
        return expr
