# ---   CLASS   --- #
# ----------------- #
class DictUtils:
    """
    :author: Alberto M. Esmoris Pena
    Class with util static methods to work with dictionaries.
    """
    # ---   METHODS   --- #
    # ------------------- #
    @staticmethod
    def delete_by_val(dict, val):
        """
        Delete all the entries on the dictionary with exactly the
            given value.
        :param dict: The dictionary to be updated.
        :param val: The value of the entries to be removed.
        :return: A version of the input dictionary after deleting the
            requested entries.
        """
        return {k: v for k, v in dict.items() if v != val}
