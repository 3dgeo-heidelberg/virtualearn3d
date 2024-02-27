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

    @staticmethod
    def add_defaults(dict, defaults):
        """
        For any value that is not explicitly available in the input dictionary
        `dict`, set it from the `defaults` dictionary (if available).

        NOTE updates are done in place.

        :param dict: The input dictionary whose defaults must be set.
        :param defaults: The dictionary with the default values.
        :return: The updated input dictionary `dict`.
        """
        # Return dict directly when no defaults are given
        if defaults is None:
            return dict
        # Set any missing default
        for k, v in defaults.items():
            dict.setdefault(k, v)
        return dict
