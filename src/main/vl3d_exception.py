class VL3DException(Exception):
    """
    :author: Alberto M. Esmoris Pena

    Base class for VirtuaLearn3D custom exceptions.

    :ivar message: The default message providing a string representation of
        the exception.
    :vartype message: str
    """
    def __init__(self, message=''):
        self.message = message

    def __str__(self):
        return self.message
