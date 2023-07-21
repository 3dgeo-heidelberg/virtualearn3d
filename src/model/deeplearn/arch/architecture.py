# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class Architecture:
    """
    :author: Alberto M. Esmoris Pena

    The architecture class represents any deep learning architecture for
    classification and regression tasks in point clouds.

    Any deep learning architecture can be divided intro three parts:

    1) The preprocessing. What needs to be computed on the input before feeding
    it to the neural network. See :meth:`architecture.Architecture.run_pre`.

    2) The neural network. The neural network itself. See
    :meth:`architecture.Architecture.build`.

    3) The postprocessing. What needs to be computed on the neural network's
    output to obtain the final output. See
    :meth:`architecture.Architecture.run_post`.

    The responsibility of an architecture is to define all the previous parts.
    However, compiling, training, and predicting are operations that lie
    outside the scope of the architecture. Instead, they must be handled by
    any model that uses the architecture.

    :ivar pre_runnable: The callable to run the preprocessing logic.
    :vartype pre_runnable: callable
    :ivar post_runnable: The callable to run the postprocessing logic.
    :vartype post_runnable: callable
    :ivar nn: The built neural network.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, **kwargs):
        """
        Initialize the member attributes of the architecture.

        :param kwargs: The key-word specification defining the architecture.
        """
        # Call parent's init
        super().__init__()
        # Assign architecture attributes
        self.pre_runnable = kwargs.get('pre_runnable', None)
        self.post_runnable = kwargs.get('post_runnable', None)
        self.nn = None  # By default, there is no built neural network

    # ---  ARCHITECTURE METHODS  --- #
    # ------------------------------ #
    def run_pre(self, inputs, **kwargs):
        """
        Run the preprocessing logic.

        :param inputs: The arbitrary inputs for the architecture.
        :param kwargs: The key-word arguments to govern the preprocessing.
        :return: The transformed input ready for the neural network.
        """
        if self.pre_runnable is None:
            return inputs
        return self.pre_runnable(inputs, **kwargs)

    def run_post(self, outputs, **kwargs):
        """
        Run the postprocessing logic.

        :param outputs: The outputs from the architecture's neural network.
        :param kwargs: The key-word arguments to govern the postprocessing.
        :return: The transformed output.
        """
        if self.post_runnable is None:
            return outputs
        return self.post_runnable(outputs, **kwargs)

    def build(self, **kwargs):
        r"""
        Build the neural network model.

        This method can be overriden by derived classes to update the building
        logic. However, the baseline architecture provides a basic building
        logic to any child model:

        1) Build the input layer, :math:`x_{\mathrm{in}}`
        2) Build the hidden layer, :math:`x = f(x_{\mathrm{in}})`
        3) Build the output layer, :math:`y = g(x)`

        :param kwargs: The key-word arguments to govern the building of the
            neural network.
        :return: Nothing, but the nn attribute is updated.
        """
        # Input layer
        inlayer = self.build_input(**kwargs)
        # Hidden layers
        hidlayer = self.build_hidden(inlayer, **kwargs)
        # Output layer
        outlayer = self.build_output(hidlayer, **kwargs)
        # Model
        self.nn = tf.keras.Model(
            inputs=inlayer,
            outputs=outlayer,
            name='PointNet'
        )

    def is_built(self):
        """
        Check whether the architecture has been built (True) or not (False).

        :return: True if the architecture has been built, False otherwise.
        :rtype: bool
        """
        return self.nn is not None

    @abstractmethod
    def build_input(self, **kwargs):
        """
        Any derived class that aims to provide an operating architecture must
        override this method to define the inputs of the neural network.

        :param kwargs: The key-word arguments governing the neural network's
            inputs.
        :return: The input layer or a list/tuple/dict of input layers.
        :rtype: :class:`tf.Tensor` or list or tuple or dict
        """
        pass

    @abstractmethod
    def build_hidden(self, inputs, **kwargs):
        """
        Any derived class that aims to provide an operating architecture must
        override this method to define the hidden layers of the architecture.

        :param inputs: The neural network's inputs.
        :param kwargs: The key-word arguments governing the neural network's
            hidden layers.
        :return: The last hidden layer or a list/tuple/dict of hidden layers.
        :rtype: :class:`tf.Tensor` or list or tuple or dict
        """
        pass

    @abstractmethod
    def build_output(self, inputs, **kwargs):
        """
        Any derived class that aims to provide an operating architecture must
        override this method to define the outputs of the neural network.

        :param inputs: The inputs to compute the outputs (i.e., the inputs for
            the output layer/s, not the inputs for the neural network).
        :param kwargs: The key-word arguments governing the neural network's
            output layers.
        :return: The output layer or a list/tuple/dict of output layers.
        :rtype: :class:`tf.Tensor` or list or tuple or dict
        """
        pass
