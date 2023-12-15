# ---   IMPORTS   --- #
# ------------------- #
from abc import abstractmethod
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.regularizer.features_orthogonal_regularizer import \
    FeaturesOrthogonalRegularizer
from src.model.deeplearn.layer.features_structuring_layer import \
    FeaturesStructuringLayer
from src.model.deeplearn.layer.rbf_feat_extract_layer import \
    RBFFeatExtractLayer
from src.model.deeplearn.layer.rbf_feat_processing_layer import \
    RBFFeatProcessingLayer
from src.inout.io_utils import IOUtils
import src.main.main_logger as LOGGING
import tensorflow as tf
import os


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
    :ivar nn_path: The path to the file that represents the built neural
        network. The neural network will only be serialized when a not None
        path is provided.
    :vartype nn_path: str
    :ivar build_args: The key-word arguments used to build the neural network.
    :vartype build_args: dict
    :ivar architecture_graph_path: The path where the graph representing the
        architecture will be stored. If None, no architecture graph is
        plotted.
    :vartype architecture_graph_path: str
    :ivar architecture_graph_args: The key-word arguments governing the
        format of the graph representing the architecture.
    :vartype architecture_graph_args: dict
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
        self.nn_path = None  # By default, no file contains the built neuralnet
        self.build_args = None  # At instantiation, build args are not given
        self.architecture_graph_path = kwargs.get(
            'architecture_graph_path', None
        )
        self.architecture_graph_args = kwargs.get(
            'architecture_graph_args', {
                "show_shapes": True,
                "show_dtype": True,
                "show_layer_names": True,
                "rankdir": "TB",
                "expand_nested": True,
                "dpi": 300,
                "show_layer_activations": True
            }
        )
        # Internal references
        self.inlayer = None  # The input layer

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
        # Cache build args
        self.build_args = kwargs
        # Input layer
        self.inlayer = self.build_input(**kwargs)
        # Hidden layers
        hidlayer = self.build_hidden(self.inlayer, **kwargs)
        # Output layer
        outlayer = self.build_output(hidlayer, **kwargs)
        # Model
        self.nn = tf.keras.Model(
            inputs=self.inlayer,
            outputs=outlayer,
            name='PointNet'
        )
        # Plot the architecture's graph
        self.plot()

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

    def plot(self):
        """
        Plot the model's architecture as a graph if requested, i.e.,
        a path for the architecture's graph is available as member attribute.

        :return: Nothing, but the plot is written to a file.
        """
        # Plot model graph
        if self.architecture_graph_path is not None:
            IOUtils.validate_path_to_directory(
                os.path.dirname(self.architecture_graph_path),
                'Deep learning Architecture received a path that does not '
                'point to an accessible directory:',
                True
            )
            tf.keras.utils.plot_model(
                self.nn,
                to_file=self.architecture_graph_path,
                **self.architecture_graph_args
            )
            LOGGING.LOGGER.info(
                'Deep learning architecture graph exported to '
                f'"{self.architecture_graph_path}"'
            )

    def overwrite_pretrained_model(self, spec):
        """
        Assist the :meth:`model.Model.overwrite_pretrained_model` method
        through assisting the
        :meth:`dl_model_handler.DLModelHandler.overwrite_pretrained_model`
        method.

        :param spec: The key-word specification containing the model's
            arguments.
        :type spec: dict
        """
        spec_keys = spec.keys()
        # Overwrite architecture's attributes
        if 'architecture_graph_args' in spec_keys:
            self.architecture_graph_args = spec['architecture_graph_args']
        if 'architecture_graph_path' in spec_keys:
            self.architecture_graph_path = spec['architecture_graph_path']
            self.plot()
        # Overwrite the attributes of the pre-processor
        if self.pre_runnable is not None and 'pre_processing' in spec_keys:
            if hasattr(self.pre_runnable, 'overwrite_pretrained_model'):
                self.pre_runnable.overwrite_pretrained_model(
                    spec['pre_processing']
                )

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def __getstate__(self):
        """
        Method to be called when saving the serialized deep learning
        architecture.

        :return: The state's dictionary of the object.
        :rtype: dict
        """
        # Save the built neural network
        nn_path = None
        if self.nn_path is not None and self.nn is not None:
            nn_path = self.nn_path
            self.nn.save(
                nn_path,
                overwrite=True,
                # include_optimizer=True,  # Not supported with Keras format
                save_format='keras'
            )
        # Return architecture state (for serialization)
        return {  # Must not include built architecture
            'pre_runnable': self.pre_runnable,
            'post_runnable': self.post_runnable,
            'nn': None,
            'nn_path': nn_path,
            'build_args': self.build_args
        }

    def __setstate__(self, state):
        """
        Method to be called when loading and deserializing a previously
        serialized deep learning architecture.

        :param state: The state's dictionary of the saved deep learning
            architecture.
        :type state: dict
        :return: Nothing, but modifies the internal state of the object.
        """
        # Must rebuild the architecture (it was not serialized)
        self.pre_runnable = state['pre_runnable']
        self.post_runnable = state['post_runnable']
        self.nn_path = state['nn_path']
        self.build_args = state['build_args']
        self.architecture_graph_path = None
        self.architecture_graph_args = None
        # Load or rebuild
        if self.nn_path is not None:  # If path to neuralnet, load it
            if not os.path.exists(self.nn_path):
                self.nn_path = input(
                    '\n'
                    f'The "{self.nn_path}" file does not exist.\n'
                    'Please, type the path to the serialized neural network: '
                )
                print()
            self.nn = tf.keras.models.load_model(
                self.nn_path,
                custom_objects={
                    'FeaturesOrthogonalRegularizer':
                        FeaturesOrthogonalRegularizer,
                    'FeaturesStructuringLayer': FeaturesStructuringLayer,
                    "RBFFeatExtractLayer": RBFFeatExtractLayer,
                    'RBFFeatProcessingLayer': RBFFeatProcessingLayer

                },
                compile=False
            )
        elif self.build_args is not None:  # Otherwise, rebuild
            self.build(**self.build_args)
        else:
            raise DeepLearningException(
                'Deep learning Architecture suffers an inconsistent state '
                'on deserialization.'
            )
