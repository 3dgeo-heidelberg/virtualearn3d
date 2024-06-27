# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.utils.ptransf.simple_data_augmentor import SimpleDataAugmentor
from src.model.deeplearn.arch.point_net import PointNet
from src.model.deeplearn.arch.rbfnet import RBFNet
from src.model.deeplearn.arch.conv_autoenc_pwise_classif import \
    ConvAutoencPwiseClassif
import tensorflow as tf
import numpy as np


# ---   CLASS   --- #
# ----------------- #
class DLSequencer(tf.keras.utils.Sequence):
    """
    :author: Alberto M. Esmoris Pena

    A deep learning sequencer that governs how the input data is fed to a
    neural network. It handles two main points:

    1) **Memory management**: Explicitly take control of the memory handling
        logic to prevent undesired scenarios, e.g., `InternalError`, or
        memory exhaustion due to TensorFlow loading the full array of batches
        into the GPU memory.

    2) **Data augmentation**: Provides rotations around an arbitrary axis,
        jitter, and scaling through :class:`SimpleDataAugmentor`. Besides,
        it also enables random shuffle of the indices to unbias the training
        process wrt the initial input order.

    :ivar X: The input structure space.
    :ivar y: The input reference values.
    :ivar arch: The neural network architecture.
    :vartype arch: :class:`.Architecture`
    :ivar batch_size: The number of elements per batch.
    :vartype batch_size: int
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, X, y, batch_size, **kwargs):
        """
        Initialize the member attributes of the DLSequencer.

        :param X: The input data.
        :param y: The input reference values.
        :param batch_size: The number of elements per batch.
        :type batch_size: int
        :param kwargs: The key-word specification to parametrize the
            sequencer.
        """
        # Call parent's init
        super().__init__()
        # Assign values to member attributes
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.arch = kwargs.get('arch', None)
        augmentor_spec = kwargs.get('augmentor', None)
        self.augmentor = None
        if augmentor_spec is not None:
            self.augmentor = SimpleDataAugmentor(**augmentor_spec)
        self.random_shuffle_indices = kwargs.get(
            'random_shuffle_indices', False
        )
        # Cache attributes
        self.Irandom = None
        self.shuffled = False

    # ---   SEQUENCE METHODS   --- #
    # ---------------------------- #
    def __len__(self):
        """
        Obtain the number of input batches.

        :return: Number of input batches.
        :rtype: int
        """
        if isinstance(self.X, list):
            return int(np.ceil(len(self.X[0]) / self.batch_size))
        else:
            return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        """
        Obtain the batch corresponding to the given index.

        :param idx: The index identifying the batch that must be obtained.
        :type idx: int
        :return: The batch corresponding to the given index as a tuple (X, y).
        :rtype: tuple
        """
        # Apply random shuffle, if necessary
        if (
            self.random_shuffle_indices and
            self.Irandom is not None and
            not self.shuffled
        ):
            if isinstance(self.X, list):
                tensors_per_pcloud = len(self.X)  # Tensors per input pcloud
                for i in range(tensors_per_pcloud):
                    self.X[i] = self.X[i][self.Irandom]
                self.y = self.y[self.Irandom]
            else:
                self.X, self.y = self.X[self.Irandom], self.y[self.Irandom]
            self.shuffled = True
        # Obtain start and end points for the indexing interval
        max_idx = len(self.X[0]) if isinstance(self.X, list) else len(self.X)
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, max_idx)
        # Extract batch
        batch_X = self.extract_input_batch(start_idx, end_idx)
        batch_y = self.extract_reference_batch(start_idx, end_idx)
        # Apply data augmentation
        if self.augmentor is not None:
            K = self.find_augmentation_elements()
            if K is None or len(K) < 1:  # Only one element per input
                batch_X = self.augmentor.augment(batch_X)
            else:  # Many elements per input
                batch_X_K = [batch_X[k] for k in K]
                batch_X_K = self.augmentor.augment(batch_X_K)
                for i, k in enumerate(K):
                    batch_X[k] = batch_X_K[i]
                batch_X_K = None
        # Return batch
        return batch_X, batch_y

    def on_epoch_end(self):
        """
        Logic to handle the dataset between epochs. For example, it can be used
        to random shuffle the indices of the input data, so they are given
        in a different order at each epoch.

        :return: Nothing at all, but the internal state of the sequencer is
            updated.
        """
        # Random index shuffling
        if self.random_shuffle_indices:
            if isinstance(self.X, list):
                tensors_per_pcloud = len(self.X)  # Tensors per input pcloud
                if self.Irandom is None:  # First random shuffle of indices
                    self.init_random_indices()
                else:  # After the first random shuffle of indices
                    # Undo previous shuffle
                    for i in range(tensors_per_pcloud):
                        self.X[i][self.Irandom] = np.array(self.X[i])
                    self.y[self.Irandom] = np.array(self.y)
            else:
                if self.Irandom is None:  # First random shuffle of indices
                    self.init_random_indices()
                else:  # After the first random shuffle of indices
                    # Undo previous shuffle
                    self.X[self.Irandom] = np.array(self.X)
                    self.y[self.Irandom] = np.array(self.y)
            np.random.shuffle(self.Irandom)  # Shuffle indices
            self.shuffled = False  # Flag to shuffle on first __getitem__ call

    # ---  BATCH EXTRACTION METHODS  --- #
    # ---------------------------------- #
    def extract_input_batch(self, start_idx, end_idx):
        """
        Extract the input batch inside the given indexing interval.

        :param start_idx: Start point of the indexing interval (inclusive).
        :param end_idx: End point of the indexing interval (exclusive).
        :return: The extracted input batch inside the indexing interval.
        :rtype: list or :class:`np.ndarray`
        """
        if isinstance(self.X, list):
            return [np.array(Xi[start_idx:end_idx]) for Xi in self.X]
        else:
            return np.array(self.X[start_idx:end_idx])

    def extract_reference_batch(self, start_idx, end_idx):
        """
        Extract the reference batch inside the given indexing interval.

        :param start_idx: Start point of the indexing interval (inclusive).
        :param end_idx: End point of the indexing interval (exclusive).
        :return: The extracted reference batch inside the indexing interval.
        :rtype: :class:`np.ndarray`
        """
        return np.array(self.y[start_idx:end_idx])

    def find_augmentation_elements(self):
        """
        Find the indices of the elements in the batch that must be considered
        for data augmentation. Note that these indices will depend on the
        underlying neural network architecture.

        :return: List of indices representing the elements that must be
            considered for data augmentation.
        :rtype: list of int
        """
        if isinstance(self.arch, ConvAutoencPwiseClassif):
            return [0] + [2+i for i in range(self.arch.max_depth-1)]
        elif isinstance(self.arch, PointNet):
            return [0]
        elif isinstance(self.arch, RBFNet):
            return [0]
        else:
            raise DeepLearningException(
                'DLSequencer does not support data augmentation for the '
                f'neural network architecture {self.arch.__class__.__name__}.'
            )

    # ---  RANDOM INDEXING METHODS  --- #
    # --------------------------------- #
    def init_random_indices(self):
        # Determine int type
        m = self.X[0].shape[0]  # Number of input point clouds
        int_type = np.uint64
        if m <= 256:
            int_type = np.uint8
        elif m <= 65536:
            int_type = np.uint16
        elif m<= 4294967296:
            int_type = np.uint32
        # Initialize random indices
        self.Irandom = np.arange(  # Index for each input pcloud
            m, dtype=int_type
        )
