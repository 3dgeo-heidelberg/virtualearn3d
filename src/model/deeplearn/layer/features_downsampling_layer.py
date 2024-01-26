# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
import src.main.main_logger as LOGGING
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class FeaturesDownsamplingLayer(Layer):
    # TODO Rethink : Doc
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self, filter='mean', **kwargs):
        """
        See :class:`.Layer` and :meth:`layer.Layer.__init__`.
        """
        # Call parent's init
        super().__init__(**kwargs)
        # Assign attributes
        self.filter = filter  # Either mean or gaussian
        filter_low = self.filter.lower()
        if filter_low == 'mean':
            self.filter_f = self.mean_filter
        elif filter_low == 'gaussian':
            self.filter_f = self.gaussian_filter
        else:
            raise DeepLearningException(
                'FeaturesDownsamplingLayer cannot be built for requested '
                f'filter: "{self.filter}"'
            )

    # ---   LAYER METHODS   --- #
    # ------------------------- #
    def build(self, dim_in):
        """
        See :class:`.Layer` and :meth:`layer.Layer.build`.
        """
        # Call parent's build
        super().build(dim_in)

    def call(self, inputs, training=False, mask=False):
        # TODO Rethink : Doc
        # Extract input
        Xa = inputs[0]
        Xb = inputs[1]
        Fin = inputs[2]
        ND = inputs[3]
        return self.filter_f(Xa, Xb, Fin, ND)

    # ---  DOWNSAMPLING FILTERS  --- #
    # ------------------------------ #
    @staticmethod
    def mean_filter(Xa, Xb, Fin, ND):
        # TODO Rethink : Doc
        Fout = FeaturesDownsamplingLayer.gather_input_features(Fin, ND)
        Fout = tf.reduce_mean(Fout, axis=2)  # Output features from mean
        return Fout

    @staticmethod
    def gaussian_filter(Xa, Xb, Fin, ND):
        # TODO Rethink : Doc
        D_sq = FeaturesDownsamplingLayer.compute_squared_distances(Xa, Xb, ND)
        omega_sq = tf.reduce_max(D_sq, axis=2)  # The kernels' squared lengths
        gaussians = tf.exp(D_sq/tf.expand_dims(omega_sq, 2))
        gaussian_norms = tf.reduce_sum(gaussians, axis=2)
        Fout = FeaturesDownsamplingLayer.gather_input_features(Fin, ND)
        Fout = tf.transpose(  # Gaussian x F_downsampmled
            gaussians * tf.transpose(Fout, [3, 0, 1, 2]),
            [1, 2, 3, 0]
        )
        Fout = tf.reduce_sum(Fout, axis=2)  # Output features
        Fout = tf.transpose(  # Normalized output features
            tf.transpose(Fout, [2, 0, 1])/gaussian_norms,
            [1, 2, 0]
        )
        return Fout

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def compute_squared_distances(Xa, Xb, ND):
        # TODO Rethink : Doc
        XaND = tf.gather(Xa, ND, axis=1, batch_dims=1)
        Xdiff = tf.transpose(tf.transpose(XaND, [2, 0, 1, 3])-Xb, [1, 2, 0, 3])
        Xsd = tf.reduce_sum(tf.square(Xdiff), axis=3)  # Squared differences
        return Xsd

    @staticmethod
    def gather_input_features(Fin, ND):
        # TODO Rethink : Doc
        return tf.gather(Fin, ND, axis=1, batch_dims=1)


    # TODO Rethink : Implement serialization
