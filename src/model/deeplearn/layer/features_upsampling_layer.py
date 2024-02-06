# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
from src.model.deeplearn.layer.features_downsampling_layer import \
    FeaturesDownsamplingLayer
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class FeaturesUpsamplingLayer(Layer):
    r"""
    :author: Alberto M. Esmoris Pena

    A features upsampling layer receives batches of :math:`R` points with
    :math:`n_f` features each and upsamples them to :math:`m` points with
    :math:`n_f` features each.

    It can use a mean-based filter to propagate the features (
    see :meth:`FeaturesUpsamplingLayer.mean_filter`) or a Gaussian-like
    filter to propagate the features considering the distances
    between the points (see :meth:`FeaturesUpsamplingLayer.gaussian_filter`).
    Alternatively, a simpler nearest-neighbor filter can be used too
    (see :meth:`FeaturesUpsamplingLayer.nearest_filter`).

    The input feature space is a tensor
    :math:`\mathcal{F} \in \mathbb{R}^{K \times R \times n_f}` whose slices
    represent independent receptive fields. The output feature space is an
    upsampled version of the input one
    :math:`\mathcal{Y} \in \mathbb{R}^{K \times m \times n_f}`. Where
    :math:`K` is the batch size.

    :ivar filter: The name of the filter to be used. Either ``"mean"``,
        ``"gaussian"``, or ``"nreaest"``.
    :vartype filter: str
    :ivar filter_f: The method to be used for filtering, derived from the
        ``filter`` attribute.
    :vartype filter_f: Callable
    """
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
        elif filter_low == 'nearest':
            self.filter_f = self.nearest_filter
        else:
            raise DeepLearningException(
                'FeaturesUpsamplingLayer cannot be built for requested '
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
        r"""
        Upsample the features from :math:`R` input points to :math:`m` output
        points, where :math:`m \geq R`.

        :param inputs: The inputs such that:

            -- inputs[0]
                is the structure space tensor before the upsampling.

                .. math::
                    \mathcal{X}_a \in \mathbb{R}^{K \times R \times n_x}

            -- inputs[1]
                is the structure space tensor after the upsampling.

                .. math::
                    \mathcal{X}_b \in \mathbb{R}^{K \times m \times n_x}

            -- inputs[2]
                is the feature space tensor before the upsampling.

                .. math::
                    \mathcal{F} \in \mathbb{R}^{K \times R \times n_f}

            -- inputs[3]
                is the indexing tensor representing the upsampling topology
                :math:`\mathcal{N}^U \in \mathbb{Z}^{K \times m \times n_n}`,
                i.e., for each point in the upsampled space to what points
                it is connected in the non-upsampled space.

        :return: The upsampled features
            :math:`\mathcal{Y} \in \mathbb{R}^{K \times m \times n_f}`
        """
        # Extract input
        Xa = inputs[0]
        Xb = inputs[1]
        Fin = inputs[2]
        ND = inputs[3]
        return self.filter_f(Xa, Xb, Fin, ND)

    # ---  DOWNSAMPLING FILTERS  --- #
    # ------------------------------ #
    @staticmethod
    def mean_filter(Xa, Xb, Fin, NU):
        r"""
        .. math::
            y_{kij} = \dfrac{1}{n_n} \sum_{p=1}^{n_n}{
                f_{kn_{kip}^Uj}
            }

        :return: :math:`\mathcal{Y} \in \mathbb{R}^{K \times m \times n_f}`
        """
        Fout = FeaturesDownsamplingLayer.gather_input_features(Fin, NU)
        Fout = tf.reduce_mean(Fout, axis=2)  # Output features from mean
        return Fout

    @staticmethod
    def gaussian_filter(Xa, Xb, Fin, NU):
        r"""
        .. math::
            y_{kij} = \left(\sum_{p=1}^{n_n}{g_{kip}}\right)^{-1}
                \sum_{p=1}^{n_n}{
                    g_{kip}
                    f_{kn_{kip}^Uj}
                }

        Where:

        .. math::
            g_{kip} = \exp\left(\dfrac{
                \lVert
                    (\mathcal{X}_a)_{kn_{kip}^U*} -
                    (\mathcal{X}_b)_{ki*}
                \rVert^2
            }{
                (d_{ki}^*)^2
            }\right)

        And:

        .. math::
            d_{ki}^* = \max_{1 \leq p \leq n_n} \; {
                \lVert
                    (\mathcal{X}_a)_{kn_{kip}^U*} -
                    (\mathcal{X}_b)_{ki*}
                \rVert
            }


        :return: :math:`\mathcal{Y} \in \mathbb{R}^{K \times m \times n_f}`
        """
        D_sq = FeaturesDownsamplingLayer.compute_squared_distances(Xa, Xb, NU)
        omega_sq = tf.reduce_max(D_sq, axis=2)  # The kernels' squared lengths
        gaussians = tf.exp(D_sq/tf.expand_dims(omega_sq, 2))
        gaussian_norms = tf.reduce_sum(gaussians, axis=2)
        Fout = FeaturesDownsamplingLayer.gather_input_features(Fin, NU)
        Fout = tf.transpose(  # Gaussian x F_upsampled
            gaussians * tf.transpose(Fout, [3, 0, 1, 2]),
            [1, 2, 3, 0]
        )
        Fout = tf.reduce_sum(Fout, axis=2)  # Output features
        Fout = tf.transpose(  # Normalized output features
            tf.transpose(Fout, [2, 0, 1])/gaussian_norms,
            [1, 2, 0]
        )
        return Fout

    @staticmethod
    def nearest_filter(Xa, Xb, Fin, NU):
        r"""
        .. math::
            y_{kij} = f_{kn_{ip^*}^Uj}

        Where:

        .. math::
            p^* = \operatorname*{argmin}_{1 \leq p \leq n_n} \; { \lVert
                (\mathcal{X}_a)_{kn_{ip}^U*} -
                (\mathcal{X}_b)_{ki*}
            } \rVert^2

        :return: :math:`\mathcal{Y} \in \mathbb{R}^{K \times m \times n_f}`
        """
        # Gather input feature from nearest neighbor
        Fout = FeaturesDownsamplingLayer.gather_input_features(Fin, NU)
        return Fout[:, :, 0, :]

    # ---   SERIALIZATION   --- #
    # ------------------------- #
    def get_config(self):
        """Return necessary data to serialize the layer"""
        # Call parent's get_config
        config = super().get_config()
        # Update config with custom attributes
        config.update({
            # Base attributes
            'filter': self.filter
        })
        # Return updated config
        return config

    @classmethod
    def from_config(cls, config):
        """Use given config data to deserialize the layer"""
        fsl = cls(**config)
        # Return deserialized layer
        return fsl
