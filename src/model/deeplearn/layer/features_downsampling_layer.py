# ---   IMPORTS   --- #
# ------------------- #
from src.model.deeplearn.deep_learning_exception import DeepLearningException
from src.model.deeplearn.layer.layer import Layer
import tensorflow as tf


# ---   CLASS   --- #
# ----------------- #
class FeaturesDownsamplingLayer(Layer):
    r"""
    :author: Alberto M. Esmoris Pena

    A features downsampling layer receives batches of :math:`m` points with
    :math:`n_f` features each and downsamples them to :math:`R` points with
    :math:`n_f` features each.

    It can use a mean-based filter to reduce the features (
    see :meth:`FeaturesDownsamplingLayer.mean_filter`) or a Gaussian-like
    filter to reduce the features considering the distances between the
    points (see :meth:`FeaturesDownsamplingLayer.gaussian_filter`).
    Alternatively, a simpler nearest-neighbor filter can be used too
    (see :meth:`FeaturesDownsamplingLayer.nearest_filter`).

    The input feature space is a tensor
    :math:`\mathcal{F} \in \mathbb{R}^{K \times m \times n_f}` whose slices
    represent independent receptive fields. The output feature space is a
    downsampled version of the input one
    :math:`\mathcal{Y} \in \mathbb{R}^{K \times R \times n_f}`. Where
    :math:`K` is the batch size.

    :ivar filter: The name of the filter to be used. Either ``"mean"``,
        ``"gaussian"``, or ``"nearest"``.
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
        r"""
        Downsample the features from :math:`m` input points to :math:`R`
        output points, where :math:`R \leq m`.

        :param inputs: The inputs such that:

            -- inputs[0]
                is the structure space tensor before the downsampling.

                .. math::
                    \mathcal{X}_a \in \mathbb{R}^{K \times m \times n_x}

            -- inputs[1]
                is the structure space tensor after the downsampling.

                .. math::
                    \mathcal{X}_b \in \mathbb{R}^{K \times R \times n_x}

            -- inputs[2]
                is the feature space tensor before the downsampling.

                .. math::
                    \mathcal{F} \in \mathbb{R}^{K \times m \times n_f}

            -- inputs[3]
                is the indexing tensor representing the downsampling topology
                :math:`\mathcal{N}^D \in \mathbb{Z}^{K \times R \times n_n}`,
                i.e., for each point in the downsampled space to what points
                it is connected in the non-downsampled space.

        :return: The downsampled features
            :math:`\mathcal{Y} \in \mathbb{R}^{K \times R \times n_f}`.
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
    def mean_filter(Xa, Xb, Fin, ND):
        r"""
        .. math::
            y_{kij} = \dfrac{1}{n_n} \sum_{p=1}^{n_n}{
                f_{kn_{kip}^Dj}
            }

        :return: :math:`\mathcal{Y} \in \mathbb{R}^{K \times R \times n_f}`
        """
        Fout = FeaturesDownsamplingLayer.gather_input_features(Fin, ND)
        Fout = tf.reduce_mean(Fout, axis=2)  # Output features from mean
        return Fout

    @staticmethod
    def gaussian_filter(Xa, Xb, Fin, ND):
        r"""
        .. math::
            y_{kij} = \left(\sum_{p=1}^{n_n}{g_{kip}}\right)^{-1}
                \sum_{p=1}^{n_n}{
                    g_{kip}
                    f_{kn_{kip}^Dj}
                }

        Where:

        .. math::
            g_{kip} = \exp\left(\dfrac{
                \lVert
                    (\mathcal{X}_a)_{kn_{kip}^D*} -
                    (\mathcal{X}_b)_{ki*}
                \rVert^2
            }{
                (d_{ki}^*)^2
            }\right)

        And:

        .. math::
            d_{ki}^* = \max_{1 \leq p \leq n_n} \; {
                \lVert
                    (\mathcal{X}_a)_{kn_{kip}^D*} -
                    (\mathcal{X}_b)_{ki*}
                \rVert
            }


        :return: :math:`\mathcal{Y} \in \mathbb{R}^{K \times R \times n_f}`
        """
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

    @staticmethod
    def nearest_filter(Xa, Xb, Fin, ND):
        r"""
        .. math::
            y_{kij} = f_{kn_{ip^*}^Dj}

        Where:

        .. math::
            p^* = \operatorname*{argmin}_{1 \leq p \leq n_n} \; { \lVert
                (\mathcal{X}_a)_{kn_{ip}^D*} -
                (\mathcal{X}_b)_{ki*}
            } \rVert^2

        :return: :math:`\mathcal{Y} \in \mathbb{R}^{K \times R \times n_f}`
        """
        # Gather input feature from nearest neighbor
        Fout = FeaturesDownsamplingLayer.gather_input_features(Fin, ND)
        return Fout[:, :, 0, :]

    # ---   UTIL METHODS   --- #
    # ------------------------ #
    @staticmethod
    def compute_squared_distances(Xa, Xb, ND):
        r"""
        Compute the squared distances between the downsampled points and those
        points in the non-downsampled space topologically connected to them
        (typically because they are the nearest neighbors).

        Let :math:`\pmb{X}_a \in \mathbb{R}^{m \times n_x}` be a structure
        space matrix (i.e., a matrix of point-wise coordinates) representing
        the points before the downsampling and
        :math:`\pmb{X}_b \in \mathbb{R}^{R \times n_x}` be a structure space
        matrix representing the points after the downsampling. For then,
        the :math:`\pmb{N}^D \in \mathbb{Z}^{R \times n_n}` indexing matrix
        represents the indices of the neighbors for each point in the
        downsampled space with respect to the non-downsampled space.

        Considering all the previous matrices (that can be seen as the slices
        of the input tensors), it is possible to compute the distance
        matrix :math:`\pmb{D} \in \mathbb{R}^{R \times n_n}`. In this matrix,
        each element is given by the following expression:

        .. math::
            d_{ij} = \lVert{
                (\pmb{X}_b)_{i*} - (\pmb{X}_a)_{n^D_{ij}*}
            }\rVert


        Note that the squared distances matrix is computed instead, which is
        the same as :math:`\pmb{D}` but considering the element-wise squares.

        :param Xa: The structure space before downsampling.
        :param Xb: The structure space after downsampling.
        :param ND: The indexing tensor whose slices are indexing matrices.
        :return: The tensor with the squared distances for each downsampled
            point with respect to its neighbors in the non-downsampled
            space.
        :rtype: :class:`tf.Tensor`
        """
        XaND = tf.gather(Xa, ND, axis=1, batch_dims=1)
        Xdiff = tf.transpose(tf.transpose(XaND, [2, 0, 1, 3])-Xb, [1, 2, 0, 3])
        Xsd = tf.reduce_sum(tf.square(Xdiff), axis=3)  # Squared differences
        return Xsd

    @staticmethod
    def gather_input_features(Fin, ND):
        r"""
        Gather the input features corresponding to given indexing tensor.

        Let :math:`\mathcal{N}^D \in \mathbb{Z}^{K \times R \times n_n}`
        be an indexing tensor of :math:`K` batches of :math:`R` points with
        :math:`n_n` neighbors each. Also, let
        :math:`\mathcal{F} \in \mathbb{R}^{K \times R \times n_f}` be a
        features tensor of :math:`K` batches of :math:`m` points with
        :math:`n_f` features each.

        The resulting tensor of gathered input features can be seen as a tensor
        :math:`\mathcal{Y} \in \mathbb{R}^{K \times R \times n_n \times n_f}`
        of :math:`K` batches of :math:`R` points each such that for each point
        :math:`n_f` features are gathered for each of its :math:`n_n`
        neighbors. Note that, :math:`R \leq m` because :math:`\mathcal{N}^D`
        is an indexing tensor meant to be used for downsampling purposes.


        :param Fin: The input features.
        :param ND: The indexing tensor whose slices are indexing matrices.
        :return: Tensor of gathered input features.
        :rtype: :class:`tf.Tensor`
        """
        return tf.gather(Fin, ND, axis=1, batch_dims=1)

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
