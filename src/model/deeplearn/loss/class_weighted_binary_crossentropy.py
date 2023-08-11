import tensorflow as tf


def vl3d_class_weighted_binary_crossentropy(class_weight):
    r"""
    Function to compute a weighted binary crossentropy loss.

    Let :math:`\mathcal{L}(\pmb{y}, \pmb{\hat{y}}) \in \mathbb{R}^{m}` be a
    binary crossentropy loss on :math:`m` samples. Now, let
    :math:`\pmb{w} \in \mathbb{R}^2` be a vector of class weights for
    binary classification, i.e., two classes. Thus, it is possible to
    define a vector
    :math:`\pmb{\omega} \in \mathbb{R}^{m}` such that
    :math:`\pmb{\omega}_{i} = y_i w_2 + (1-y_i) w_1`, where any :math:`y_i`
    must be either zero or one. For then, the class weighted binary
    crossentropy can be obtained simply by computing the following Hadamard
    Product (where :math:`\pmb{\hat{y}}` is the vector of binary predictions):

    ..  math::
        \mathcal{L}(\pmb{y}, \pmb{\hat{y}}) \odot \pmb{\omega}

    :param class_weight: The vector of class weights. The component i of this
        vector (:math:`\pmb{w}`) is the weight for class i.
    :return: The weighted binary cross entropy loss.
    """

    def _vl3d_class_weighted_binary_crossentropy(y_true, y_pred):
        # Baseline binary cross entropy
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        # Compute vector of class weights
        cw = y_true * class_weight[1] + (1.0 - y_true) * class_weight[0]
        # Compute weighted binary cross entropy
        wbce = bce * cw
        # Return mean weighted binary cross entropy
        return tf.keras.backend.mean(wbce)

    return _vl3d_class_weighted_binary_crossentropy


