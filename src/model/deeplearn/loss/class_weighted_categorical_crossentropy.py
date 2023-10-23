import tensorflow as tf

def vl3d_class_weighted_categorical_crossentropy(class_weight):
    r"""
    Function to compute a weighted categorical cross-entropy loss.

    Let :math:`\mathcal{L}(\pmb{y}, \pmb{\hat{y}}) \in \mathbb{R}^{m}` be a
    categorical crossentropy loss on :math:`m` samples.  Now, let
    :math:`\pmb{w} \in \mathbb{R}^n` be a vector of class weights for
    multiclass classification, i.e., potentially many classes. Thus, it is
    possible to define a vector
    :math:`\pmb{\omega} \in \mathbb{R}^{m}` such that
    :math:`\omega_i = \langle{\pmb{w}, \pmb{y}}\rangle`, where any :math:`y_j`
    must be either zero or one for :math:`j=1,\ldots,n`. For then, the class
    weighted categorical crossentropy can be obtained simply by computing the
    following Hadamard Product (where :math:`\pmb{\hat{y}}` is the vector of
    one-hot-encoded multiclass predictions).

    .. math::
        \mathcal{L}(\pmb{y}, \pmb{\hat{y}}) \odot \pmb{\omega}

    :param class_weight: The vector of class weights. The component i of this
        vector (:math:`\pmb{w}`) is the weight for class i.
    :return: The weighted categorical cross entropy loss
    """

    def _vl3d_class_weighted_categorical_crossentropy(y_true, y_pred):
        # Baseline categorical cross entropy
        cce = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
        # Compute class weights
        cw = tf.linalg.matvec(tf.cast(y_true, dtype=tf.float32), class_weight)
        # Compute weighted categorical cross entropy
        wcce = cce * cw
        # Return mean weighted categorical cross entropy
        return tf.keras.backend.mean(wcce)

    return _vl3d_class_weighted_categorical_crossentropy
