import tensorflow as tf

def vl3d_class_weighted_categorical_crossentropy(class_weight):
    r"""
    Function to compute a weighted categorical cross-entropy loss.

    # TODO Rethink : Add more documentation (see the binary case)

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
