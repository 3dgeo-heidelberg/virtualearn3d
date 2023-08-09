import tensorflow as tf

# TODO Rethink : Doc (and check sphinx handles functions outside classes well)
def vl3d_class_weighted_binary_crossentropy(class_weight):

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


