import tensorflow as tf


def enforce_4D(image, dtype=tf.float32):
    """Enforce an image shape [batch, x, y, channels]"""
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image)
    ndims = image.get_shape().ndims
    if ndims == 2:
        image = image[None, :, :, None]
    elif ndims == 3 and image.shape[0] == 1:
        image = image[:, :, :, None]
    elif ndims == 3:
        image = image[None, :, :, :]
    elif ndims != 4:
        raise ValueError('Unknown image shape: {}'.format(image.shape))
    return tf.cast(image, dtype)
