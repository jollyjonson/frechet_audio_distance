import tensorflow as tf


class Statistics:
    def __init__(self, dim: int):
        self.covariance = tf.Variable(tf.zeros((dim, dim), dtype=tf.float64))
        self.mean = tf.Variable(tf.zeros((dim,), dtype=tf.float64))
        self.num_items_processed = tf.Variable(0.0, dtype=tf.float64)

    def update(self, data: tf.Tensor) -> None:  # pragma: no cover
        """
        Updates the means and covariances held by an instance of this class
        """
        data = tf.cast(data, dtype=tf.float64)
        num_items_this_update = tf.cast(tf.shape(data)[0], dtype=tf.float64)
        self.num_items_processed.assign_add(
            tf.cast(num_items_this_update, dtype=tf.float64)
        )

        x_norm_old = data - self.mean
        self.mean.assign_add(
            tf.reduce_sum(x_norm_old, axis=0) / self.num_items_processed
        )
        x_norm_new = data - self.mean

        self.covariance.assign(
            self.covariance
            * (self.num_items_processed - num_items_this_update)
            / self.num_items_processed
        )
        self.covariance.assign_add(
            tf.matmul(tf.transpose(x_norm_old), x_norm_new)
            / self.num_items_processed
        )
