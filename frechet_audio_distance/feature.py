from abc import ABCMeta, abstractmethod

import tensorflow as tf


class FADFeature(metaclass=ABCMeta):
    """
    Interface for a Feature extractor. Implements a mapping from time-domain
    audio signals to some sort of feature space in which the empirical frechet
    distance is computed. The original publication used a 'VGGish' model, the
    corresponding implementation can be found in `vggish.py`.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Returns the output dimensionality of the feature. E.g. when given
        audio data of dimensionality (batch_size x num_samples), the
        implementation will return a feature of
        (batch_size x num_features x output_dim) on __call__.
        """

    @property
    @abstractmethod
    def input_sample_rate_in_hz(self) -> float:
        """
        Returns the sample rate needed for the feature computation in Hz.
        E.g. the VGGish feature proposed by the original FAD authors
        expects its input audio sampled at 16kHz.
        """

    @abstractmethod
    def __call__(self, audio: tf.Tensor) -> tf.Tensor:
        """
        Compute the feature. Implementations should be decorated with
        `tf.function` for performance. The input has the shape
        (batch_size x num_audio_samples).

        Parameters
        ----------
        audio: tf.Tensor
            A batch of audio snippets from which to compute features. Will have
            dimensionality (batch_size x num_samples). Always assu

        Returns
        -------
        feature: tf.Tensor
            The feature implemented by the feature extractor. Must have
            dimensionality (batch_size x num_features x output_dim)
        """
