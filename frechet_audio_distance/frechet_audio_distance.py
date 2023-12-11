from typing import Any, Dict, Optional

import tensorflow as tf
import tensorflow_io as tfio

try:
    from fad_statistics import Statistics
    from feature import FADFeature
    from vggish import VGGish
except ModuleNotFoundError:
    from .fad_statistics import Statistics
    from .feature import FADFeature
    from .vggish import VGGish


class FrechetAudioDistance(tf.keras.metrics.Metric):
    """
    Keras metric to compute the Frechet Audio Distance (FAD).
    Expects time-domain inputs at the sample rate given to
    the init of this class of shape `(batch_size, num_samples, num_channels)`.

    Parameters
    ----------
    sample_rate: float
        The sample rate of the input in Hz
    downmix_to_mono: bool = False
        Whether or not to downmix multi-channel input before computing the
        metric. If false, all channels will be treated as separate inputs.
    step_size_in_s: float = .5
    name: Optional[str] = None
    kwargs: Dict[Any, Any]

    Notes
    -----
    - The authors of [1] recommend to use at the very least ~25min of
      material to compute accurate FAD scores [1, App. B].

    References
    ----------
    [1] Kilgour, K., Zuluaga, M., Roblek, D., & Sharifi, M. (2019).
        Fréchet Audio Distance: A Reference-Free
        Metric for Evaluating Music Enhancement Algorithms.
        In INTERSPEECH (pp. 2350-2354).
    """

    _true_statistics: Statistics
    _pred_statistics: Statistics

    internal_sample_rate_in_hz: float = 16000
    default_name = "frechet_audio_distance"

    def __init__(
        self,
        sample_rate: float,
        downmix_to_mono: bool = False,
        name: Optional[str] = None,
        feature: FADFeature = VGGish(),
        **kwargs: Dict[Any, Any]
    ):
        super().__init__(
            name=self.default_name or name, dtype=tf.float32, **kwargs
        )

        self._sample_rate = sample_rate
        self._downmix_to_mono = downmix_to_mono
        self._feature = feature

        self.reset_state()

        # needed to be able to switch the metric off during training and only
        # run during eval and val in model.fit
        self.switch = tf.Variable(True)

    def update_state(
        self,
        y_true: tf.Tensor | None,
        y_pred: tf.Tensor | None,
        *args,
        **kwargs
    ) -> None:  # pragma: no cover
        if self.switch:
            for data, associated_statistics in zip(
                [y_true, y_pred],
                [self._true_statistics, self._pred_statistics],
            ):
                if data is not None:
                    data = tf.cast(data, tf.float32)
                    if self._sample_rate != self.internal_sample_rate_in_hz:
                        data = self._resample_to_internal_sample_rate(data)
                    data = self._possibly_add_batch_dim_and_reduce_channel_dim(
                        data
                    )
                    data_embedding = self._feature(data)
                    associated_statistics.update(data_embedding)

    def result(self) -> tf.Tensor:  # pragma: no cover
        return self._compute_distance()

    def reset_state(self) -> None:
        self._true_statistics = Statistics(self._feature.output_dim)
        self._pred_statistics = Statistics(self._feature.output_dim)

    def merge_state(self, *args, **kwargs):
        raise NotImplementedError(
            "Merging states for distributed operation is not yet implemented"
        )

    def _possibly_add_batch_dim_and_reduce_channel_dim(
        self, audio: tf.Tensor
    ) -> tf.Tensor:  # pragma: no cover
        """
        Add a batch dim, if the data has none.
        Either downmix to mono or rearrange the different channels in the
        batch dimension.
        """
        if tf.executing_eagerly():
            if len(audio.shape) == 2:
                audio = tf.expand_dims(audio, axis=0)
        if self._downmix_to_mono:
            return tf.reduce_mean(audio, axis=-1)
        else:
            return tf.reshape(
                audio,
                (tf.shape(audio)[0] * tf.shape(audio)[-1], tf.shape(audio)[1]),
            )

    def _resample_to_internal_sample_rate(self, audio: tf.Tensor) -> tf.Tensor:
        return tf.cast(
            tfio.audio.resample(
                tf.cast(audio, tf.float32),
                self._sample_rate,
                self.internal_sample_rate_in_hz,
            ),
            dtype=tf.float32,
        )

    @staticmethod
    def _stable_trace_sqrt_product(
        covariance_true: tf.Tensor, covariance_pred: tf.Tensor
    ) -> tf.Tensor:  # pragma: no cover
        """
        Avoids some problems when computing the srqt of product of sigmas.
        Based on Dougal J. Sutherland's contribution here:
        https://github.com/bioinf-jku/TTUR/blob/master/fid.py

        Raises
        ------
          ValueError: If the sqrt of the product of the sigmas contains complex
              numbers with large imaginary parts (>1e-3).
        """
        # product might be almost singular
        sqrt_product = tf.linalg.sqrtm(
            tf.matmul(covariance_true, covariance_pred)
        )

        if not tf.reduce_all(tf.math.is_finite(sqrt_product)):
            offset = (
                tf.eye(sqrt_product.shape[0], dtype=tf.float64) * 1e-7
            )  # add eps to the diagonal to avoid a singular product.
            sqrt_product = tf.linalg.sqrtm(
                tf.matmul(covariance_true + offset, covariance_pred + offset)
            )

        # Might have a slight imaginary component.
        # if not tf.reduce_max(
        #                   tf.math.imag(tf.linalg.diag(sqrt_product))) < 1e-3:
        #    raise ValueError('sqrt_product contains large complex numbers.')
        sqrt_product = tf.math.real(sqrt_product)

        return tf.linalg.trace(sqrt_product)

    def _compute_distance(self) -> tf.Tensor:  # pragma: no cover
        """Compute the FAD from acquired means and covariances"""
        trace_sqrt_product = self._stable_trace_sqrt_product(
            self._true_statistics.covariance, self._pred_statistics.covariance
        )
        mean_difference = (
            self._true_statistics.mean - self._pred_statistics.mean
        )

        return (
            tf.tensordot(mean_difference, mean_difference, axes=1)
            + tf.linalg.trace(self._true_statistics.covariance)
            + tf.linalg.trace(self._pred_statistics.covariance)
            - 2.0 * trace_sqrt_product
        )


class RunFrechetAudioDistanceOnlyOnValidationAndTestCallback(
    tf.keras.callbacks.Callback
):
    """
    On test begin (i.e. when evaluate() is called or validation data is
    run during fit()) toggle metric flag
    """

    def _set_fad_switch(self, switch_value: bool) -> None:
        for metric in self.model.metrics:
            if isinstance(metric, FrechetAudioDistance):
                metric.switch.assign(switch_value)

    def on_epoch_begin(self, epoch, logs: Any = None) -> None:
        self._set_fad_switch(False)
        super().on_epoch_begin(epoch, logs)

    def on_test_begin(self, logs: Any = None) -> None:
        self._set_fad_switch(True)
        super().on_test_begin(logs)

    def on_test_end(self, logs: Any = None) -> None:
        self._set_fad_switch(False)
        super().on_test_end(logs)
