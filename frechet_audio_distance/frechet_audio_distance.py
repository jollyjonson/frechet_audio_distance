import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from .vggish_params import VGGishParams


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

    class _Statistics:
        def __init__(self, dim: int):
            self.covariance = tf.Variable(
                tf.zeros((dim, dim), dtype=tf.float64)
            )
            self.mean = tf.Variable(tf.zeros((dim,), dtype=tf.float64))

    _statistics_dim: int = VGGishParams.EMBEDDING_SIZE
    _num_items_processed: tf.Variable  # dtype = tf.float64
    _true_statistics: _Statistics
    _pred_statistics: _Statistics

    internal_sample_rate_in_hz: float = 16000
    default_name = "frechet_audio_distance"

    def __init__(
        self,
        sample_rate: float,
        downmix_to_mono: bool = False,
        step_size_in_s: float = 0.5,
        name: Optional[str] = None,
        **kwargs: Dict[Any, Any]
    ):
        super().__init__(
            name=self.default_name or name, dtype=tf.float32, **kwargs
        )

        self._sample_rate = sample_rate
        self._downmix_to_mono = downmix_to_mono
        self._step_size_in_samples = int(
            round(step_size_in_s * self.internal_sample_rate_in_hz)
        )
        self._vggish_model: tf.keras.Model = self._init_vggish_model()

        self.reset_state()

        # needed to be able to switch the metric off during training and only
        # run during eval and val in model.fit
        self.switch = tf.Variable(True)

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, *args, **kwargs
    ) -> None:  # pragma: no cover
        if self.switch:
            y_true, y_pred = map(
                lambda x: tf.cast(x, tf.float32), [y_true, y_pred]
            )
            if self._sample_rate != self.internal_sample_rate_in_hz:
                y_true, y_pred = map(
                    self._resample_to_internal_sample_rate, [y_true, y_pred]
                )
            y_true, y_pred = map(
                self._possibly_add_batch_dim_and_reduce_channel_dim,
                [y_true, y_pred],
            )
            y_true_features, y_pred_features = map(
                self._extract_mel_features, [y_true, y_pred]
            )
            y_true_embedding, y_pred_embedding = map(
                self._vggish_model, [y_true_features, y_pred_features]
            )
            self._num_items_processed.assign_add(
                tf.cast(tf.shape(y_true_embedding)[0], dtype=tf.float64)
            )
            for data, statistics in zip(
                [y_true_embedding, y_pred_embedding],
                [self._true_statistics, self._pred_statistics],
            ):
                self._update_statistics(
                    statistics, data, self._num_items_processed
                )

    def result(self) -> tf.Tensor:  # pragma: no cover
        return self._compute_distance()

    def reset_state(self) -> None:
        self._num_items_processed = tf.Variable(0, dtype=tf.float64)
        self._true_statistics = self._Statistics(self._statistics_dim)
        self._pred_statistics = self._Statistics(self._statistics_dim)

    def merge_state(self, *args, **kwargs):
        raise NotImplementedError(
            "Merging states for distributed operation is not yet implemented"
        )

    def _extract_mel_features(self, audio_batch: tf.Tensor) -> tf.Tensor:
        normalized_audio_batch = self._normalize_audio(audio_batch)
        framed_audio = tf.signal.frame(
            normalized_audio_batch,
            VGGishParams.SAMPLE_RATE_IN_HZ,
            self._step_size_in_samples,
        )
        batched_framed_audio = tf.reshape(
            framed_audio,
            (
                tf.shape(framed_audio)[0] * tf.shape(framed_audio)[1],
                tf.shape(framed_audio)[2],
            ),
        )
        return tf.map_fn(self._log_mel_spectrogram, batched_framed_audio)

    @staticmethod
    def _normalize_audio(audio_batch: tf.Tensor) -> tf.Tensor:
        min_ratio_for_normalization = tf.convert_to_tensor(
            0.1, dtype=audio_batch.dtype
        )  # = 10**(max_db/-20) with max_db = 20
        normalization_coeff = tf.maximum(
            min_ratio_for_normalization,
            tf.reduce_max(audio_batch, axis=-1, keepdims=True),
        )
        return audio_batch / normalization_coeff

    @staticmethod
    def _stabilized_log(
        x: tf.Tensor, additive_offset: float, floor: float
    ) -> tf.Tensor:  # pragma: no cover
        """TF version of mfcc_mel.StabilizedLog."""
        return tf.math.log(tf.math.maximum(x, floor) + additive_offset)

    # spectrogram params as given by FAD paper
    _num_mel_bins = 64
    _log_additive_offset = 0.001
    _log_floor = 1e-12
    _window_length_secs = 0.025
    _hop_length_secs = 0.010
    _window_length_samples = int(
        round(internal_sample_rate_in_hz * _window_length_secs)
    )
    _hop_length_samples = int(
        round(internal_sample_rate_in_hz * _hop_length_secs)
    )
    _fft_length = 2 ** int(
        np.ceil(np.log(_window_length_samples) / np.log(2.0))
    )

    # spectrogram to mel transform operator
    _spec_to_mel_mat = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=_num_mel_bins,
        num_spectrogram_bins=_fft_length // 2 + 1,
        sample_rate=internal_sample_rate_in_hz,
        lower_edge_hertz=VGGishParams.MEL_MIN_HZ,
        upper_edge_hertz=VGGishParams.MEL_MAX_HZ,
        dtype=tf.dtypes.float32,
    )

    @classmethod
    def _log_mel_spectrogram(
        cls, audio: tf.Tensor
    ) -> tf.Tensor:  # pragma: no cover
        """TF version of mfcc_mel.LogMelSpectrogram."""
        spectrogram = tf.abs(
            tf.signal.stft(
                tf.cast(audio, tf.dtypes.float32),
                frame_length=cls._window_length_samples,
                frame_step=cls._hop_length_samples,
                fft_length=cls._fft_length,
                window_fn=tf.signal.hann_window,
            )
        )
        # somehow the shapes don't really work by default,
        # therefore we throw away two frames here, shouldn't matter
        mel = tf.matmul(spectrogram, cls._spec_to_mel_mat)[1:-1]
        return cls._stabilized_log(
            mel, cls._log_additive_offset, cls._log_floor
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
    def _update_statistics(
        statistics: _Statistics,
        data: tf.Tensor,
        num_items_processed: Union[int, tf.Variable],
    ):
        """
        Updates the means and covariances held by an instance of
        `FrechetAudioDistance.Statistics.`
        """
        data = tf.cast(data, dtype=tf.float64)
        batch_size = tf.cast(tf.shape(data)[0], dtype=tf.float64)

        x_norm_old = data - statistics.mean
        statistics.mean.assign_add(
            tf.reduce_sum(x_norm_old, axis=0) / num_items_processed
        )
        x_norm_new = data - statistics.mean

        statistics.covariance.assign(
            statistics.covariance
            * (num_items_processed - batch_size)
            / num_items_processed
        )
        statistics.covariance.assign_add(
            tf.matmul(tf.transpose(x_norm_old), x_norm_new)
            / num_items_processed
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

    _vggish_model_checkpoint_url: str = (
        "https://storage.googleapis.com/tfhub-modules/google/vggish/1.tar.gz"
    )

    @classmethod
    def _init_vggish_model(cls) -> tf.keras.Model:
        model_path = os.path.dirname(
            tf.keras.utils.get_file(
                "vggish_model.tar.gz",
                cls._vggish_model_checkpoint_url,
                extract=True,
                cache_subdir="vggish",
            )
        )
        return cls._assign_weights_to_model(
            cls._load_vggish_weights(model_path),
            cls._build_vggish_as_keras_model(),
        )

    @staticmethod
    def _load_vggish_weights(saved_model_path: str) -> List[tf.Variable]:
        weights = []
        loaded_obj = tf.saved_model.load(saved_model_path)
        for weight_name_in_orig_model in VGGishParams.VAR_NAMES:
            for (
                weight_var
            ) in (
                loaded_obj._variables
            ):  # only way I got this SOMEHOW to work at all... might break.
                if weight_var.name == weight_name_in_orig_model:
                    weights.append(weight_var)
        return weights

    @staticmethod
    def _assign_weights_to_model(
        weights: List[tf.Variable], keras_model: tf.keras.Model
    ) -> tf.keras.Model:
        for layer in keras_model.layers:
            for w in layer.trainable_weights:
                w.assign(weights.pop(0))
        assert len(weights) == 0
        return keras_model

    @staticmethod
    def _build_vggish_as_keras_model():
        conv_layer_kwargs = {
            "kernel_size": (3, 3),
            "strides": (1, 1),
            "padding": "SAME",
            "activation": "relu",
        }
        pool_layer_kwargs = {"strides": (2, 2), "padding": "SAME"}

        input = tf.keras.layers.Input(
            shape=(VGGishParams.NUM_FRAMES, VGGishParams.NUM_BANDS)
        )
        x = tf.reshape(
            input, [-1, VGGishParams.NUM_FRAMES, VGGishParams.NUM_BANDS, 1]
        )
        x = tf.keras.layers.Conv2D(64, **conv_layer_kwargs)(x)
        x = tf.keras.layers.MaxPool2D(**pool_layer_kwargs)(x)
        x = tf.keras.layers.Conv2D(128, **conv_layer_kwargs)(x)
        x = tf.keras.layers.MaxPool2D(**pool_layer_kwargs)(x)
        x = tf.keras.layers.Conv2D(256, **conv_layer_kwargs)(x)
        x = tf.keras.layers.Conv2D(256, **conv_layer_kwargs)(x)
        x = tf.keras.layers.MaxPool2D(**pool_layer_kwargs)(x)
        x = tf.keras.layers.Conv2D(512, **conv_layer_kwargs)(x)
        x = tf.keras.layers.Conv2D(512, **conv_layer_kwargs)(x)
        x = tf.keras.layers.MaxPool2D(**pool_layer_kwargs)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4096, activation="relu")(x)
        x = tf.keras.layers.Dense(4096, activation="relu")(x)
        x = tf.keras.layers.Dense(
            VGGishParams.EMBEDDING_SIZE, activation=None
        )(x)
        embedding = tf.identity(x, name="embedding")
        return tf.keras.Model(inputs=[input], outputs=[embedding])


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
