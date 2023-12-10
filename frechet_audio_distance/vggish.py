import os

import numpy as np
import tensorflow as tf

from .feature import FADFeature

VGGISH_PUBLIC_MODEL_CHECKPOINT_URL: str = (
    "https://storage.googleapis.com/tfhub-modules/google/vggish/1.tar.gz"
)


class VGGish(FADFeature):

    def __init__(self, step_size_in_s: float = .5):
        self._step_size_in_samples = int(
            round(step_size_in_s * self.input_sample_rate_in_hz)
        )
        self.model = self._init_vggish_model()

    @property
    def input_sample_rate_in_hz(self) -> float:
        return self.sample_rate_in_hz

    @property
    def output_dim(self):
        return self.embedding_size

    @tf.function
    def __call__(self, audio: tf.Tensor) -> tf.Tensor:  # pragma: no cover
        mel_feature = self._extract_mel_features(audio)
        embeddings = self.model(mel_feature)
        return embeddings

    # Parameters used in the VGGish model by the original authors.
    # Content copied straight from
    # https://github.com/tensorflow/models/tree/master/research/audioset/vggish

    # Architectural constants.
    num_frames = 96  # Frames in input mel-spectrogram patch.
    embedding_size = 128  # Size of embedding layer.

    # Hyperparameters used in feature and example generation.
    sample_rate_in_hz = 16000
    stft_window_length_seconds = 0.025
    STFT_HOP_LENGTH_SECONDS = 0.010
    num_mel_bins = 64
    mel_min_hz = 125
    mel_max_hz = 7500
    log_offset = (
        0.01  # Offset used for stabilized log of input mel-spectrogram.
    )
    example_window_seconds = 0.96  # Each example contains 96 10ms frames
    example_hop_seconds = 0.96  # with zero overlap.

    var_names = [
        "vggish/conv1/weights:0",
        "vggish/conv1/biases:0",
        "vggish/conv2/weights:0",
        "vggish/conv2/biases:0",
        "vggish/conv3/conv3_1/weights:0",
        "vggish/conv3/conv3_1/biases:0",
        "vggish/conv3/conv3_2/weights:0",
        "vggish/conv3/conv3_2/biases:0",
        "vggish/conv4/conv4_1/weights:0",
        "vggish/conv4/conv4_1/biases:0",
        "vggish/conv4/conv4_2/weights:0",
        "vggish/conv4/conv4_2/biases:0",
        "vggish/fc1/fc1_1/weights:0",
        "vggish/fc1/fc1_1/biases:0",
        "vggish/fc1/fc1_2/weights:0",
        "vggish/fc1/fc1_2/biases:0",
        "vggish/fc2/weights:0",
        "vggish/fc2/biases:0",
    ]

    # spectrogram params
    _num_mel_bins = 64
    _log_additive_offset = 0.001
    _log_floor = 1e-12
    _window_length_secs = 0.025
    _hop_length_secs = 0.010

    _window_length_samples = int(round(sample_rate_in_hz
                                       * _window_length_secs))
    _hop_length_samples = int(
        round(sample_rate_in_hz * _hop_length_secs)
    )
    _fft_length = 2 ** int(
        np.ceil(np.log(_window_length_samples) / np.log(2.0))
    )

    # spectrogram to mel transform operator
    _spec_to_mel_mat = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=_num_mel_bins,
        num_spectrogram_bins=_fft_length // 2 + 1,
        sample_rate=sample_rate_in_hz,
        lower_edge_hertz=mel_min_hz,
        upper_edge_hertz=mel_max_hz,
        dtype=tf.dtypes.float32,
    )

    @staticmethod
    def _normalize_audio(
            audio_batch: tf.Tensor) -> tf.Tensor:  # pragma: no cover
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

    def _extract_mel_features(
            self,
            audio_batch: tf.Tensor) -> tf.Tensor:  # pragma: no cover
        normalized_audio_batch = self._normalize_audio(audio_batch)
        framed_audio = tf.signal.frame(
            normalized_audio_batch,
            VGGish.sample_rate_in_hz,
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

    def _log_mel_spectrogram(self, audio: tf.Tensor
                             ) -> tf.Tensor:  # pragma: no cover
        spectrogram = tf.abs(
            tf.signal.stft(
                tf.cast(audio, tf.dtypes.float32),
                frame_length=self._window_length_samples,
                frame_step=self._hop_length_samples,
                fft_length=self._fft_length,
                window_fn=tf.signal.hann_window,
            )
        )
        # somehow the shapes don't really work by default,
        # therefore we throw away two frames here, shouldn't matter
        # in the big picture
        mel = tf.matmul(spectrogram, self._spec_to_mel_mat)[1:-1]
        return self._stabilized_log(
            mel, self._log_additive_offset, self._log_floor
        )

    @classmethod
    def _init_vggish_model(cls) -> tf.keras.Model:
        model_path = os.path.dirname(
            tf.keras.utils.get_file(
                "vggish_model.tar.gz",
                VGGISH_PUBLIC_MODEL_CHECKPOINT_URL,
                extract=True,
                cache_subdir="vggish",
            )
        )
        return cls._assign_weights_to_model(
            cls._load_vggish_weights(model_path),
            cls._build_vggish_as_keras_model(),
        )

    @staticmethod
    def _load_vggish_weights(saved_model_path: str) -> list[tf.Variable]:
        weights = []
        loaded_obj = tf.saved_model.load(saved_model_path)
        for weight_name_in_orig_model in VGGish.var_names:
            # accessing this protected member of this class was the only way I
            # got this SOMEHOW to work at all... might break someday.
            for weight_var in loaded_obj._variables:
                if weight_var.name == weight_name_in_orig_model:
                    weights.append(weight_var)
        return weights

    @staticmethod
    def _assign_weights_to_model(
            weights: list[tf.Variable], keras_model: tf.keras.Model
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

        input_layer = tf.keras.layers.Input(
            shape=(VGGish.num_frames, VGGish.num_mel_bins)
        )
        x = tf.reshape(
            input_layer, [-1, VGGish.num_frames,
                          VGGish.num_mel_bins, 1]
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
            VGGish.embedding_size, activation=None
        )(x)
        embedding = tf.identity(x, name="embedding")
        return tf.keras.Model(inputs=[input_layer], outputs=[embedding])
