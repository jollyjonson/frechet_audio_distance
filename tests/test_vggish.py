import unittest

import numpy as np
import numpy.testing as npt
import tensorflow as tf

from frechet_audio_distance.vggish import VGGish
from .test_signals import (
    EXPECTED_EMBEDDING_FROM_TEST_INPUT,
    VGGISH_TEST_INPUT,
    EXPECTED_EMBEDDING_FROM_1S_1KHZ_AUDIO,
)


class VGGishTests(unittest.TestCase):
    def test_vggish_model_computes_correct_embeddings_from_given_feature(self):
        actual_embeddings = VGGish().model(VGGISH_TEST_INPUT)
        self.assertTrue(
            np.allclose(
                actual_embeddings,
                EXPECTED_EMBEDDING_FROM_TEST_INPUT,
                atol=1e-6,
            )
        )

    @staticmethod
    def _generate_1s_1khz_test_signal() -> tf.Tensor:
        test_signal_len_in_s = 1.0
        time = (
            np.arange(0, test_signal_len_in_s * VGGish.sample_rate_in_hz)
            / VGGish.sample_rate_in_hz
        )
        test_signal_freq_in_hz = 1000.0
        test_signal_1khz = np.sin(2.0 * np.pi * test_signal_freq_in_hz * time)
        test_signal_1khz_with_batch_dim = np.expand_dims(test_signal_1khz, 0)
        return tf.convert_to_tensor(
            test_signal_1khz_with_batch_dim, dtype=tf.float32
        )

    def test_vggish_class_computes_correct_embeddings_from_audio(self):
        actual_embeddings = VGGish()(self._generate_1s_1khz_test_signal())
        self.assertTrue(
            np.allclose(
                actual_embeddings,
                EXPECTED_EMBEDDING_FROM_1S_1KHZ_AUDIO,
                atol=1e-3,
            )
        )

    def test_vggish_output_dim_is_correct(self):
        self.assertEqual(VGGish().output_dim, 128)


if __name__ == "__main__":
    unittest.main()
