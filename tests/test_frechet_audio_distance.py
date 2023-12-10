import unittest
from typing import List, Optional, Any

import numpy as np
import tensorflow as tf

from frechet_audio_distance import (
    FrechetAudioDistance,
    RunFrechetAudioDistanceOnlyOnValidationAndTestCallback,
)
from tests.test_signals import REGRESSION_TEST_EXPECTED_VALUE


FAD_REGRESSION_TEST_EXPECTED_RESULT = 32.92703601606229


class FrechetAudioDistanceTests(unittest.TestCase):
    _test_sine_frequencies = [200, 1000, 2000, 2500]
    _test_noise_levels = [0.1, 0.25, 0.5]
    _test_signal_length_in_s = 1.2
    _test_sample_rates_in_hz = [
        16000,
        44100,
    ]  # non-resampling and resampling case

    _before_distance_scores: Optional[List[float]] = None

    def _generate_test_signals(self, sample_rate_in_hz: float) -> np.ndarray:
        time = np.linspace(
            0,
            self._test_signal_length_in_s,
            int(round(sample_rate_in_hz * self._test_signal_length_in_s)),
        )
        test_signals = np.zeros(
            (len(self._test_sine_frequencies), len(time)), dtype=float
        )
        for idx, frequency in enumerate(self._test_sine_frequencies):
            test_signals[idx, :] = np.sin(2.0 * np.pi * frequency * time)
        return np.expand_dims(test_signals, axis=-1)

    @staticmethod
    def _distort_signals_with_additive_gaussian_noise(
        signals: np.ndarray, noise_level: float
    ) -> np.ndarray:
        random_state = np.random.RandomState(
            1999
        )  # setUp is not called, when running through PyCharm
        return signals + noise_level * random_state.randn(*signals.shape)

    def test_instance_has_correct_name(self):
        distance_instance = FrechetAudioDistance(44100)
        self.assertEqual(
            distance_instance.name, distance_instance.default_name
        )

    def test_distance_is_zero_for_equal_inputs(self):
        distance = FrechetAudioDistance(self._test_sample_rates_in_hz[0])
        test_batch = np.random.randn(
            1, 2 * self._test_sample_rates_in_hz[0], 2
        )
        distance.update_state(test_batch, test_batch)
        distance_value = distance.result().numpy()
        self.assertTrue(np.allclose(distance_value, 0.0, atol=1e-4))

    def test_resetting_instance_works(self):
        test_batch = np.random.randn(
            1, 2 * self._test_sample_rates_in_hz[0], 2
        )
        distance = FrechetAudioDistance(self._test_sample_rates_in_hz[0])

        distance.update_state(test_batch, test_batch + 0.1)
        distance_value_1 = distance.result().numpy()

        distance.reset_state()

        distance.update_state(test_batch, test_batch + 0.1)
        distance_value_2 = distance.result().numpy()

        self.assertTrue(np.allclose(distance_value_1, distance_value_2))

    def test_distance_is_approximately_equal_for_same_signals_at_different_sample_rates(
        self,
    ):
        distance_values = []
        for sample_rate in self._test_sample_rates_in_hz:
            distance = FrechetAudioDistance(sample_rate)
            test_signals = self._generate_test_signals(sample_rate)
            distance.update_state(
                test_signals,
                self._distort_signals_with_additive_gaussian_noise(
                    test_signals, sample_rate
                ),
            )  # adding an offset, distortion invariant to resampling
            distance_values.append(distance.result().numpy())

        for distance_value in distance_values:
            self.assertGreater(
                distance_value, 20.0
            )  # needed for super large tolerance in next assertion
        self.assertTrue(np.allclose(*distance_values, atol=3.0))

    def _assert_scores_are_roughly_equal_to_last_batched_or_unbatched_result_or_set_result_if_none(
        self, distance_scores: List[float]
    ) -> None:
        """
        Called by the test methods which computes the FAD with batches and with single items to make sure results are
        equal.
        """
        if self._before_distance_scores is None:
            self._before_distance_scores = distance_scores
            return
        self.assertTrue(
            np.allclose(
                distance_scores, self._before_distance_scores, atol=1e-6
            )
        )

    def test_distance_increases_as_distortion_increases(self):
        sample_rate_in_hz = self._test_sample_rates_in_hz[0]
        distance = FrechetAudioDistance(sample_rate_in_hz)
        test_signals = self._generate_test_signals(sample_rate_in_hz)
        distance_values = []
        for distortion_level in self._test_noise_levels:
            distance.update_state(
                test_signals,
                self._distort_signals_with_additive_gaussian_noise(
                    test_signals, distortion_level
                ),
            )
            distance_values.append(distance.result().numpy())
            distance.reset_state()

        self._assert_distance_increases_as_distortion_increases(
            distance_values
        )
        self._assert_scores_are_roughly_equal_to_last_batched_or_unbatched_result_or_set_result_if_none(
            distance_values
        )

    def test_distance_also_works_when_single_elements_are_passed_and_results_are_equal_to_batch_processing(
        self,
    ):
        sample_rate_in_hz = self._test_sample_rates_in_hz[0]
        distance = FrechetAudioDistance(sample_rate_in_hz)
        test_signals = self._generate_test_signals(sample_rate_in_hz)
        distance_values = []
        for distortion_level in self._test_noise_levels:
            for test_signal in test_signals:
                test_signal = np.expand_dims(test_signal, axis=0)
                distance.update_state(
                    test_signal,
                    self._distort_signals_with_additive_gaussian_noise(
                        test_signal, distortion_level
                    ),
                )
            distance_values.append(distance.result().numpy())
            distance.reset_state()

        self._assert_distance_increases_as_distortion_increases(
            distance_values
        )
        self._assert_scores_are_roughly_equal_to_last_batched_or_unbatched_result_or_set_result_if_none(
            distance_values
        )

    def _assert_distance_increases_as_distortion_increases(
        self, distance_values: List[float]
    ) -> None:
        distortion_levels_increase = all(
            [d > 0.0 for d in np.diff(self._test_noise_levels)]
        )
        self.assertTrue(
            distortion_levels_increase,
            "This test only works if the test distortion levels increase, "
            "please change them accordingly!",
        )
        fad_scores_increase = all([d > 0.0 for d in np.diff(distance_values)])
        self.assertTrue(fad_scores_increase)

    def test_distance_works_with_model_fit(self):
        sample_rate = 44100
        test_signals = self._generate_test_signals(sample_rate)
        model = tf.keras.Sequential(
            [tf.keras.layers.Conv1D(1, 2, padding="same")]
        )
        model.compile(
            loss=tf.keras.losses.MSE,
            metrics=[FrechetAudioDistance(sample_rate)],
        )
        history = model.fit(
            test_signals + 0.1,
            test_signals,
            epochs=2,
            callbacks=[
                RunFrechetAudioDistanceOnlyOnValidationAndTestCallback()
            ],
            validation_split=0.5,
            verbose=False,
        )

    def test_regression_test(self):
        sample_rate_in_hz = 16000
        distance = FrechetAudioDistance(sample_rate_in_hz)
        test_signals = self._generate_test_signals(sample_rate_in_hz)
        for distortion_level in self._test_noise_levels:
            for multipliers in [1.0, 0.9, 0.85]:
                distance.update_state(
                    test_signals,
                    self._distort_signals_with_additive_gaussian_noise(
                        test_signals * multipliers, distortion_level
                    ),
                )

        diff = distance.result().numpy() - FAD_REGRESSION_TEST_EXPECTED_RESULT
        tolerance = 1e-1
        self.assertLess(diff, tolerance)

    def test_another_fad_regression_test(self):
        sample_rate = 16000
        test_signals = self._generate_test_signals(sample_rate)
        fad = FrechetAudioDistance(sample_rate)
        fad.update_state(test_signals, test_signals + 0.1)
        fad_value = fad.result().numpy()
        diff = np.abs(fad_value - REGRESSION_TEST_EXPECTED_VALUE)
        tolerance = 2e-3
        self.assertLess(diff, tolerance)

    def test_attempting_to_merge_states_raises(self):
        fad = FrechetAudioDistance(16000)
        self.assertRaises(NotImplementedError, fad.merge_state)


if __name__ == "__main__":
    unittest.main()
