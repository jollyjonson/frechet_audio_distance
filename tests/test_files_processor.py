import os.path
import shutil
import unittest

import numpy as np
import tensorflow as tf
from scipy.io import wavfile

from frechet_audio_distance import FrechetAudioDistance
from frechet_audio_distance.files_processor import Path, FilesProcessor


class FilesProcessorTests(unittest.TestCase):
    _test_directory_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tmp_files_processor_testing",
    )

    _num_testfiles_per_sample_rate = 3
    _sample_rates = (44100, 16000)
    _channels = (1, 2)
    _reference_files: list[Path]
    _estimate_files: list[Path]

    def __init__(self, *args, **kwargs):
        self._reference_files, self._estimate_files = [], []
        super().__init__(*args, **kwargs)

    def setUp(self):
        """
        Create a facade of reference and estimate files with random samples.
        """
        os.makedirs(self._test_directory_path, exist_ok=True)
        file_container_and_prefixes = {
            "reference_": self._reference_files,
            "estimate_": self._estimate_files,
        }
        for prefix, file_container in file_container_and_prefixes.items():
            for file_idx in map(
                str, range(self._num_testfiles_per_sample_rate)
            ):
                for sample_rate in self._sample_rates:
                    for num_channels in self._channels:
                        path = os.path.join(
                            self._test_directory_path,
                            prefix
                            + str(num_channels)
                            + "_"
                            + str(sample_rate)
                            + "_"
                            + file_idx
                            + ".wav",
                        )
                        wavfile.write(
                            path,
                            sample_rate,
                            self._generate_random_samples(
                                10 * sample_rate, num_channels
                            ),
                        )
                        file_container.append(path)

    @staticmethod
    def _generate_random_samples(
        len_in_samples: int, num_channels: int
    ) -> np.ndarray:
        return np.random.randn(len_in_samples, num_channels)

    def tearDown(self):
        """Clean up the random files used for testing"""
        shutil.rmtree(self._test_directory_path)

    def test_processor_works_for_unequal_amount_of_files(self):
        sample_rate_in_hz = 16000
        distance_val = FilesProcessor(
            self._reference_files, self._estimate_files[:-2]
        )(FrechetAudioDistance(sample_rate_in_hz), block_size_in_s=3)
        self.assertTrue(np.isfinite(distance_val))

    def test_processor_raises_for_non_stateful_metric(self):
        sample_rate_in_hz = 16000
        processor = FilesProcessor(self._reference_files, self._estimate_files)
        self.assertRaises(
            RuntimeError,
            lambda: processor(tf.keras.metrics.MSE, block_size_in_s=3),
        )


if __name__ == "__main__":
    unittest.main()
