import os

import librosa.core.audio
import numpy as np
import tensorflow as tf
import tqdm

Path = str | os.PathLike


class FilesProcessor:
    """
    Compute a stateful keras metric such as the FAD on a set of reference
    and estimated files.
    """

    def __init__(
        self,
        reference_files: list[Path],
        estimate_files: list[Path],
        verbose: bool = True,
        **kwargs,
    ):
        self._reference_files = reference_files
        self._estimate_files = estimate_files
        self.verbose = verbose

    def __call__(
        self,
        metric: tf.keras.metrics.Metric,
        samplerate_in_hz: int = 16000,
        downmix_to_mono: bool = False,
        block_size_in_s: float = 30,
    ) -> float:
        self.check_metric_for_compatibility(metric)
        block_size_in_samples = block_size_in_s * samplerate_in_hz
        if self.verbose:
            progress_bar = tqdm.tqdm(
                total=len(self._estimate_files) + len(self._reference_files)
            )

        for file_list, update_method_putter in zip(
            [self._estimate_files, self._reference_files],
            [lambda x: (None, x), lambda x: (x, None)],
        ):
            for file_path in file_list:
                if self.verbose:
                    progress_bar.update()
                audio = self.load_audio_and_readjust_dimensionality(
                    downmix_to_mono, file_path, samplerate_in_hz
                )

                num_samples = audio.shape[1]
                num_blocks_this_file = int(
                    np.ceil(num_samples / block_size_in_samples)
                )

                # feed the metric all blocks but the last block
                if num_blocks_this_file >= 2:
                    for block_idx in range(num_blocks_this_file - 1):
                        start_idx = int(block_idx * block_size_in_samples)
                        end_idx = int((block_idx + 1) * block_size_in_samples)
                        metric.update_state(
                            *update_method_putter(
                                tf.convert_to_tensor(
                                    audio[:, start_idx:end_idx, :]
                                )
                            )
                        )
                # feed last or only block
                start_idx = int(
                    (num_blocks_this_file - 1) * block_size_in_samples
                )
                metric.update_state(
                    *update_method_putter(
                        tf.convert_to_tensor(audio[:, start_idx:, :])
                    )
                )

        return metric.result().numpy()

    def load_audio_and_readjust_dimensionality(
        self, downmix_to_mono: bool, file_path: Path, samplerate_in_hz: int
    ) -> np.ndarray:
        """
        Load audio from a file, transpose it to channels first, add a channel
        dimension if it's mono as well as add a batch dimension
        for compatibility with the metric.
        """
        audio = librosa.core.audio.load(
            file_path, sr=samplerate_in_hz, mono=downmix_to_mono
        )[0].T
        if audio.ndim == 1:  # add channel dim if missing
            audio = np.expand_dims(audio, -1)
        # add batch dim
        audio = np.expand_dims(audio, 0)
        return audio

    @staticmethod
    def check_metric_for_compatibility(metric) -> None:
        if not getattr(metric, "stateful", False):
            raise RuntimeError(
                f"This class assumes a stateful metric, "
                f"the given metric instance '{metric}' does not have "
                f"a stateful property or it is False."
            )
