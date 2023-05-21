"""
Content copied straight from
https://github.com/tensorflow/models/tree/master/research/audioset/vggish
"""


class VGGishParams:
    # Architectural constants.
    NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
    NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
    EMBEDDING_SIZE = 128  # Size of embedding layer.

    # Hyperparameters used in feature and example generation.
    SAMPLE_RATE_IN_HZ = 16000
    STFT_WINDOW_LENGTH_SECONDS = 0.025
    STFT_HOP_LENGTH_SECONDS = 0.010
    NUM_MEL_BINS = NUM_BANDS
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = (
        0.01  # Offset used for stabilized log of input mel-spectrogram.
    )
    EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
    EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.

    # Hyperparameters used in training.
    INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.

    VAR_NAMES = [
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
