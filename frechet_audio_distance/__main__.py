import argparse

from frechet_audio_distance import FrechetAudioDistance
from frechet_audio_distance.files_processor import FilesProcessor

BLOCK_SIZE_IN_S = 10
SAMPLERATE_IN_HZ = 16000

if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Compute the FAD given a set of reference "
        "and estimate audio files."
    )

    parser.add_argument(
        "-e",
        "--estimates",
        required=True,
        nargs="+",
        help="Estimated audio files given like "
        "'-e path/to/file1.wav path/to/file2.wav'",
    )
    parser.add_argument(
        "-r",
        "--references",
        required=True,
        nargs="+",
        help="Estimated audio files given like "
        "'-r path/to/file1.wav path/to/file2.wav'",
    )
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    distance = FilesProcessor(
        reference_files=args.references, estimate_files=args.estimates
    )(
        metric=FrechetAudioDistance(SAMPLERATE_IN_HZ),
        samplerate_in_hz=SAMPLERATE_IN_HZ,
        downmix_to_mono=False,
        block_size_in_s=BLOCK_SIZE_IN_S,
    )

    print(f"FAD: {distance}")

    if args.output is not None:
        with open(args.output, "w") as file_handle:
            file_handle.write(str(distance))
