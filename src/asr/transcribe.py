"""
Use model to transcribe a corpus (data parallel)
"""

import argparse
import csv
import subprocess
from pathlib import Path

import whisper
from tqdm.auto import tqdm

from asr.utils import DATA_DIR


def process_subset(subset, rank, model_name="large", overwrite_existing=False):
    """Process the subset."""
    model = whisper.load_model(model_name, device=f"cuda:{rank % 4}")

    input_dir = subset[0].parent
    output_dir = DATA_DIR / "transcriptions" / input_dir.name
    print(output_dir)

    for audio_file in tqdm(subset, desc=f"Process rank {rank}"):
        result = model.transcribe(str(audio_file))
        output_file = output_dir / f"{audio_file.stem}.txt"
        if output_file.exists() and not overwrite_existing:
            continue

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(str(result["text"]) + "\n")


def compile_files(inputs):
    input_dir = inputs[0].parent
    output_dir = DATA_DIR / "transcriptions" / input_dir.name
    output_file = DATA_DIR / "transcriptions" / f"{input_dir.name}.tsv"

    writer = csv.DictWriter(
        output_file.open("w"),
        fieldnames=["utterance_id", "system_text", "transcription_path"],
        delimiter="\t",
    )
    writer.writeheader()
    for file in tqdm(list(output_dir.glob("*.txt"))):
        writer.writerow(
            {
                "utterance_id": file.stem,
                "system_text": file.open("r").read(),
                "transcription_path": str(file),
            }
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe a corpus using a model with data parallel processing."
    )
    parser.add_argument("input_dir", type=str)
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank of the process (for distributed processing).",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Total number of processes for distributed processing.",
    )
    parser.add_argument(
        "--only-compile",
        action="store_true",
        help="Only compile the transcribed TXTs into a single file; do not run the transcription.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_dir = Path(args.input_dir)
    # Get all .wav files in the input directory
    inputs = list(input_dir.glob("*.wav"))
    rank = args.rank
    num_processes = args.num_processes
    # start processes with other ranks
    subprocesses = []
    if rank == 0:
        print(f"Found {len(inputs)} .wav files in {input_dir}")

        for i in range(1, num_processes):
            proc = subprocess.Popen(
                [
                    "python",
                    __file__,
                    "--rank",
                    str(i),
                    "--num-processes",
                    str(num_processes),
                    "--only-compile" if args.only_compile else "",
                    str(input_dir),
                ]
            )
            subprocesses.append(proc)

    if not args.only_compile:
        # process texts for the current rank
        process_subset(inputs[rank::num_processes], rank=rank)

    # wait for all subprocesses to finish
    if rank == 0:
        for proc in subprocesses:
            proc.wait()
        print("All subprocesses finished.")
        print("Compiling into single file.")
        compile_files(inputs)
        print("Finished compiling into single file.")

    else:
        print(f"Process {rank} finished transcribing texts.")
    pass
