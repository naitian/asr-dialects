"""
Transcribe a corpus's chunks with a model (data parallel across GPUs).

Reads the per-utterance clips under ``data/chunks/{corpus}/`` and writes one
system's output to ``data/transcriptions/{corpus}__{model}.tsv`` (plus per-clip
``.txt`` files under ``data/transcriptions/{corpus}/{model}/``).

    python -m asr.transcribe coraal --model whisper-large --num-processes 4

Keying outputs by both corpus and model lets several systems coexist for
evaluation; see ``asr.dataset.load`` to read them back.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm

from asr.models import MODELS, build_model
from asr.utils import DATA_DIR


def chunks_dir(corpus: str) -> Path:
    return DATA_DIR / "chunks" / corpus


def output_dir(corpus: str, model: str) -> Path:
    return DATA_DIR / "transcriptions" / corpus / model


def compiled_path(corpus: str, model: str) -> Path:
    return DATA_DIR / "transcriptions" / f"{corpus}__{model}.tsv"


def process_subset(
    clips: list[Path],
    corpus: str,
    model_name: str,
    rank: int,
    overwrite_existing: bool = False,
) -> None:
    model = build_model(model_name, device=f"cuda:{rank % 4}")
    out_dir = output_dir(corpus, model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    for clip in tqdm(clips, desc=f"{model_name} rank {rank}"):
        output_file = out_dir / f"{clip.stem}.txt"
        if output_file.exists() and not overwrite_existing:
            continue
        text = model.transcribe(clip)
        output_file.write_text(text + "\n")


def compile_outputs(corpus: str, model_name: str) -> None:
    out_dir = output_dir(corpus, model_name)
    with compiled_path(corpus, model_name).open("w") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["utterance_id", "system_text", "system", "transcription_path"],
            delimiter="\t",
        )
        writer.writeheader()
        for file in tqdm(sorted(out_dir.glob("*.txt")), desc="Compiling"):
            writer.writerow(
                {
                    "utterance_id": file.stem,
                    "system_text": file.read_text().strip(),
                    "system": model_name,
                    "transcription_path": str(file),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="corpus name (chunks live in data/chunks/<corpus>)")
    parser.add_argument(
        "--model", default="whisper-large", choices=list(MODELS), help="model to run"
    )
    parser.add_argument("--rank", type=int, default=0, help="this process's rank")
    parser.add_argument(
        "--num-processes", type=int, default=1, help="number of data-parallel processes"
    )
    parser.add_argument(
        "--only-compile",
        action="store_true",
        help="only compile per-clip TXTs into the TSV; do not transcribe.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clips = sorted(chunks_dir(args.corpus).glob("*.wav"))

    subprocesses = []
    if args.rank == 0:
        print(f"Found {len(clips)} clips for {args.corpus} -> {args.model}")
        for rank in range(1, args.num_processes):
            cmd = [
                sys.executable, "-m", "asr.transcribe", args.corpus,
                "--model", args.model,
                "--rank", str(rank),
                "--num-processes", str(args.num_processes),
            ]
            if args.only_compile:
                cmd.append("--only-compile")
            if args.overwrite:
                cmd.append("--overwrite")
            subprocesses.append(subprocess.Popen(cmd))

    if not args.only_compile:
        process_subset(
            clips[args.rank :: args.num_processes],
            corpus=args.corpus,
            model_name=args.model,
            rank=args.rank,
            overwrite_existing=args.overwrite,
        )

    if args.rank == 0:
        for proc in subprocesses:
            proc.wait()
        print("Compiling into single file.")
        compile_outputs(args.corpus, args.model)
        print(f"Wrote {compiled_path(args.corpus, args.model)}")
    else:
        print(f"Process {args.rank} finished transcribing.")


if __name__ == "__main__":
    main()
