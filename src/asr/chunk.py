"""
Create chunks
"""

import csv
import itertools
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree

import pandas as pd
from torchcodec.decoders import AudioDecoder  # type: ignore
from torchcodec.encoders import AudioEncoder  # type: ignore
from tqdm import tqdm

from asr.utils import DATA_DIR


@dataclass
class Chunk:
    start_time: float
    end_time: float
    transcript_path: str
    audio_path: str
    # The utterance ID can be used to uniquely identify the chunk within the corpus
    utterance_id: str
    utterance_path: str
    gold_text: str


@dataclass
class TranscribedChunk(Chunk):
    system_text: str
    system: str | None = None


class Model:
    name: str

    def __init__(self) -> None:
        pass


class Corpus:
    name: str
    path: Path
    chunks: list[Chunk] | None = None
    transcriptions: dict[str, str] | None = None

    @staticmethod
    def load_chunks(chunks_csv: str | Path) -> list[Chunk]:
        """
        side effect: stores in self.chunks
        """
        with open(chunks_csv, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            chunks = [Chunk(**row) for row in reader]

        return chunks

    @classmethod
    def load_transcriptions(cls, transcription_dir: Path) -> dict[str, str]:
        """
        load a map of chunks

        side effect: stores in self.transcriptions
        """
        chunks = cls.load_chunks()
        result = {}
        for chunk in self.chunks:
            transcript = (
                (transcription_dir / f"{Path(chunk.audio_path).stem}.txt").open().read()
            )
            result[chunk.utterance_id] = transcript
        return result


class CORAALCorpus(Corpus):
    name = "coraal"

    def __init__(self, path=Path("/data/corpora/coraal")):
        self.path = Path(path)

    def _parse_transcript(
        self, transcript_file: Path
    ) -> Iterator[tuple[float, float, dict]]:
        # Implement logic to parse transcript file and yield (start_time, end_time, meta) tuples
        with open(transcript_file, "r") as f:
            for line in f.readlines()[1:]:  # Skip header line
                line_no, spkr, start, text, end = line.strip().split("\t")
                yield (
                    float(start),
                    float(end),
                    {"line_no": line_no, "spkr": spkr, "text": text},
                )

    def _create_chunks_for_file(
        self, audio_file: Path, output_dir: Path, overwrite_existing=False
    ) -> list[Chunk]:
        audio = AudioDecoder(str(audio_file))
        sample_rate = audio.metadata.sample_rate
        assert sample_rate is not None, f"Sample rate is None for {audio_file}"
        transcript_file = audio_file.parent / f"{audio_file.stem}.txt"
        audio = audio.get_all_samples()

        chunks = []
        for start, end, meta in self._parse_transcript(transcript_file):
            utterance_id = f"c-{transcript_file.stem.lower()}-{meta['line_no']}"
            utterance_path = output_dir / f"{utterance_id}.wav"
            chunks.append(
                Chunk(
                    start_time=start,
                    end_time=end,
                    transcript_path=str(transcript_file),
                    audio_path=str(audio_file),
                    utterance_id=utterance_id,
                    utterance_path=str(utterance_path),
                    gold_text=meta["text"],
                )
            )

            if utterance_path.exists() and not overwrite_existing:
                continue

            try:
                audio_chunk = audio.data[
                    :, int(start * sample_rate) : int(end * sample_rate)
                ]
                AudioEncoder(audio_chunk, sample_rate=sample_rate).to_file(
                    utterance_path
                )
            except Exception as e:
                print(audio.data.shape)
                print(start, end, sample_rate)
                print(audio_chunk.shape)
                print(f"Error processing {audio_file} for chunk {utterance_id}: {e}")
                return []
        return chunks

    def create_chunks(self, overwrite_existing=False, num_procs=-1) -> list[Chunk]:
        # Implement logic to read CORAAL corpus and yield Chunk instances
        output_dir = DATA_DIR / "chunks" / self.name
        output_dir.mkdir(parents=True, exist_ok=True)
        files = [p for p in self.path.glob("*.wav") if not p.name.startswith(".")]
        chunks = []
        with Pool(num_procs) as pool:
            for result in tqdm(
                pool.imap_unordered(
                    partial(
                        self._create_chunks_for_file,
                        output_dir=output_dir,
                        overwrite_existing=overwrite_existing,
                    ),
                    files,
                    chunksize=10,
                ),
                total=len(files),
                desc="Creating chunks",
            ):
                chunks.extend(result)
        output_path = DATA_DIR / "chunks" / f"{self.name}_chunks.tsv"
        pd.DataFrame([chunk.__dict__ for chunk in chunks]).to_csv(
            output_path, sep="\t", index=False
        )
        return chunks


class SCOSYACorpus(Corpus):
    name = "scosya"
    regions = [
        "Ayrshire A",
        "Ayrshire B",
        "Borders",
        "Caithness",
        "Dumfries",
        "Fife",
        "Glasgow A",
        "Glasgow B",
        "Highlands",
        "Kinross",
        "Lothian",
        "North East",
        "Orkney",
        "Shetland",
        "Stirling and Falkirk",
        "Tayside and Angus",
        "Western Isles",
    ]

    def __init__(self, path=Path("/data/corpora/scosya")):
        self.path = Path(path)

    def _parse_transcript(
        self, transcript_file: Path
    ) -> Iterator[tuple[float, float, dict]]:
        # We implement our own TRS parser because the existing library takes too long and does too much
        parser = ElementTree.XMLParser(encoding="ISO-8859-1")
        try:
            tree = ElementTree.parse(transcript_file, parser=parser)
        except Exception as e:
            print(f"Error parsing {transcript_file}: {e}")
            return

        # we don't save the speaker info aside from a speaker ID
        # speakers = {
        #     spkr.get("id"): {"id": spkr.get("id"), "name": spkr.get("name")}
        #     for spkr in tree.findall("Speakers/Speaker")
        # }

        for i, line in enumerate(tree.findall("Episode/Section/Turn")):
            try:
                start_time = float(line.get("startTime"))  # type:ignore  (we wrap in exception instead of checking the type ahead of time)
                end_time = float(line.get("endTime"))  # type:ignore
            except:
                raise ValueError(
                    f"Invalid start or end time {line.get('startTime')} -> {line.get('endTime')}"
                )
            yield (
                start_time,
                end_time,
                {"line_no": i, "spkr": line.get("speaker"), "text": line.text},
            )

    def _create_chunks_for_file(
        self, audio_file: Path, output_dir: Path, overwrite_existing=False
    ) -> list[Chunk]:
        audio = AudioDecoder(str(audio_file))
        sample_rate = audio.metadata.sample_rate
        assert sample_rate is not None, f"Sample rate is None for {audio_file}"

        transcript_file = audio_file.parent / f"{audio_file.stem}.trs"

        audio = audio.get_all_samples()
        chunks = []
        for start, end, meta in self._parse_transcript(transcript_file):
            utterance_id = f"c-{transcript_file.stem}-{meta['line_no']}"
            utterance_path = output_dir / f"{utterance_id}.wav"
            chunks.append(
                Chunk(
                    start_time=start,
                    end_time=end,
                    transcript_path=str(transcript_file),
                    audio_path=str(audio_file),
                    utterance_id=utterance_id,
                    utterance_path=str(utterance_path),
                    gold_text=meta["text"],
                )
            )

            if utterance_path.exists() and not overwrite_existing:
                continue

            try:
                audio_chunk = audio.data[
                    :, int(start * sample_rate) : int(end * sample_rate)
                ]
                AudioEncoder(audio_chunk, sample_rate=sample_rate).to_file(
                    utterance_path
                )
            except Exception as e:
                print(audio.data.shape)
                print(start, end, sample_rate)
                print(audio_chunk.shape)
                print(f"Error processing {audio_file} for chunk {utterance_id}: {e}")
                return []
        return chunks

    def create_chunks(self, overwrite_existing=False, num_procs=-1) -> list[Chunk]:
        output_dir = DATA_DIR / "chunks" / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        files = [
            p
            for region in self.regions
            for p in (self.path / region).glob("*.wav")
            if not p.name.startswith(".")
        ]
        chunks = []
        if num_procs == 1:
            chunks = list(
                itertools.chain.from_iterable(
                    tqdm(
                        map(
                            partial(
                                self._create_chunks_for_file,
                                output_dir=output_dir,
                                overwrite_existing=overwrite_existing,
                            ),
                            files,
                        ),
                        total=len(files),
                        desc="Creating chunks",
                    )
                )
            )
        else:
            with Pool(num_procs) as pool:
                for result in tqdm(
                    pool.imap_unordered(
                        partial(
                            self._create_chunks_for_file,
                            output_dir=output_dir,
                            overwrite_existing=overwrite_existing,
                        ),
                        files,
                        chunksize=10,
                    ),
                    total=len(files),
                    desc="Creating chunks",
                ):
                    chunks.extend(result)
        output_path = DATA_DIR / "chunks" / f"{self.name}_chunks.tsv"
        pd.DataFrame([chunk.__dict__ for chunk in chunks]).to_csv(
            output_path, sep="\t", index=False
        )
        return chunks


if __name__ == "__main__":
    corpus = SCOSYACorpus()
    chunks = corpus.create_chunks(num_procs=1)
    print(f"Created {len(chunks)} chunks for {corpus.name} corpus")
