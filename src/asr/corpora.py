"""
Corpora: turn source recordings + transcripts into per-utterance ``Chunk``s.

All the shared work -- decoding audio, slicing it on transcript timestamps,
writing the per-utterance clips, and compiling a ``{corpus}_chunks.tsv`` -- lives
on the :class:`Corpus` base class. A concrete corpus only declares what is
genuinely different:

    - ``audio_files()``      how to find the recordings
    - ``parse_transcript()`` how to read its transcript format
    - ``transcript_suffix``  the transcript file extension
    - ``norm_profile``       its normalization guidelines (see ``asr.normalize``)
    - optionally ``utterance_id()`` / ``chunk_region()`` / ``chunk_extra()`` for
      id formats, the shared ``region`` column, and corpus-specific metadata.

Adding a new corpus is therefore a small subclass plus one ``CORPORA`` entry.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree

import pandas as pd
from torchcodec.decoders import AudioDecoder  # type: ignore
from torchcodec.encoders import AudioEncoder  # type: ignore
from tqdm import tqdm

from asr.chunk import Chunk
from asr.normalize import PROFILES, NormProfile
from asr.utils import DATA_DIR


class Corpus(ABC):
    name: str
    norm_profile: NormProfile
    transcript_suffix: str

    def __init__(self, path: str | Path):
        self.path = Path(path)

    # ------------------------------------------------------------------ #
    # Corpus-specific hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def audio_files(self) -> list[Path]:
        """Return the recordings to chunk."""

    @abstractmethod
    def parse_transcript(
        self, transcript_file: Path
    ) -> Iterator[tuple[float, float, dict]]:
        """Yield ``(start_time, end_time, meta)`` per utterance.

        ``meta`` must contain ``line_no``, ``speaker`` and ``text``.
        """

    def transcript_path_for(self, audio_file: Path) -> Path:
        return audio_file.parent / f"{audio_file.stem}{self.transcript_suffix}"

    def utterance_id(self, transcript_file: Path, line_no) -> str:
        return f"c-{transcript_file.stem}-{line_no}"

    def chunk_region(self, audio_file: Path) -> str:
        """Geographic region for every chunk from a file (shared column)."""
        return ""

    def chunk_extra(self, audio_file: Path) -> dict:
        """Corpus-specific metadata to attach to every chunk from a file."""
        return {}

    # ------------------------------------------------------------------ #
    # Shared chunking machinery
    # ------------------------------------------------------------------ #
    def _records_for_file(self, audio_file: Path, output_dir: Path) -> list[Chunk]:
        """Parse a transcript into chunk records (no audio extraction)."""
        transcript_file = self.transcript_path_for(audio_file)
        region = self.chunk_region(audio_file)
        extra = self.chunk_extra(audio_file)
        chunks = []
        for start, end, meta in self.parse_transcript(transcript_file):
            utterance_id = self.utterance_id(transcript_file, meta["line_no"])
            chunks.append(
                Chunk(
                    corpus=self.name,
                    utterance_id=utterance_id,
                    speaker=str(meta.get("speaker", "")),
                    line_no=str(meta["line_no"]),
                    region=region,
                    start_time=start,
                    end_time=end,
                    gold_text=meta["text"] or "",
                    audio_path=str(audio_file),
                    utterance_path=str(output_dir / f"{utterance_id}.wav"),
                    transcript_path=str(transcript_file),
                    extra=dict(extra),
                )
            )
        return chunks

    def _create_chunks_for_file(
        self, audio_file: Path, output_dir: Path, overwrite_existing: bool = False
    ) -> list[Chunk]:
        chunks = self._records_for_file(audio_file, output_dir)

        # Only decode the source audio if some clips actually need extracting.
        pending = [
            chunk
            for chunk in chunks
            if overwrite_existing or not Path(chunk.utterance_path).exists()
        ]
        if not pending:
            return chunks

        audio = AudioDecoder(str(audio_file))
        sample_rate = audio.metadata.sample_rate
        assert sample_rate is not None, f"Sample rate is None for {audio_file}"
        audio = audio.get_all_samples()
        for chunk in pending:
            try:
                audio_chunk = audio.data[
                    :,
                    int(chunk.start_time * sample_rate) : int(
                        chunk.end_time * sample_rate
                    ),
                ]
                AudioEncoder(audio_chunk, sample_rate=sample_rate).to_file(
                    Path(chunk.utterance_path)
                )
            except Exception as err:
                print(
                    f"Error processing {audio_file} for chunk "
                    f"{chunk.utterance_id} [{chunk.start_time}-{chunk.end_time}]: {err}"
                )
        return chunks

    def create_chunks(
        self, overwrite_existing: bool = False, num_procs: int = -1
    ) -> list[Chunk]:
        output_dir = DATA_DIR / "chunks" / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        files = self.audio_files()
        worker = partial(
            self._create_chunks_for_file,
            output_dir=output_dir,
            overwrite_existing=overwrite_existing,
        )

        chunks: list[Chunk] = []
        if num_procs == 1:
            results: Iterator[list[Chunk]] = map(worker, files)
            for result in tqdm(results, total=len(files), desc=f"Chunking {self.name}"):
                chunks.extend(result)
        else:
            with Pool(num_procs) as pool:
                for result in tqdm(
                    pool.imap_unordered(worker, files, chunksize=10),
                    total=len(files),
                    desc=f"Chunking {self.name}",
                ):
                    chunks.extend(result)

        self._write_chunks(chunks)
        return chunks

    def build_index(self) -> list[Chunk]:
        """Rebuild only the ``{corpus}_chunks.tsv`` from transcripts.

        Parses every transcript and rewrites the chunk metadata (gold text,
        speaker, timings, ...) without decoding or extracting any audio. Use
        this to refresh the index after a transcript-parser change; audio clips
        are produced separately by :meth:`create_chunks`.
        """
        output_dir = DATA_DIR / "chunks" / self.name
        chunks: list[Chunk] = []
        for audio_file in tqdm(self.audio_files(), desc=f"Indexing {self.name}"):
            chunks.extend(self._records_for_file(audio_file, output_dir))
        self._write_chunks(chunks)
        return chunks

    def _write_chunks(self, chunks: list[Chunk]) -> None:
        output_path = DATA_DIR / "chunks" / f"{self.name}_chunks.tsv"
        records = []
        for chunk in chunks:
            row = asdict(chunk)
            row["extra"] = json.dumps(row["extra"])  # keep the TSV cell scalar
            records.append(row)
        pd.DataFrame(records).to_csv(output_path, sep="\t", index=False)


class CORAALCorpus(Corpus):
    name = "coraal"
    norm_profile = PROFILES["coraal"]
    transcript_suffix = ".txt"

    def __init__(self, path: str | Path = "/data/corpora/coraal"):
        super().__init__(path)

    def audio_files(self) -> list[Path]:
        return [p for p in self.path.glob("*.wav") if not p.name.startswith(".")]

    def utterance_id(self, transcript_file: Path, line_no) -> str:
        return f"c-{transcript_file.stem}-{line_no}"

    def chunk_region(self, audio_file: Path) -> str:
        # CORAAL filenames are `{region}_se{N}_ag{N}_{gender}_...`.
        return audio_file.stem.split("_")[0]

    def chunk_extra(self, audio_file: Path) -> dict:
        # Speaker demographics live in the first four `_`-separated filename
        # components: region (its own column), socioeconomic group, age group,
        # gender.
        _region, socioeconomic, age, gender = audio_file.stem.split("_")[:4]
        return {
            "socioeconomic_group": int(socioeconomic.removeprefix("se")),
            "age_group": int(age.removeprefix("ag")),
            "gender": gender,
        }

    def parse_transcript(
        self, transcript_file: Path
    ) -> Iterator[tuple[float, float, dict]]:
        with open(transcript_file, "r") as f:
            for line in f.readlines()[1:]:  # skip header line
                line_no, spkr, start, text, end = line.strip().split("\t")
                yield (
                    float(start),
                    float(end),
                    {"line_no": line_no, "speaker": spkr, "text": text},
                )


class SCOSYACorpus(Corpus):
    name = "scosya"
    norm_profile = PROFILES["scosya"]
    transcript_suffix = ".trs"
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

    def __init__(self, path: str | Path = "/data/corpora/scosya"):
        super().__init__(path)

    def audio_files(self) -> list[Path]:
        return [
            p
            for region in self.regions
            for p in (self.path / region).glob("*.wav")
            if not p.name.startswith(".")
            # see notes on PRESTWICK-Y-ANON split files
            and p.name not in ("PRESTWICK-Y-ANON (1).wav", "PRESTWICK-Y-ANON (2).wav")
        ]

    def chunk_region(self, audio_file: Path) -> str:
        # Recordings are filed under a per-region directory.
        return audio_file.parent.name

    def _parse(self, file_contents: str) -> Iterator[tuple[float, float, dict]]:
        parser = ElementTree.XMLParser(encoding="ISO-8859-1")
        tree = ElementTree.fromstring(file_contents, parser=parser)

        line_counter = 0
        for i, line in enumerate(tree.findall("Episode/Section/Turn")):
            # this can consist of one chunk or multiple <Sync> segments;
            # in the case of multiple segments, we want to split by each
            # <Sync> and yield them separately.
            syncs = line.findall("Sync")
            if not syncs:
                continue  # skip empty turns
            start_times = [sync.get("time") for sync in syncs]
            end_times = start_times[1:] + [line.get("endTime")]
            for sync, start_time, end_time in zip(syncs, start_times, end_times):
                # there is always at least one <Sync> in a Turn
                text = sync.tail.strip() or ""
                assert start_time is not None, f"Missing startTime for line: {text}"
                assert end_time is not None, f"Missing endTime for line: {text}"
                try:
                    yield (
                        float(start_time),
                        float(end_time),
                        {
                            "line_no": line_counter,
                            "speaker": line.get("speaker"),
                            "text": text,
                        },
                    )
                    line_counter += 1
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Invalid start or end time "
                        f"{line.get('startTime')} -> {line.get('endTime')}"
                    )

    def parse_transcript(
        self, transcript_file: Path
    ) -> Iterator[tuple[float, float, dict]]:
        # We implement our own TRS parser because the existing library takes too
        # long and does too much.
        try:
            yield from self._parse(transcript_file.read_text(encoding="ISO-8859-1"))
        except Exception as err:
            raise ValueError(f"Error parsing {transcript_file}: {err}") from err


CORPORA: dict[str, type[Corpus]] = {
    CORAALCorpus.name: CORAALCorpus,
    SCOSYACorpus.name: SCOSYACorpus,
}


if __name__ == "__main__":
    corpus = SCOSYACorpus()
    corpus.name = "scosya_sync"
    chunks = corpus.create_chunks(num_procs=8)
    print(f"Created {len(chunks)} chunks for {corpus.name} corpus")
    # corpus = CORAALCorpus()
    # chunks = corpus.create_chunks(num_procs=1)
    # print(f"Created {len(chunks)} chunks for {corpus.name} corpus")
