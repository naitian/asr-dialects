## Code layout

Each axis of variation (corpus, model, normalization rule) is a first-class,
independently editable unit:

| module             | responsibility |
|--------------------|----------------|
| `asr/chunk.py`     | `Chunk` / `TranscribedChunk` dataclasses (the on-disk/in-frame schema, incl. `corpus`, `speaker`, `line_no`, `extra`). |
| `asr/normalize.py` | Normalization pipeline. `SHARED_LEXICON` + numeric tail apply to all; each corpus is a `NormProfile` (markup steps = gold-only, lexicon = both sides). `build(profile, strip_markup=...)`; `PROFILES` registry. |
| `asr/corpora.py`   | `Corpus` ABC holding all chunking/audio logic; `CORAALCorpus`/`SCOSYACorpus` override only discovery + transcript parsing; `CORPORA` registry. |
| `asr/models.py`    | `Model` protocol (`transcribe(path)->str`), `WhisperModel`, `MODELS` registry (factories keyed by name). |
| `asr/dataset.py`   | `load(corpora, models)` -> tidy long-format DataFrame (one row per utterance x model) with `gold_norm`/`system_norm` applied per corpus, plus the cached per-utterance `wer` when `data/wer/{corpus}__{model}.tsv` exists. Normalization (the slow step) is cached to `data/norm/` and self-invalidates per row via fingerprints; see "Caching" below. The analysis entry point. |
| `asr/transcribe.py`| CLI: run a model over a corpus's chunks (data parallel), output `data/transcriptions/{corpus}__{model}.tsv`. |
| `asr/evaluate.py`  | CLI: exact-match + WER per (corpus, model) via `dataset.load`; also writes per-utterance WER to `data/wer/{corpus}__{model}.tsv` (read back by `load`). |

Adding a corpus = subclass `Corpus` + a `NormProfile` + registry entries.
Adding a model = implement `transcribe()` + a `MODELS` entry.

Normalization is applied **at load time**, not baked into stored files, so the
raw text stays canonical and revising rules just means reloading.

### Caching

Normalization runs the whisper normalizers per string and is the slow part of a
load, so `load` caches it under `data/norm/` (`{corpus}__gold.tsv`,
`{corpus}__{model}__system.tsv`) and `asr.evaluate` caches per-utterance WER
under `data/wer/`. Everything operates at the corpus (`corpus × model`) level, so
each cache is validated as a **whole file** via a sidecar `.fp` fingerprint —
match ⇒ reuse the file, mismatch ⇒ rebuild it. No manual busting:

- **norm cache**: `fp = hash(rule_signature + all inputs)`, where
  `rule_signature = hash(asr/normalize.py source + whisper version)`. Editing
  `normalize.py` or any transcript changes the fingerprint and recomputes the
  file. Pass `use_cache=False` to skip the cache entirely.
- **WER cache**: WER is a pure function of `(gold_norm, system_norm)`, so
  `fp = hash` over the scored utterances' normalized pairs — no rule signature
  needed. `load` recomputes the (now cached, cheap) normalizations and reuses the
  WER file only if the fingerprint still matches, so WER survives rule edits that
  don't change the normalized text.

Caches are plain TSVs (+ `.fp` sidecars) and safe to delete; they rebuild on the
next load.

```sh
python -m asr.corpora                                   # build chunks
python -m asr.transcribe coraal --model whisper-large --num-processes 4
python -m asr.evaluate coraal scosya --model whisper-large
```

SCOSYA `.trs` text lives in the tails after `<Sync>`/`<Who>` markers, not in
`Turn.text`; `parse_transcript` collects it via `itertext()`. To refresh just
the chunk metadata after a parser change (no audio re-extraction), use
`Corpus.build_index()`; `create_chunks()` (re)extracts audio clips and skips
ones that already exist.

---

ASR evaluation steps:

1. Standardize gold transcripts. We consult the SCOSYA and CORAAL corpora
   transcription guidelines and resolve any inconsistencies.

2. Standardize gold transcripts with expected ASR output. For example, from
   Koenecke et al. (2020), "we modified nonstandard spellings: for example, we
   changed occurrences of the word “aks” to “ask,” since no ASRs spell this
   utterance using the AAVE pronunciation"

3. Standardize all outputs. Again, from Koenecke et al. (2020):
    - Single spacing was enforced between words
    - Arabic numerals were converted to numeric strings
    - flags indicating hesitation were removed from the transcripts
    - the “$” sign was replaced with the “dollar” string
    - all other special characters and punctuation were removed
    - cardinal direction abbreviations (e.g., “NW”) were replaced with full
      words (e.g., “Northwest”)
    - full state names were replaced with their two-letter abbreviations
    - all words were converted to lowercase
    - certain spellings were standardized: for example, “cuz,” “ok,” “o,”
      “till,” “imma,” “mister,” “yup,” “gonna,” and “tryna” were, respectively,
      replaced with “cause,” “okay,” “oh,” “til,” “ima,” “mr,” “yep,” “going
      to,” and “trying to”)
    - we removed both filler words (“um,” “uh,” “mm,” “hm,” “ooh,” “woo,”
      “mhm,” “huh,” “ha”) and expletives because the ASR systems handle these
      words differently from each other


ASR inference steps:

1. Split into chunks (based on the transcript timestamps). Have a unique ID for
   each chunk.

2. Run each model and record the output.



## Standardization Notes:

### CORAAL

#### Special symbols

- Quoted speech is not represented orthographically.

TODO:
- Strip surrounding `[...]` (overlapping speech)
- Strip around `/.../` (unsure transcriptions, unintelligible, misspoken words, etc)
- Strip around `<...>` (non-linguistic noises)
- Strip around `(...)` (line-level notes, like `(laughing)`)

#### Orthographic conventions

- standard English spelling / punctuation
- all numbers are written as complete words
- abbreviations are avoided except for personal titles (`Mrs.`, etc); `junior`
  is left unabbreviated
- acronyms are writting without internal punctuation
- spelled-out letters are separated by dashes

#### Reduced forms:

- musta, woulda, shoulda, coulda, mighta
- gonna, hafta, tryna, sposta, finna, gotta, wanna, oughta
- cause (because), til (until), 'em (them), lemme (let me)
- filled pauses: uh, um, ah
- Uh-huh, Uh-h / Nuh-h, Mm-hm / Mm, Mm-mm, Mkay, Yep/Yup, Nah/Naw, Oh, Ooh, Ayo, Hoo.

#### Redactions:
- Written as /RD-TAG-NUMSYLLABLES/



### SCOSYA


## Notes

Prestwick Y is split into (1) and (2); I just concatenated them with `ffmpeg`:
```
ffmpeg -f concat -safe 0 -i <(printf "file '$PWD/%s'\n" PRESTWICK-Y-ANON\
\(1\).wav PRESTWICK-Y-ANON\ \(2\).wav) -c copy PRESTWICK-Y-ANON.wav
```
but is this right? What is the source `.wav` file that was used in Transcriber?


The audio is split between two channels for different speakers, but there are
also auxiliary speakers sometimes. What do we do about this?


There is a trailing space behind `Ayrshire A/ISLAY-Y-ANON .trs` and the `.txt`
file. I removed this.


In LOCHEE-Y-ANON.trs, there is an utterance whose timestamp I think is
mistranscribed:

> "Here's an a- here's an add from- Here's an add from [name]." {BR}

is noted as having start time 3723.976 and endtime 4857.699, but I think this
is incorrect, listening back to the audio.


coraal whisper-large-en: 15:30:00 on 2 L40s (480121 chunks)
scosya whisper-large-en: 15:30:00 on 2 L40s (227491 chunks)
