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
