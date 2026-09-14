## ASR dialects code

This repository is set up to use `pixi` for dependency management. If you have `pixi` installed, you can just run `pixi install` in this folder to install all of the libraries you need. (https://pixi.prefix.dev/latest/installation/)


`corpora.py` defines how to read a transcript:
- will probably want to update the `_parse` method to read in the updated timestamps
- the bottom of this file contains the code to actually do the chunking
- to set the path: `corpus = SCOSYACorpus(path="/path/to/your/scosya/")`

`models.py`: the models that we evaluate are defined here
- `HFModel` should work for most huggingface models that support the ASR pipeline.

`transcribe.py`: does the transcription. You can do `python -m asr.transcribe -h` to see the help menu

`evaluate.py`: reports WER. this uses the normalization that exists in `normalize.py`


`normalize.py`: contains the normalization rules.
