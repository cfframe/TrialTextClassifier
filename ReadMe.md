# Pre-requisites
Set up a virtual environment of some sort as some package version dependencies can be tricky to resolve.
## SpaCy
See https://spacy.io/usage. Check version of CUDA (e.g. <code>nvidia-smi</code> from the command line) and amend accordingly. 

```
pip install -U pip setuptools wheel
pip install -U spacy[cuda114]
python -m spacy download en_core_web_sm
```
