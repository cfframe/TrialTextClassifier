# Pre-requisites
Set up a virtual environment of some sort as some package version dependencies can be tricky to resolve.
## SpaCy
See https://spacy.io/usage. Check version of CUDA (e.g. <code>nvidia-smi</code> from the command line) and amend accordingly. 
#### Course by Ines
https://course.spacy.io/en/chapter1
#### GitHub
https://github.com/ines/spacy-course

```
pip install -U pip setuptools wheel
pip install -U spacy[cuda114]
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

## Text classification with BERT
```
pip install transformers
```
### Reference sites
Text Classification with BERT in PyTorch | by Ruben
(https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f)

Kaggle BBC dataset
(https://www.kaggle.com/sainijagjit/bbc-dataset/version/1)
