# Measuring Plot in the Age of Distant Reading. A study in the search for the narrative arc within the stories of Sherlock Holmes.

A Master's Thesis about detecting basic plot structures within the Complete Adventures of Sherlock Holmes with the help of Distant Reading and NLP.

## Setup
Install the required libraries 

```pip install -r requirements.txt``` (Python version 3.11 is recommended)

Also make sure that the required NLTK and spaCy modules are installed

```<python-version> -m nltk.downloader punkt, averaged_perceptron_tagger, wordnet, vader_lexicon, maxent_ne_chunker, words```

```<python-version> -m spacy download en_core_web_md```

## Usage
- View the code as a pre-rendered and documented Jupyter Notebook

```sherlock.ipynb```
    
- Run the code yourself

```<python-version> sherlock.py -h/--help```
