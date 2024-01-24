import os
import re
import pandas as pd
import string
import nltk

from collections import defaultdict
from source.functions import remove_numeric, segment_text, lem_text

translator = str.maketrans('', '', string.punctuation)
stopwords = nltk.corpus.stopwords.words('english')


def preprocessing():
    df_dict = defaultdict(str)
    title = None
    for f in os.listdir('./data/sherlock/'):
        with open('./data/sherlock/' + f, 'r') as text:
            lines = text.readlines()
            text.seek(0)
            ls = list()
            for number, line in enumerate(lines):
                if number == 4: # extracting the title of each story as a key
                    title = line
                    title = title.strip()
                elif number not in [4, 6] and line != "\n": # skipping unnecessary lines
                    ls.append(line.strip(" "))

            conc_text = "".join(ls)
            conc_text = re.sub(r'[-]{10}\n([\S\s]+)', '', conc_text) # removing additional lines at the bottom of the text files
            conc_text = conc_text.strip()
            df_dict[title] = conc_text


    df_sherlock = pd.DataFrame.from_dict(df_dict, orient='index', columns=['text'])
    df_sherlock.index.name = 'title'
    
    years = [1926, 1924, 1904, 1904, 1892, 1904, 1926, 1892, 1891, 1908, 1892, 1904, 1892, 1923, 1893, 1903, 1910, 1913, 1903, 1892, 1893, 1891, 1893, 1904, 1893, 1901, 1891, 1924, 1911, 1917, 1926, 1921, 1904, 1893, 1893, 1892, 1903, 1904, 1911,1891, 1893, 1893, 1926, 1891, 1904, 1927, 1890, 1892, 1904, 1903, 1892, 1891,1887, 1924, 1922, 1891, 1915, 1927, 1892, 1893]

    df_sherlock['year'] = years
    
    df_sherlock['text_prepro'] = df_sherlock['text'].apply(lambda x: remove_numeric(x))
          
    return df_sherlock


def preprocessing_lemmatisation(df_sherlock):
    
    df_sherlock['text_prepro_lemmatisation'] = df_sherlock['text_prepro'].str.lower()
    df_sherlock['text_prepro_lemmatisation'] = df_sherlock['text_prepro_lemmatisation'].apply(lambda x: x.translate(translator))

    df_sherlock['text_prepro_lemmatisation'] = df_sherlock['text_prepro_lemmatisation'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    df_sherlock['text_prepro_lemmatisation'] = df_sherlock['text_prepro_lemmatisation'].apply(lambda x: nltk.tokenize.word_tokenize(x))
    
    df_sherlock['text_lemmed'] = df_sherlock['text_prepro_lemmatisation'].apply(lem_text)
    
    return df_sherlock


def preprocessing_narrative_coherence(df_sherlock):
    df_sherlock['text_prepro_narrative'] = df_sherlock['text_prepro'].str.lower()
    df_sherlock['text_prepro_narrative'] = df_sherlock['text_prepro_narrative'].apply(lambda x: x.translate(translator))
    df_sherlock['text_prepro_narrative'] = df_sherlock['text_prepro_narrative'].apply(lambda x: nltk.tokenize.word_tokenize(x))
    df_sherlock['segments_narrative'] = df_sherlock['text_prepro_narrative'].apply(segment_text)

    df_sherlock_segments_narrative = df_sherlock.explode('segments_narrative')
    values = list(range(1, 6)) * ((len(df_sherlock_segments_narrative) // 5) + 1)
    df_sherlock_segments_narrative['segment_num'] = values[:len(df_sherlock_segments_narrative)]
    
    df_sherlock_segments_narrative.reset_index(inplace=True)
    
    return df_sherlock_segments_narrative


def preprocessing_temporal_usage(df_sherlock):
    df_sherlock['text_prepro_temp'] = df_sherlock['text_prepro'].str.lower()
    df_sherlock['text_prepro_temp'] = df_sherlock['text_prepro_temp'].apply(lambda x: x.translate(translator))

    df_sherlock['text_prepro_temp'] = df_sherlock['text_prepro_temp'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    df_sherlock['text_prepro_temp'] = df_sherlock['text_prepro_temp'].apply(lambda x: nltk.tokenize.word_tokenize(x))
    df_sherlock['segments_temp'] = df_sherlock['text_prepro_temp'].apply(segment_text)

    df_sherlock_segments_temp = df_sherlock.explode('segments_temp')
    values = list(range(1, 6)) * ((len(df_sherlock_segments_temp) // 5) + 1)
    df_sherlock_segments_temp['segment_num'] = values[:len(df_sherlock_segments_temp)]
    
    df_sherlock_segments_temp.reset_index(inplace=True)
    
    return df_sherlock_segments_temp


def preprocessing_emotion_analysis(df_sherlock):
    df_sherlock['text_prepro_tok_vader'] = df_sherlock['text_prepro'].apply(lambda x: nltk.tokenize.sent_tokenize(x))
    df_sherlock['segments_vader'] = df_sherlock['text_prepro_tok_vader'].apply(segment_text) 

    df_sherlock_segments_vader = df_sherlock.explode('segments_vader')
    values = list(range(1, 6)) * ((len(df_sherlock_segments_vader) // 5) + 1)
    df_sherlock_segments_vader['segment_num'] = values[:len(df_sherlock_segments_vader)]
    
    df_sherlock_segments_vader.reset_index(inplace=True)
    
    return df_sherlock_segments_vader


def preprocessing_ner(df_sherlock):
    df_sherlock['text_prepro_ner'] = df_sherlock['text_prepro'].apply(lambda x: x.translate(translator))
    df_sherlock['text_prepro_ner'] = df_sherlock['text_prepro_ner'].apply(lambda x: nltk.tokenize.word_tokenize(x))
    df_sherlock['segments_ner'] = df_sherlock['text_prepro_ner'].apply(segment_text)

    df_sherlock_segments_ner = df_sherlock.explode('segments_ner')
    values = list(range(1, 6)) * ((len(df_sherlock_segments_ner) // 5) + 1)
    df_sherlock_segments_ner['segment_num'] = values[:len(df_sherlock_segments_ner)]

    df_sherlock_segments_ner.reset_index(inplace=True)
    
    return df_sherlock_segments_ner


def preprocessing_ee(df_sherlock):
    df_sherlock['text_prepro_event'] = df_sherlock['text_prepro'].str.lower()
    df_sherlock['text_prepro_event'] = df_sherlock['text_prepro_event'].apply(lambda x: x.translate(translator))
    df_sherlock['text_prepro_event'] = df_sherlock['text_prepro_event'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    df_sherlock['text_prepro_event'] = df_sherlock['text_prepro_event'].apply(lambda x: nltk.tokenize.word_tokenize(x))
    df_sherlock['segments_event'] = df_sherlock['text_prepro_event'].apply(segment_text)
    df_sherlock_segments_event = df_sherlock.explode('segments_event')
    values = list(range(1, 6)) * ((len(df_sherlock_segments_event) // 5) + 1)
    df_sherlock_segments_event['segment_num'] = values[:len(df_sherlock_segments_event)]
    
    df_sherlock_segments_event.reset_index(inplace=True)
    
    return df_sherlock_segments_event


def preprocessing_lsi(df_sherlock):
    df_sherlock['text_prepro_lsi'] = df_sherlock['text_prepro'].str.lower()
    df_sherlock['text_prepro_lsi'] = df_sherlock['text_prepro_lsi'].apply(lambda x: x.translate(translator))
    df_sherlock['text_prepro_lsi'] = df_sherlock['text_prepro_lsi'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    df_sherlock['text_prepro_lsi'] = df_sherlock['text_prepro_lsi'].apply(lambda x: nltk.tokenize.word_tokenize(x))
    df_sherlock['segments_lsi'] = df_sherlock['text_prepro_lsi'].apply(segment_text)
    df_sherlock_segments_lsi = df_sherlock.explode('segments_lsi')
    values = list(range(1, 6)) * ((len(df_sherlock_segments_lsi) // 5) + 1)
    df_sherlock_segments_lsi['segment_num'] = values[:len(df_sherlock_segments_lsi)]
    
    df_sherlock_segments_lsi.reset_index(inplace=True)

    return df_sherlock_segments_lsi