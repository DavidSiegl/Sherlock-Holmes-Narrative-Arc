import os
import re
import pandas as pd
import string

from collections import defaultdict

from source.functions import remove_numeric, segment_text

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


def preprocessing_narrative_coherence(df_sherlock):
    df_sherlock['text_prepro_narrative'] = df_sherlock['text_prepro'].str.lower()
    translator = str.maketrans('', '', string.punctuation)

    df_sherlock['text_prepro_narrative'] = df_sherlock['text_prepro_narrative'].apply(lambda x: x.translate(translator))
    df_sherlock['text_prepro_narrative'] = df_sherlock['text_prepro_narrative'].apply(lambda x: nltk.tokenize.word_tokenize(x))
    df_sherlock['segments_narrative'] = df_sherlock['text_prepro_narrative'].apply(segment_text)

    df_sherlock_segments_narrative = df_sherlock.explode('segments_narrative')
    values = list(range(1, 6)) * ((len(df_sherlock_segments_narrative) // 5) + 1)
    df_sherlock_segments_narrative['segment_num'] = values[:len(df_sherlock_segments_narrative)]
    
    return df_sherlock_segments_narrative

