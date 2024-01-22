import re
import nltk
import math
from sklearn.preprocessing import MinMaxScaler

lemmatizer = nltk.stem.WordNetLemmatizer()

def remove_numeric(text):
    return re.sub(r'\d+', '', text)


def segment_text (text):
    segment_length = len(text) // 5
    segments = [text[i:i+segment_length] for i in range(0, len(text), segment_length)]
    return segments[:5]


def lem_text(text):
    lemmed_words = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(lemmed_words)


def min_max_scale_column(df, new_column_name, col_to_norm, col_length):
    """this function first computes tf-idf on a dataframe and then scales the retrieved values to a range between 0 and 1"""
    df['tf'] = df[col_to_norm] / len(df[col_length])
    df['idf'] = math.log(len(df.index)) / (df[col_to_norm] != 0).value_counts()[True]
    df[new_column_name] = df['tf'] * df['idf']
    scaler = MinMaxScaler()
    df[new_column_name] = scaler.fit_transform(df[[new_column_name]])
    return df