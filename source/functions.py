import re
import nltk
import math
import pandas as pd
import spacy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


lemmatizer = nltk.stem.WordNetLemmatizer()
vader = SentimentIntensityAnalyzer()

def remove_numeric(text):
    return re.sub(r'\d+', '', text)


def segment_text (text):
    segment_length = len(text) // 5
    segments = [text[i:i+segment_length] for i in range(0, len(text), segment_length)]
    return segments[:5]


def lem_text(text):
    lemmed_words = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(lemmed_words)

def count_matches(tokens, word_list):
    count = sum([1 for token in tokens if token in word_list])
    return count


def min_max_scale_column(df, new_column_name, col_to_norm, col_length):
    """this function first computes tf-idf on a dataframe and then scales the retrieved values to a range between 0 and 1"""
    df['tf'] = df[col_to_norm] / len(df[col_length])
    df['idf'] = math.log(len(df.index)) / (df[col_to_norm] != 0).value_counts()[True]
    df[new_column_name] = df['tf'] * df['idf']
    scaler = MinMaxScaler()
    df[new_column_name] = scaler.fit_transform(df[[new_column_name]])
    return df


def group_df(df, value):
    """this function takes as an input a df and groups it according to the corresponding segment numbers for further ANOVA testing"""
    ls = []
    for i in range(1, df['segment_num'].nunique() + 1):
        ls.append(df[df['segment_num'] == i][value])
    return ls


def pos_tag_text(text):
    pos_tags = nltk.pos_tag(text)
    return pos_tags


def tense_counts(pos_tags):
    """this function takes as an input an array of pos tags and computes the counts for past, present and future tense usage by grouping them into three distinct categories according to their respective tag"""
    past_count = 0
    present_count = 0
    future_count = 0
    
    for word, tag in pos_tags[1::2]:
        if tag.startswith('VBD'):
            past_count += 1
        elif tag.startswith('VB'):
            present_count += 1
        elif tag.startswith('MD'):
            future_count += 1
    
    counts = {'past_count': past_count,
              'present_count': present_count,
              'future_count': future_count}
    
    return pd.Series(counts)


def get_sentiment_scores(sentence_list):
    scores = []
    for sentence in sentence_list:
        score = vader.polarity_scores(sentence)
        scores.append(score['compound'])
    return scores


def extract_entities(text):
    pos_tags = nltk.pos_tag(text)
    tree = nltk.ne_chunk(pos_tags)
    return tree


def count_entities(parse_tree, *labels):
    """this function takes as input a parsed ner tree and one or more labels and then computes the frequencies for the given label(s)"""
    tree = parse_tree
    count = 0
    for subtree in tree.subtrees():
        if subtree.label() in labels:
            count += 1
    return count


def most_common_entities(row, labels):
    """helper function for extracting the most common entities across all texts"""
    tree = row
    entities = []
    for label in labels:
        entities.extend([subtree.leaves() for subtree in tree.subtrees(lambda t: t.label() == label)])
    flattened_entities = [item for sublist in entities for item in sublist]
    if flattened_entities:
        most_common_entity = max(set(flattened_entities), key=flattened_entities.count)
        return most_common_entity
    else:
        return ''
    
def extract_events(text):
    """this function applies the basic spacy pipeline for event extraction to the textual input and then extracts certain entities based on the ruleset provided afterwards"""
    
    nlp = spacy.load('en_core_web_md')

    processed_text = ' '.join(text)
    doc = nlp(processed_text)

    events = []
    
    for ent in doc.ents:
        if ent.label_ == 'EVENT':
            event = {
                'event': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'context': [t.text for t in ent.sent],
            }
            events.append(event)
            
    for sent in doc.sents:
        persons = [ent for ent in sent.ents if ent.label_ == 'PERSON']
        locations = [ent for ent in sent.ents if ent.label_ in ['LOC', 'GPE', 'FAC', 'TIME']]
        
        for person in persons:
            for location in locations:
                event = {
                    'event': person.text + ' ' + location.text,
                    'start': min(person.start_char, location.start_char),
                    'end': max(person.end_char, location.end_char),
                    'context': [t.text for t in sent],
                }
                events.append(event)
                           
    return events


def compute_lsi_on_subsets(df, text_column, category_column):
    vectorizer = TfidfVectorizer()
    lsa = TruncatedSVD(n_components=30, random_state=19)
    results = {}

    unique_categories = df[category_column].unique()

    for category in unique_categories:
        subset_df = df[df[category_column] == category]
        dtm = vectorizer.fit_transform(subset_df[text_column].apply(lambda x: ' '.join(x)))
        lsa.fit(dtm)
        lsa_vectors = lsa.transform(dtm)
        feature_names = vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(lsa.components_):
            top_features = [feature_names[i] for i in topic.argsort()[:-5 - 1:-1]]
            topics.append(top_features)

        results[category] = topics

    return results


def vectorize_lsi_results(lsi_results):
    vectorizer = TfidfVectorizer()
    category_vectors = {}

    for category, topics in lsi_results.items():
        features = [' '.join(topic) for topic in topics]
        category_vectors[category] = vectorizer.fit_transform(features)

    return category_vectors


def compute_difference(category_vectors):
    categories = list(category_vectors.keys())
    num_categories = len(categories)
    differences = np.zeros((num_categories, num_categories))
    
    svd = TruncatedSVD(n_components=min([vector.shape[1] for vector in category_vectors.values()]))
    reduced_vectors = {category: svd.fit_transform(vector) for category, vector in category_vectors.items()}


    for i in range(num_categories):
        vector_i = reduced_vectors[categories[i]]
        for j in range(i+1, num_categories):
            vector_j = reduced_vectors[categories[j]]
            similarity = cosine_similarity(vector_i, vector_j)[0][0]
            difference = 1 - similarity
            differences[i, j] = difference
            differences[j, i] = difference

    return differences