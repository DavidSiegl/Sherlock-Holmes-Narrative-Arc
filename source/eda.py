import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

matplotlib.use('TkAgg')

stopwords = nltk.corpus.stopwords.words('english')

def eda(df_sherlock):
    df_sherlock.reset_index(inplace=True)
    df_sherlock_publications = pd.DataFrame(df_sherlock.groupby('year').count()['title'])
    df_sherlock_publications.reset_index(inplace=True)
    all_years = range(df_sherlock_publications['year'].min(), df_sherlock_publications['year'].max() + 1)
    df_sherlock_publications = df_sherlock_publications.set_index('year').reindex(all_years, fill_value=0).reset_index()
    sns.barplot(df_sherlock_publications, x='year', y='title', color='black')
    plt.xticks(rotation=90)
    plt.title('Publications of Sherlock Holmes stories over time')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.show()
    
    df_sherlock['word_counts'] = df_sherlock['text'].apply(lambda x: len(x.split()))
    sns.histplot(df_sherlock, x='word_counts', kde=False)
    plt.title('Distribution of word counts')
    plt.xlabel('Word Count')
    plt.ylabel('Count')
    plt.show()
     
    print(df_sherlock.nlargest(5, 'word_counts')[["title",'word_counts']])
    
    df_sherlock["avg_sent_len"] = df_sherlock["text"].map(lambda x: np.mean([len(w.split()) for w in nltk.tokenize.sent_tokenize(x)]))
    sns.histplot(df_sherlock, x='avg_sent_len', kde=True)
    plt.title('Distribution of average sentence length')
    plt.xlabel('Avg Sentence Length')
    plt.ylabel('Count')
    plt.show()
 
    all_text = ' '.join(df_sherlock['text_lemmed'])
    all_text = nltk.word_tokenize(all_text)
    
    word_counts = Counter(all_text)
    most_common = word_counts.most_common(25)
    words, frequency = [], []
    for word, count in most_common:
        words.append(word)
        frequency.append(count)

    sns.barplot(x = frequency, y = words, color='black')
    plt.title('Most frequent words across all texts')
    plt.xlabel('Count')
    plt.ylabel('Words')
    plt.show()
    
    df_sherlock['text_prepro_ngrams'] = df_sherlock['text_lemmed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

    cv = CountVectorizer(ngram_range=(2,2))
    bigrams = cv.fit_transform(df_sherlock['text_prepro_ngrams'])
    count_values = bigrams.toarray().sum(axis=0)
    ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv.vocabulary_.items()], reverse = True))
    ngram_freq.columns = ["frequency", "ngram"]
    sns.barplot(x=ngram_freq['frequency'][:25], y=ngram_freq['ngram'][:25], color='black')
    plt.title('Top 25 Most Frequently Occuring Bigrams')
    plt.xlabel('Ngrams')
    plt.ylabel('Count')
    plt.show()
    
    cv2 = CountVectorizer(ngram_range=(3,3))
    trigrams = cv2.fit_transform(df_sherlock['text_prepro_ngrams'])
    count_values = trigrams.toarray().sum(axis=0)
    ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv2.vocabulary_.items()], reverse = True))
    ngram_freq.columns = ["frequency", "ngram"]
    sns.barplot(x=ngram_freq['frequency'][:25], y=ngram_freq['ngram'][:25], color='black')
    plt.title('Top 25 Most Frequently Occuring Trigrams')
    plt.xlabel('Ngrams')
    plt.ylabel('Count')
    plt.show()

    return None