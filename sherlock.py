import argparse

parser = argparse.ArgumentParser(
    prog = "Sherlock Holmes Narrative Arc",
    description = "This script is part of a Master's Thesis about detecting basic plot structures within the Complete Adventures of Sherlock Holmes with the help of Distant Reading and NLP. The script takes as input the respective module (e. g. narrative_arc) via the flag -m/--module and generates the corresponding visualisations and similar results for a given subtask of the carried out analysis",
)

parser.add_argument("-m", "--module", dest="module", required=True, type=str, choices=["eda", "narrative_coherence", "temporal_usage", "emotion_analysis", "ner", "event_extraction", "lsi", "clustering", "pca", "logistic_regression", "svm", "narrative_arc"], help="Required: Specify which part of the analysis should be printed out to standard output.")

args = parser.parse_args()

module = args.module

import joblib
import warnings

from source.preprocessing import preprocessing, preprocessing_narrative_coherence, preprocessing_temporal_usage, preprocessing_ee, preprocessing_lemmatisation, preprocessing_emotion_analysis, preprocessing_ner, preprocessing_lsi
from source.eda import eda
from source.feature_engineering import feature_engineering_narrative_coherence, feature_engineering_temporal_usage, feature_engineering_emotion_analysis, feature_engineering_ner
from source.topic_extraction import event_extraction, lsi
from source.modelling import clustering, pca, logistic_regression, svm, narrative_arc
from pandas.errors import SettingWithCopyWarning


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

df_sherlock = preprocessing()

if module == "eda":

    df_sherlock = preprocessing_lemmatisation(df_sherlock)
    eda(df_sherlock)


if module == "narrative_coherence":

    df_sherlock = preprocessing_narrative_coherence(df_sherlock)
    feature_engineering_narrative_coherence(df_sherlock)


if module == "temporal_usage":

    df_sherlock = preprocessing_temporal_usage(df_sherlock)
    feature_engineering_temporal_usage(df_sherlock)


if module == "emotion_analysis":

    df_sherlock = preprocessing_emotion_analysis(df_sherlock)
    feature_engineering_emotion_analysis(df_sherlock)

if module == "ner":

    df_sherlock = preprocessing_ner(df_sherlock)
    feature_engineering_ner(df_sherlock)

if module == "event_extraction":

    df_sherlock = preprocessing_ee(df_sherlock)
    event_extraction(df_sherlock)

if module == "lsi":

    df_sherlock = preprocessing_lsi(df_sherlock)
    lsi(df_sherlock)

if module == "clustering":

    df_sherlock_modelling = joblib.load('./data/df_sherlock_modelling.lib')
    df_sherlock_modelling['period'] = df_sherlock_modelling['year'].apply(lambda x: 0 if x <= 1893 else 1)
    clustering(df_sherlock_modelling)

if module == "pca":

    df_sherlock_modelling = joblib.load('./data/df_sherlock_modelling.lib')
    df_sherlock_modelling['period'] = df_sherlock_modelling['year'].apply(lambda x: 0 if x <= 1893 else 1)
    pca(df_sherlock_modelling)

if module == "logistic_regression":

    df_sherlock_modelling = joblib.load('./data/df_sherlock_modelling.lib')
    df_sherlock_modelling['period'] = df_sherlock_modelling['year'].apply(lambda x: 0 if x <= 1893 else 1)
    logistic_regression(df_sherlock_modelling)

if module == "svm":

    df_sherlock_modelling = joblib.load('./data/df_sherlock_modelling.lib')
    df_sherlock_modelling['period'] = df_sherlock_modelling['year'].apply(lambda x: 0 if x <= 1893 else 1)
    svm(df_sherlock_modelling)

if module == "narrative_arc":

    df_sherlock_modelling = joblib.load('./data/df_sherlock_modelling.lib')
    df_sherlock_modelling['period'] = df_sherlock_modelling['year'].apply(lambda x: 0 if x <= 1893 else 1)
    narrative_arc(df_sherlock_modelling)
