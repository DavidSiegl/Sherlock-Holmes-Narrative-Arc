import warnings
import argparse
import joblib

from source.preprocessing import preprocessing, preprocessing_narrative_coherence, preprocessing_temporal_usage, preprocessing_ee, preprocessing_lemmatisation, preprocessing_emotion_analysis, preprocessing_ner, preprocessing_lsi
from source.eda import eda
from source.feature_engineering import feature_engineering_narrative_coherence, feature_engineering_temporal_usage, feature_engineering_emotion_analysis, feature_engineering_ner
from source.topic_extraction import event_extraction, lsi
from source.modeling import clustering, pca, logistic_regression, svm, narrative_arc
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

#df_sherlock = preprocessing()

#df_sherlock = preprocessing_narrative_coherence(df_sherlock)
#feature_engineering_narrative_coherence(df_sherlock)

#df_sherlock = preprocessing_temporal_usage(df_sherlock)
#feature_engineering_temporal_usage(df_sherlock)

#df_sherlock = preprocessing_emotion_analysis(df_sherlock)
#feature_engineering_emotion_analysis(df_sherlock)

#df_sherlock = preprocessing_ner(df_sherlock)
#feature_engineering_ner(df_sherlock)

#df_sherlock = preprocessing_ee(df_sherlock)
#event_extraction(df_sherlock)

# df_sherlock = preprocessing_lsi(df_sherlock)
# lsi(df_sherlock)

df_sherlock_modeling = joblib.load('./data/df_sherlock_modeling.lib')
df_sherlock_modeling['period'] = df_sherlock_modeling['year'].apply(lambda x: 0 if x <= 1893 else 1)

#pca(df_sherlock_modeling)

#logistic_regression(df_sherlock_modeling)

#svm(df_sherlock_modeling)

df_sherlock_modeling_cluster = clustering(df_sherlock_modeling)
narrative_arc(df_sherlock_modeling, df_sherlock_modeling_cluster)