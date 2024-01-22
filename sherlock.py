from source.preprocessing import preprocessing, preprocessing_narrative_coherence, preprocessing_temporal_usage, preprocessing_ee, preprocessing_lemmatisation
from source.eda import eda

df_sherlock = preprocessing()
df_sherlock = preprocessing_temporal_usage(df_sherlock)
df_sherlock = preprocessing_lemmatisation(df_sherlock)
df_sherlock = eda(df_sherlock)
