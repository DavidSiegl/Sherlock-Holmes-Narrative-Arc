from source.preprocessing import preprocessing, preprocessing_narrative_coherence, preprocessing_temporal_usage, preprocessing_ee, preprocessing_lemmatisation, preprocessing_emotion_analysis
from source.eda import eda
from source.feature_engineering import feature_engineering_narrative_coherence, feature_engineering_temporal_usage, feature_engineering_emotion_analysis
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

df_sherlock = preprocessing()

#df_sherlock = preprocessing_narrative_coherence(df_sherlock)
#feature_engineering_narrative_coherence(df_sherlock)

# df_sherlock = preprocessing_temporal_usage(df_sherlock)
# feature_engineering_temporal_usage(df_sherlock)

df_sherlock = preprocessing_emotion_analysis(df_sherlock)
feature_engineering_emotion_analysis(df_sherlock)
