from source.preprocessing import preprocessing, preprocessing_narrative_coherence, preprocessing_temporal_usage, preprocessing_ee, preprocessing_lemmatisation
from source.eda import eda
from source.feature_engineering import feature_engineering_narrative_coherence
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

df_sherlock = preprocessing()
df_sherlock = preprocessing_narrative_coherence(df_sherlock)
feature_engineering_narrative_coherence(df_sherlock)
