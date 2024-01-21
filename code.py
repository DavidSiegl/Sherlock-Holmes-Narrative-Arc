from source.preprocessing import preprocessing, preprocessing_narrative_coherence, preprocessing_temporal_usage


df_sherlock = preprocessing()

df_sherlock = preprocessing_narrative_coherence(df_sherlock)

df_sherlock = preprocessing_temporal_usage(df_sherlock)

print(df_sherlock.head())