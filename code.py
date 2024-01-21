from source.preprocessing import preprocessing, preprocessing_narrative_coherence


df_sherlock = preprocessing()

df_sherlock = preprocessing_narrative_coherence(df_sherlock)

print(df_sherlock.head())