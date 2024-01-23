import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np

from scipy.stats import f_oneway
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from source.functions import count_matches, min_max_scale_column, group_df, pos_tag_text, tense_counts, get_sentiment_scores, extract_entities, count_entities, most_common_entities

matplotlib.use('TkAgg')


values_to_drop = ['THE HOUND OF THE BASKERVILLES', 'THE VALLEY OF FEAR', 'A STUDY IN SCARLET', 'THE SIGN OF THE FOUR']

unique_colors = sns.color_palette(n_colors=5)

def feature_engineering_narrative_coherence(df_sherlock_segments_narrative):
    keywords_dict = pd.read_csv('./data/AON Dict - Without Overlaps.dic', sep='\t')

    keywords_dict.reset_index(inplace=True)

    keywords_dict = keywords_dict.iloc[4:]

    keywords_dict['index'] = keywords_dict['index'].str.replace('*', '')
    df_staging = keywords_dict.loc[keywords_dict['%'] == '1']
    staging = df_staging['index'].tolist()

    df_plot_progress = keywords_dict.loc[keywords_dict['%'] == '2']
    plot_progress = df_plot_progress['index'].tolist()

    df_cognitive_tension = keywords_dict.loc[keywords_dict['%'] == '3']
    cognitive_tension = df_cognitive_tension['index'].tolist()
    
    df_sherlock_segments_narrative_outliers = df_sherlock_segments_narrative.loc[df_sherlock_segments_narrative['title'].isin(values_to_drop)]

    df_sherlock_segments_narrative = df_sherlock_segments_narrative.loc[~df_sherlock_segments_narrative['title'].isin(values_to_drop)]
    
    df_sherlock_segments_narrative['staging_count'] = df_sherlock_segments_narrative.apply(lambda row: count_matches(row['segments_narrative'], staging), axis=1)

    df_sherlock_segments_narrative['plot_progress_count'] = df_sherlock_segments_narrative.apply(lambda row: count_matches(row['segments_narrative'], plot_progress), axis=1)

    df_sherlock_segments_narrative['cognitive_tension_count'] = df_sherlock_segments_narrative.apply(lambda row: count_matches(row['segments_narrative'], cognitive_tension), axis=1)

    df_sherlock_segments_narrative_outliers['staging_count'] = df_sherlock_segments_narrative_outliers.apply(lambda row: count_matches(row['segments_narrative'], staging), axis=1)

    df_sherlock_segments_narrative_outliers['plot_progress_count'] = df_sherlock_segments_narrative_outliers.apply(lambda row: count_matches(row['segments_narrative'], plot_progress), axis=1)

    df_sherlock_segments_narrative_outliers['cognitive_tension_count'] = df_sherlock_segments_narrative_outliers.apply(lambda row: count_matches(row['segments_narrative'], cognitive_tension), axis=1)
    
    df_sherlock_segments_narrative = min_max_scale_column(df_sherlock_segments_narrative, 'staging_count_norm', 'staging_count', 'segments_narrative')

    df_sherlock_segments_narrative = min_max_scale_column(df_sherlock_segments_narrative, 'plot_progress_count_norm', 'plot_progress_count', 'segments_narrative')

    df_sherlock_segments_narrative = min_max_scale_column(df_sherlock_segments_narrative, 'cognitive_tension_count_norm', 'cognitive_tension_count', 'segments_narrative')
    df_sherlock_segments_narrative_outliers = min_max_scale_column(df_sherlock_segments_narrative_outliers, 'staging_count_norm', 'staging_count', 'segments_narrative')

    df_sherlock_segments_narrative_outliers = min_max_scale_column(df_sherlock_segments_narrative_outliers, 'plot_progress_count_norm', 'plot_progress_count', 'segments_narrative')

    df_sherlock_segments_narrative_outliers = min_max_scale_column(df_sherlock_segments_narrative_outliers, 'cognitive_tension_count_norm', 'cognitive_tension_count', 'segments_narrative')
    
    summary_stats_narrative = df_sherlock_segments_narrative[['segment_num', 'staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm']].groupby('segment_num').agg(['mean', 'std'])
    
    plt.figure(1)
    sns.set_theme()
    for i, color in enumerate(unique_colors):
        sns.barplot(
            x=[summary_stats_narrative.index[i]],
            y=[summary_stats_narrative[('staging_count_norm', 'mean')].iloc[i]],
            yerr=[summary_stats_narrative[('staging_count_norm', 'std')].iloc[i]],
            color=color
        )
    plt.title('Mean/Std for Plot Staging per segment')
    plt.xlabel('Segment')
    plt.ylabel('Plot Staging Frequency')
    
    plt.figure(2)
    sns.set_theme()
    for i, color in enumerate(unique_colors):
        sns.barplot(
            x=[summary_stats_narrative.index[i]],
            y=[summary_stats_narrative[('plot_progress_count_norm', 'mean')].iloc[i]],
            yerr=[summary_stats_narrative[('plot_progress_count_norm', 'std')].iloc[i]],
            color=color
        )
    plt.title('Mean/Std for Plot Progression per segment')
    plt.xlabel('Segment')
    plt.ylabel('Plot Progression Frequency')
    plt.tight_layout()
    
    plt.figure(3)
    sns.set_theme()
    for i, color in enumerate(unique_colors):
        sns.barplot(
            x=[summary_stats_narrative.index[i]],
            y=[summary_stats_narrative[('cognitive_tension_count_norm', 'mean')].iloc[i]],
            yerr=[summary_stats_narrative[('cognitive_tension_count_norm', 'std')].iloc[i]],
            color=color
        )
    plt.title('Mean/Std for Cognitive Tension per segment')
    plt.xlabel('Segment')
    plt.ylabel('Plot Cognitive Tension Frequency')
    
    summary_stats_narrative = summary_stats_narrative.reset_index()
    summary_stats_narrative = pd.melt(summary_stats_narrative, id_vars=['segment_num'], var_name='category', value_name='value')
    summary_stats_narrative = summary_stats_narrative.drop(index=summary_stats_narrative[(summary_stats_narrative['category'] == 'level_0') | summary_stats_narrative['category'] =='index'].index)
    label_map = {'staging_count_norm' : 'Staging' , 'plot_progress_count_norm' : 'Plot Progression' , 'cognitive_tension_count_norm' : 'Cognitive Tension'}

    summary_stats_narrative['Plot Element'] = summary_stats_narrative['category'].replace(label_map)

    plt.figure(4)
    sns.set_theme()
    sns.boxplot(data = summary_stats_narrative, x='segment_num', y='value', hue='Plot Element')
    plt.title('Mean/Std for Narrative Coherence per segment')
    plt.xlabel('Segment')
    plt.ylabel('Narrative Coherence Frequency')
    
    summary_stats_narrative = df_sherlock_segments_narrative[['segment_num', 'staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm']].groupby('segment_num').agg(['mean'])

    summary_stats_narrative = summary_stats_narrative.reset_index()
    summary_stats_narrative = pd.melt(summary_stats_narrative, id_vars=['segment_num'], var_name='category', value_name='value')
    summary_stats_narrative = summary_stats_narrative.drop(index=summary_stats_narrative[(summary_stats_narrative['category'] == 'level_0') | summary_stats_narrative['category'] =='index'].index)

    label_map = {'staging_count_norm' : 'Staging' , 'plot_progress_count_norm' : 'Plot Progression' , 'cognitive_tension_count_norm' : 'Cognitive Tension'}

    summary_stats_narrative['Plot Element'] = summary_stats_narrative['category'].replace(label_map)
    
    plt.figure(5)
    sns.set_theme()
    sns.lineplot(data = summary_stats_narrative, x='segment_num', y='value', hue='Plot Element')
    plt.gca().set_xticks(range(1, 6, 1))
    plt.title('Mean Narrative Coherence over time')
    plt.xlabel('Segment')
    plt.ylabel('Narrative Coherence')
    
    ls_anova = group_df(df_sherlock_segments_narrative, 'staging_count_norm')

    fvalue_staging, pvalue_staging = f_oneway(ls_anova[0], ls_anova[1], ls_anova[2], ls_anova[3], ls_anova[4])
    print('Results for ANOVA test for plot staging (fvalue, pvalue): {}, {}'.format(fvalue_staging, pvalue_staging))
    ls_anova = group_df(df_sherlock_segments_narrative, 'plot_progress_count_norm')

    fvalue_progress, pvalue_progress = f_oneway(ls_anova[0], ls_anova[1], ls_anova[2], ls_anova[3], ls_anova[4])
    print('Results for ANOVA test for plot progression (fvalue, pvalue): {}, {}'.format(fvalue_progress, pvalue_progress))
    ls_anova = group_df(df_sherlock_segments_narrative, 'cognitive_tension_count_norm')

    fvalue_cog, pvalue_cog = f_oneway(ls_anova[0], ls_anova[1], ls_anova[2], ls_anova[3], ls_anova[4])
    print('Results for ANOVA test for cognitive tension (fvalue, pvalue): {}, {}'.format(fvalue_cog, pvalue_cog))
    col1 = [fvalue_staging, fvalue_progress, fvalue_cog, pvalue_staging, pvalue_progress, pvalue_cog]
    col2 = ['Staging', 'Plot Progression', 'Cognitive Tension', 'Staging', 'Plot Progression', 'Cognitive Tension']
    col3 = ['fvalue', 'fvalue', 'fvalue', 'pvalue', 'pvalue', 'pvalue']

    df_dict = {'value': col1, 'plot_element': col2, 'category_value': col3}

    df_anova_narrative = pd.DataFrame(df_dict)

    plt.figure(6)
    sns.set_theme()
    g = sns.barplot(data = df_anova_narrative, x='plot_element', y ='value', hue = 'category_value')
    g.get_legend().set_title("")
    plt.title('Results of ANOVA tests for Narrative Coherence per segment')
    plt.xlabel('Plot Element')
    plt.ylabel('Value')
    
    df_sherlock_segments_narrative_vec = df_sherlock_segments_narrative[['title', 'staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm']]

    df_sherlock_segments_narrative_vec = df_sherlock_segments_narrative_vec.groupby('title').agg(lambda x: x.tolist())

    df_sherlock_segments_narrative_vec['vector'] = df_sherlock_segments_narrative_vec.apply(lambda row: [val for sublist in row.values for val in sublist], axis=1)
    df_sherlock_segments_narrative_vec.reset_index(inplace=True)
    df_sherlock_segments_narrative_vec = df_sherlock_segments_narrative_vec[['title', 'vector']]
    matrix_vec = np.array(df_sherlock_segments_narrative_vec['vector'].tolist())

    similarity_matrix = cosine_similarity(matrix_vec)
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))

    sns.set(font_scale=1)
    fig, ax = plt.subplots(figsize=(30, 30))
    plt.figure(7)
    sns.heatmap(similarity_matrix, annot=False, cmap='Blues', mask=mask, ax=ax)
    plt.subplots_adjust(left=0.25, bottom=0.25, right=0.95, top=0.95)
    plt.title('Cosine similarity matrix')
    
    linkage_matrix = linkage(similarity_matrix, method='ward')

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.figure(8)
    dendrogram(linkage_matrix, labels=range(1,57))
    plt.title('Dendrogram of Cosine Similarity Matrix')
    plt.xlabel('Texts')
    plt.ylabel('Distance')
    
    cluster_assignments = fcluster(linkage_matrix, t=2, criterion = 'maxclust')

    df_sherlock_segments_narrative_vec['cluster'] = cluster_assignments
    df_sherlock_segments_narrative = pd.merge(df_sherlock_segments_narrative, df_sherlock_segments_narrative_vec[['title', 'cluster']], on='title', how='left') # adding the retrieved cluster assignment to the initial dataframe
    df_sherlock_pivot = df_sherlock_segments_narrative[['title', 'segment_num', 'staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm', 'cluster']]

    df_sherlock_pivot = pd.melt(df_sherlock_pivot, id_vars=['segment_num','title', 'cluster'], var_name='var', value_name= 'val')
    df_sherlock_pivot = df_sherlock_pivot.rename(columns={'cluster' : 'Cluster'})

    label_map = {'staging_count_norm' : 'Staging' , 'plot_progress_count_norm' : 'Plot Progression' , 'cognitive_tension_count_norm' : 'Cognitive Tension'}

    df_sherlock_pivot['Plot Element'] = df_sherlock_pivot['var'].replace(label_map)

    fg = sns.FacetGrid(df_sherlock_pivot, col='Cluster')
    fg.map_dataframe(sns.lineplot, x='segment_num', y='val', hue='Plot Element')
    fg.add_legend()
    fg.set(xticks = range(1, 6, 1))
    fg.fig.suptitle('Narrative Coherence per Cluster')
    fg.set_axis_labels('Segments' , 'Narrative Coherence')
   
    df_sherlock_pivot_cluster1 = df_sherlock_pivot.loc[df_sherlock_pivot['Cluster'] == 1]

    df_sherlock_pivot_cluster2 = df_sherlock_pivot.loc[df_sherlock_pivot['Cluster'] == 2]
    
    df_sherlock_pivot = df_sherlock_segments_narrative_outliers[['title', 'segment_num', 'staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm']]

    df_sherlock_pivot = pd.melt(df_sherlock_pivot, id_vars=['segment_num','title'], var_name='var', value_name= 'val')

    df_sherlock_pivot = df_sherlock_pivot.rename(columns={'title' : 'Title'})

    label_map = {'staging_count_norm' : 'Staging' , 'plot_progress_count_norm' : 'Plot Progression' , 'cognitive_tension_count_norm' : 'Cognitive Tension'}

    df_sherlock_pivot['Plot Element'] = df_sherlock_pivot['var'].replace(label_map)
    
    fg = sns.FacetGrid(df_sherlock_pivot, row='Title')

    fg.map_dataframe(sns.lineplot, x='segment_num', y='val', hue='Plot Element')
    fg.add_legend(loc='center right', bbox_to_anchor=(1.2, 0.25))
    fg.fig.subplots_adjust(top=0.85)
    fg.set(xticks = range(1, 6, 1))
    fg.fig.suptitle('Narrative Coherence per Title')
    fg.set_axis_labels('Segments' , 'Narrative Coherence')
    fg.set_titles(row_template="{row_name}")
    
    plt.show()

    return None


def feature_engineering_temporal_usage(df_sherlock_segments_temp):
    
    df_sherlock_segments_temp_outliers = df_sherlock_segments_temp.loc[df_sherlock_segments_temp['title'].isin(values_to_drop)]

    df_sherlock_segments_temp = df_sherlock_segments_temp.loc[~df_sherlock_segments_temp['title'].isin(values_to_drop)]
    
    df_sherlock_segments_temp['text_pos'] = df_sherlock_segments_temp['segments_temp'].apply(pos_tag_text)

    df_sherlock_segments_temp_outliers['text_pos'] = df_sherlock_segments_temp_outliers['segments_temp'].apply(pos_tag_text)
    
    df_sherlock_segments_temp[['past_count', 'present_count', 'future_count']] = df_sherlock_segments_temp['text_pos'].apply(tense_counts)

    df_sherlock_segments_temp_outliers[['past_count', 'present_count', 'future_count']] = df_sherlock_segments_temp_outliers['text_pos'].apply(tense_counts)
    
    df_sherlock_segments_temp = min_max_scale_column(df_sherlock_segments_temp, 'future_count_norm', 'future_count', 'segments_temp')

    df_sherlock_segments_temp = min_max_scale_column(df_sherlock_segments_temp, 'past_count_norm', 'past_count', 'segments_temp')

    df_sherlock_segments_temp = min_max_scale_column(df_sherlock_segments_temp, 'present_count_norm', 'present_count', 'segments_temp')
    df_sherlock_segments_temp_outliers = min_max_scale_column(df_sherlock_segments_temp_outliers, 'future_count_norm', 'future_count', 'segments_temp')

    df_sherlock_segments_temp_outliers = min_max_scale_column(df_sherlock_segments_temp_outliers, 'past_count_norm', 'past_count', 'segments_temp')

    df_sherlock_segments_temp_outliers = min_max_scale_column(df_sherlock_segments_temp_outliers, 'present_count_norm', 'present_count', 'segments_temp')
    df_sherlock_pivot = df_sherlock_segments_temp[['title', 'segment_num', 'future_count_norm', 'past_count_norm', 'present_count_norm']]

    df_sherlock_pivot = pd.melt(df_sherlock_pivot, id_vars=['segment_num','title'], var_name='var', value_name= 'val')
    summary_stats_tenses = df_sherlock_segments_temp[['segment_num', 'present_count_norm', 'future_count_norm', 'past_count_norm']].groupby('segment_num').agg(['mean', 'std'])
    
    plt.figure(1)
    sns.set_theme()
    for i, color in enumerate(unique_colors):
        sns.barplot(
            x=[summary_stats_tenses.index[i]],
            y=[summary_stats_tenses[('present_count_norm', 'mean')].iloc[i]],
            yerr=[summary_stats_tenses[('present_count_norm', 'std')].iloc[i]],
            color=color
        )
    plt.title('Mean/Std for Present Tense Usage per segment')
    plt.xlabel('Segment')
    plt.ylabel('Present Tense Frequency')
    
    plt.figure(2)
    sns.set_theme()
    for i, color in enumerate(unique_colors):
        sns.barplot(
            x=[summary_stats_tenses.index[i]],
            y=[summary_stats_tenses[('past_count_norm', 'mean')].iloc[i]],
            yerr=[summary_stats_tenses[('past_count_norm', 'std')].iloc[i]],
            color=color
        )
    plt.title('Mean/Std for Past Tense Usage per segment')
    plt.xlabel('Segment')
    plt.ylabel('Past Tense Frequency')
    
    plt.figure(3)
    sns.set_theme()
    for i, color in enumerate(unique_colors):
        sns.barplot(
            x=[summary_stats_tenses.index[i]],
            y=[summary_stats_tenses[('future_count_norm', 'mean')].iloc[i]],
            yerr=[summary_stats_tenses[('future_count_norm', 'std')].iloc[i]],
            color=color
        )
    plt.title('Mean/Std for Future Tense Usage per segment')
    plt.xlabel('Segment')
    plt.ylabel('Future Tense Frequency')
    
    summary_stats_tenses = summary_stats_tenses.reset_index()
    summary_stats_tenses = pd.melt(summary_stats_tenses, id_vars=['segment_num'], var_name='stat', value_name='value')
    summary_stats_tenses = summary_stats_tenses.drop(index=summary_stats_tenses[(summary_stats_tenses['stat'] == 'level_0') | summary_stats_tenses['stat'] =='index'].index)
    label_map = {'past_count_norm' : 'Past Tense' , 'present_count_norm' : 'Present Tense' , 'future_count_norm' : 'Future Tense'}

    summary_stats_tenses['Tenses'] = summary_stats_tenses['stat'].replace(label_map)

    plt.figure(4)
    sns.set_theme()
    sns.boxplot(data = summary_stats_tenses, x='segment_num', y='value', hue='Tenses')
    plt.title('Mean/Std for Tense Usage per segment')
    plt.xlabel('Segment')
    plt.ylabel('Tense Usage Frequency')
    
    summary_stats_tenses = df_sherlock_segments_temp[['segment_num', 'present_count_norm', 'future_count_norm', 'past_count_norm']].groupby('segment_num').agg(['mean'])

    summary_stats_tenses = summary_stats_tenses.reset_index()
    summary_stats_tenses = pd.melt(summary_stats_tenses, id_vars=['segment_num'], var_name='stat', value_name='value')
    summary_stats_tenses = summary_stats_tenses.drop(index=summary_stats_tenses[(summary_stats_tenses['stat'] == 'level_0') | summary_stats_tenses['stat'] =='index'].index)

    label_map = {'past_count_norm' : 'Past Tense' , 'present_count_norm' : 'Present Tense' , 'future_count_norm' : 'Future Tense'}

    summary_stats_tenses['Tenses'] = summary_stats_tenses['stat'].replace(label_map)
    
    plt.figure(5)
    sns.set_theme()
    sns.lineplot(data = summary_stats_tenses, x='segment_num', y='value', hue='Tenses')
    plt.gca().set_xticks(range(1, 6, 1))
    plt.title('Mean Temporal Usage over time')
    plt.xlabel('Segment')
    plt.ylabel('Temporal Usage')
    
    ls_anova = group_df(df_sherlock_segments_temp, 'past_count_norm')

    fvalue_past, pvalue_past = f_oneway(ls_anova[0], ls_anova[1], ls_anova[2], ls_anova[3], ls_anova[4])
    print('Results for ANOVA test for past tense usage (fvalue, pvalue): {}, {}'.format(fvalue_past, pvalue_past))
    ls_anova = group_df(df_sherlock_segments_temp, 'present_count_norm')

    fvalue_present, pvalue_present = f_oneway(ls_anova[0], ls_anova[1], ls_anova[2], ls_anova[3], ls_anova[4])
    print('Results for ANOVA test for present tense usage (fvalue, pvalue): {}, {}'.format(fvalue_present, pvalue_present))
    ls_anova = group_df(df_sherlock_segments_temp, 'future_count_norm')

    fvalue_future, pvalue_future = f_oneway(ls_anova[0], ls_anova[1], ls_anova[2], ls_anova[3], ls_anova[4])
    print('Results for ANOVA test for future tense usage (fvalue, pvalue): {}, {}'.format(fvalue_future, pvalue_future))
    col1 = [fvalue_past, fvalue_present, fvalue_future, pvalue_past, pvalue_present, pvalue_future]
    col2 = ['Past Tense', 'Present Tense', 'Future Tense', 'Past Tense', 'Present Tense', 'Future Tense']
    col3 = ['fvalue', 'fvalue', 'fvalue', 'pvalue', 'pvalue', 'pvalue']

    df_dict = {'value': col1, 'tense': col2, 'category_value': col3}

    df_anova_tenses = pd.DataFrame(df_dict)

    plt.figure(6)
    sns.set_theme()
    g = sns.barplot(data = df_anova_tenses, x='tense', y ='value', hue = 'category_value')
    g.get_legend().set_title("")
    plt.title('Results of ANOVA tests for Temporal Usage per segment')
    plt.xlabel('Tenses')
    plt.ylabel('Value')
    
    df_sherlock_segments_temp_vec = df_sherlock_segments_temp[['title', 'present_count_norm', 'past_count_norm', 'future_count_norm']]

    df_sherlock_segments_temp_vec = df_sherlock_segments_temp_vec.groupby('title').agg(lambda x: x.tolist())

    df_sherlock_segments_temp_vec['vector'] = df_sherlock_segments_temp_vec.apply(lambda row: [val for sublist in row.values for val in sublist], axis=1)
    df_sherlock_segments_temp_vec.reset_index(inplace=True)
    df_sherlock_segments_temp_vec = df_sherlock_segments_temp_vec[['title', 'vector']]  
    matrix_vec = np.array(df_sherlock_segments_temp_vec['vector'].tolist())

    similarity_matrix = cosine_similarity(matrix_vec)

    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))

    sns.set(font_scale=1)
    fig, ax = plt.subplots(figsize=(30, 30))
    plt.figure(7)
    sns.heatmap(similarity_matrix, annot=False, cmap='Blues', mask=mask, ax=ax)
    plt.subplots_adjust(left=0.25, bottom=0.25, right=0.95, top=0.95)
    plt.title('Cosine similarity matrix')
    
    linkage_matrix = linkage(similarity_matrix, method='ward')

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=range(1,57))
    plt.title('Dendrogram of Cosine Similarity Matrix')
    plt.xlabel('Texts')
    plt.ylabel('Distance')
    cluster_assignments = fcluster(linkage_matrix, t=2, criterion = 'maxclust')

    df_sherlock_segments_temp_vec['cluster'] = cluster_assignments
    df_sherlock_segments_temp = pd.merge(df_sherlock_segments_temp, df_sherlock_segments_temp_vec[['title', 'cluster']], on='title', how='left')
    df_sherlock_pivot = df_sherlock_segments_temp[['title', 'segment_num', 'past_count_norm', 'present_count_norm', 'future_count_norm', 'cluster']]

    df_sherlock_pivot = pd.melt(df_sherlock_pivot, id_vars=['segment_num','title', 'cluster'], var_name='var', value_name= 'val')

    df_sherlock_pivot = df_sherlock_pivot.rename(columns={'cluster' : 'Cluster'})

    label_map = {'past_count_norm' : 'Past Tense' , 'present_count_norm' : 'Present Tense' , 'future_count_norm' : 'Future Tense'}

    df_sherlock_pivot['Tenses'] = df_sherlock_pivot['var'].replace(label_map)

    fg = sns.FacetGrid(df_sherlock_pivot, col='Cluster')

    fg.map_dataframe(sns.lineplot, x='segment_num', y='val', hue='Tenses')
    fg.set(xticks = range(1, 6, 1))
    fg.fig.suptitle('Temporal Usage per Cluster')
    fg.set_axis_labels('Segments' , 'Temporal Usage')

    fg.add_legend()

    df_sherlock_pivot_cluster1 = df_sherlock_pivot.loc[df_sherlock_pivot['Cluster'] == 1]

    df_sherlock_pivot_cluster2 = df_sherlock_pivot.loc[df_sherlock_pivot['Cluster'] == 2]

    df_sherlock_pivot = df_sherlock_segments_temp_outliers[['title', 'segment_num', 'past_count_norm', 'present_count_norm', 'future_count_norm']]

    df_sherlock_pivot = pd.melt(df_sherlock_pivot, id_vars=['segment_num','title'], var_name='var', value_name= 'val')
    df_sherlock_pivot = df_sherlock_pivot.rename(columns={'title' : 'Title'})

    label_map = {'past_count_norm' : 'Past Tense' , 'present_count_norm' : 'Present Tense' , 'future_count_norm' : 'Future Tense'}

    df_sherlock_pivot['Tenses'] = df_sherlock_pivot['var'].replace(label_map)

    fg = sns.FacetGrid(df_sherlock_pivot, row='Title')

    fg.map_dataframe(sns.lineplot, x='segment_num', y='val', hue='Tenses')
    fg.add_legend(loc='center right', bbox_to_anchor=(1.2, 0.25))
    fg.fig.subplots_adjust(top=0.85)
    fg.set(xticks = range(1, 6, 1))
    fg.fig.suptitle('Temporal Usage per Title')
    fg.set_axis_labels('Segments' , 'Temporal Usage')
    fg.set_titles(row_template="{row_name}")
    
    plt.show()
    
    return None


def feature_engineering_emotion_analysis(df_sherlock_segments_vader):
        
    df_sherlock_segments_vader_outliers = df_sherlock_segments_vader.loc[df_sherlock_segments_vader['title'].isin(values_to_drop)]

    df_sherlock_segments_vader = df_sherlock_segments_vader.loc[~df_sherlock_segments_vader['title'].isin(values_to_drop)]
    
    df_sherlock_segments_vader['sentiment_score'] = df_sherlock_segments_vader['segments_vader'].apply(get_sentiment_scores)

    get_avg_compound_score = lambda scores: sum([score for score in scores])/len([score for score in scores])
    df_sherlock_segments_vader['avg_compound_score'] = df_sherlock_segments_vader['sentiment_score'].apply(get_avg_compound_score)

    df_sherlock_segments_vader = df_sherlock_segments_vader.drop('sentiment_score', axis=1)
    
    df_sherlock_segments_vader_outliers['sentiment_score'] = df_sherlock_segments_vader_outliers['segments_vader'].apply(get_sentiment_scores)

    df_sherlock_segments_vader_outliers['avg_compound_score'] = df_sherlock_segments_vader_outliers['sentiment_score'].apply(get_avg_compound_score)

    df_sherlock_segments_vader_outliers = df_sherlock_segments_vader_outliers.drop('sentiment_score', axis=1)
    summary_stats_emotion = df_sherlock_segments_vader[['segment_num', 'avg_compound_score']].groupby('segment_num').agg(['mean', 'std'])
    
    plt.figure(1)
    sns.set_theme()
    for i, color in enumerate(unique_colors):
        sns.barplot(
            x=[summary_stats_emotion.index[i]],
            y=[summary_stats_emotion[('avg_compound_score', 'mean')].iloc[i]],
            yerr=[summary_stats_emotion[('avg_compound_score', 'std')].iloc[i]],
            color=color
        )
    plt.title('Mean/Std for Compound Score per segment')
    plt.xlabel('Segment')
    plt.ylabel('Average Compound Score')
    
    summary_stats_emotion = df_sherlock_segments_vader[['segment_num', 'avg_compound_score']].groupby('segment_num').agg(['mean'])

    summary_stats_emotion = summary_stats_emotion.reset_index()
    summary_stats_emotion = pd.melt(summary_stats_emotion, id_vars=['segment_num'], var_name='stat', value_name='value')
    summary_stats_emotion = summary_stats_emotion.drop(index=summary_stats_emotion[(summary_stats_emotion['stat'] == 'level_0') | summary_stats_emotion['stat'] =='index'].index)
    
    plt.figure(2)
    sns.set_theme()
    sns.lineplot(data = summary_stats_emotion, x='segment_num', y='value')
    plt.gca().set_xticks(range(1, 6, 1))
    plt.title('Mean Compound Score over time')
    plt.xlabel('Segment')
    plt.ylabel('Average Compound Score')
    ls_anova = group_df(df_sherlock_segments_vader, 'avg_compound_score')

    fvalue_emotion, pvalue_emotion = f_oneway(ls_anova[0], ls_anova[1], ls_anova[2], ls_anova[3], ls_anova[4])
    print('Results for ANOVA test for emotion analysis (fvalue, pvalue): {}, {}'.format(fvalue_emotion, pvalue_emotion))
    
    df_sherlock_pivot = df_sherlock_segments_vader[['title', 'segment_num', 'avg_compound_score']]

    df_sherlock_pivot = pd.melt(df_sherlock_pivot, id_vars=['segment_num','title'], var_name='var', value_name= 'val')
    df_sherlock_segments_vader_vec = df_sherlock_segments_vader[['title', 'avg_compound_score']]

    df_sherlock_segments_vader_vec = df_sherlock_segments_vader_vec.groupby('title').agg(lambda x: x.tolist())

    df_sherlock_segments_vader_vec.reset_index(inplace=True)
    matrix_vec = np.array(df_sherlock_segments_vader_vec['avg_compound_score'].tolist())

    similarity_matrix = cosine_similarity(matrix_vec)
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))

    sns.set(font_scale=1)
    fig, ax = plt.subplots(figsize=(30, 30))
    plt.figure(3)
    sns.heatmap(similarity_matrix, annot=False, cmap='Blues', mask=mask, ax=ax)
    plt.subplots_adjust(left=0.25, bottom=0.25, right=0.95, top=0.95)
    plt.title('Cosine similarity matrix')
    
    linkage_matrix = linkage(similarity_matrix, method='ward')

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.figure(4)
    dendrogram(linkage_matrix, labels=range(1,57))
    plt.title('Dendrogram of Cosine Similarity Matrix')
    plt.xlabel('Items')
    plt.ylabel('Distance')
    
    cluster_assignments = fcluster(linkage_matrix, t=3, criterion = 'maxclust')

    df_sherlock_segments_vader_vec['cluster'] = cluster_assignments
    df_sherlock_segments_vader = pd.merge(df_sherlock_segments_vader, df_sherlock_segments_vader_vec[['title', 'cluster']], on='title', how='left')
    df_sherlock_pivot = df_sherlock_segments_vader[['title', 'segment_num', 'avg_compound_score', 'cluster']]
    df_sherlock_pivot = pd.melt(df_sherlock_pivot, id_vars=['segment_num','title', 'cluster'], var_name='var', value_name= 'val')
    df_sherlock_pivot = df_sherlock_pivot.rename(columns={'cluster' : 'Cluster'})

    fg = sns.FacetGrid(df_sherlock_pivot, col='Cluster')

    fg.map_dataframe(sns.lineplot, x='segment_num', y='val')
    fg.set(xticks = range(1, 6, 1))
    fg.fig.suptitle('Emotion Analysis per Cluster')
    fg.set_axis_labels('Segments' , 'Average Compound Score')
    
    df_sherlock_pivot_cluster1 = df_sherlock_pivot.loc[df_sherlock_pivot['Cluster'] == 1]

    df_sherlock_pivot_cluster2 = df_sherlock_pivot.loc[df_sherlock_pivot['Cluster'] == 2]

    df_sherlock_pivot_cluster3 = df_sherlock_pivot.loc[df_sherlock_pivot['Cluster']== 3]
    df_sherlock_pivot = df_sherlock_segments_vader[['title', 'segment_num', 'avg_compound_score']]

    df_sherlock_pivot = pd.melt(df_sherlock_pivot, id_vars=['segment_num','title'], var_name='var', value_name= 'val')
    df_sherlock_pivot = df_sherlock_pivot.rename(columns={'title' : 'Title'})

    fg = sns.FacetGrid(df_sherlock_pivot, col='Title', col_wrap=8)

    fg.map_dataframe(sns.lineplot, x='segment_num', y='val')
    fg.fig.subplots_adjust(top=0.8, wspace=1, hspace=1.5)
    fg.set(xticks = range(1, 6, 1))
    fg.fig.suptitle('Emotion Analysis per Title')
    fg.set_axis_labels('Segments' , 'Average Compound Score', labelpad=35)
    fg.set_titles(col_template="{col_name}", rotation=25)
    
    df_sherlock_pivot = df_sherlock_segments_vader_outliers[['title', 'segment_num', 'avg_compound_score']]

    df_sherlock_pivot = pd.melt(df_sherlock_pivot, id_vars=['segment_num','title'], var_name='var', value_name= 'val')
    df_sherlock_pivot = df_sherlock_pivot.rename(columns={'title' : 'Title'})

    fg = sns.FacetGrid(df_sherlock_pivot, row='Title')

    fg.map_dataframe(sns.lineplot, x='segment_num', y='val')
    fg.fig.subplots_adjust(top=0.85)
    fg.set(xticks = range(1, 6, 1))
    fg.fig.suptitle('Emotion Analysis per Title')
    fg.set_axis_labels('Segments' , 'Average Compound Score')
    fg.set_titles(row_template="{row_name}")
    
    plt.show()
    
    return None


def feature_engineering_ner(df_sherlock_segments_ner):
    
    df_sherlock_segments_ner_outliers = df_sherlock_segments_ner.loc[df_sherlock_segments_ner['title'].isin(values_to_drop)]

    df_sherlock_segments_ner = df_sherlock_segments_ner.loc[~df_sherlock_segments_ner['title'].isin(values_to_drop)]
    df_sherlock_segments_ner['entities'] = df_sherlock_segments_ner['segments_ner'].apply(extract_entities)
    df_sherlock_segments_ner_outliers['entities'] = df_sherlock_segments_ner_outliers['segments_ner'].apply(extract_entities)
    
    df_sherlock_segments_ner['person_entity_count'] = df_sherlock_segments_ner['entities'].apply(lambda parse_tree: count_entities(parse_tree, 'PERSON'))

    df_sherlock_segments_ner['location_entity_count'] = df_sherlock_segments_ner['entities'].apply(lambda parse_tree: count_entities(parse_tree, 'LOCATION', 'GPE', 'FACILITY'))
    df_sherlock_segments_ner_outliers['person_entity_count'] = df_sherlock_segments_ner_outliers['entities'].apply(lambda parse_tree: count_entities(parse_tree, 'PERSON'))

    df_sherlock_segments_ner_outliers['location_entity_count'] = df_sherlock_segments_ner_outliers['entities'].apply(lambda parse_tree: count_entities(parse_tree, 'LOCATION', 'GPE', 'FACILITY'))
 
    most_common_persons = df_sherlock_segments_ner['entities'].apply(lambda row: most_common_entities(row, ['PERSON']))
    print("Most common persons \n", most_common_persons.value_counts().nlargest(10))
    most_common_locations = df_sherlock_segments_ner['entities'].apply(lambda row: most_common_entities(row, ['LOCATION', 'GPE', 'FACILITY']))
    print("Most common locations \n", most_common_locations.value_counts().nlargest(10))
    
    df_sherlock_segments_ner = min_max_scale_column(df_sherlock_segments_ner, 'person_entity_count_norm', 'person_entity_count', 'text_prepro_ner')

    df_sherlock_segments_ner = min_max_scale_column(df_sherlock_segments_ner, 'location_entity_count_norm', 'location_entity_count', 'text_prepro_ner')
    df_sherlock_segments_ner_outliers = min_max_scale_column(df_sherlock_segments_ner_outliers, 'person_entity_count_norm', 'person_entity_count', 'text_prepro_ner')

    df_sherlock_segments_ner_outliers = min_max_scale_column(df_sherlock_segments_ner_outliers, 'location_entity_count_norm', 'location_entity_count', 'text_prepro_ner')
    summary_stats_ner = df_sherlock_segments_ner[['segment_num', 'person_entity_count_norm', 'location_entity_count_norm']].groupby('segment_num').agg(['mean', 'std'])
    
    plt.figure(1)
    sns.set_theme()
    for i, color in enumerate(unique_colors):
        sns.barplot(
            x=[summary_stats_ner.index[i]],
            y=[summary_stats_ner[('person_entity_count_norm', 'mean')].iloc[i]],
            yerr=[summary_stats_ner[('person_entity_count_norm', 'std')].iloc[i]],
            color=color
        )
    plt.title('Mean/Std for Person Count per segment')
    plt.xlabel('Segment')
    plt.ylabel('Person Count')
    
    plt.figure(2)
    sns.set_theme()
    for i, color in enumerate(unique_colors):
        sns.barplot(
            x=[summary_stats_ner.index[i]],
            y=[summary_stats_ner[('location_entity_count_norm', 'mean')].iloc[i]],
            yerr=[summary_stats_ner[('location_entity_count_norm', 'std')].iloc[i]],
            color=color
        )
    plt.title('Mean/Std for Location Count per segment')
    plt.xlabel('Segment')
    plt.ylabel('Location Count')
    
    summary_stats_ner = df_sherlock_segments_ner[['segment_num', 'person_entity_count_norm', 'location_entity_count_norm']].groupby('segment_num').agg(['mean'])

    summary_stats_ner = summary_stats_ner.reset_index()
    summary_stats_ner = pd.melt(summary_stats_ner, id_vars=['segment_num'], var_name='stat', value_name='value')
    summary_stats_ner = summary_stats_ner.drop(index=summary_stats_ner[(summary_stats_ner['stat'] == 'level_0') | summary_stats_ner['stat'] =='index'].index)

    label_map = {'person_entity_count_norm' : 'PERSON' , 'location_entity_count_norm' : 'LOCATION'}

    summary_stats_ner['Entities'] = summary_stats_ner['stat'].replace(label_map)
    
    plt.figure(3)
    sns.set_theme()
    sns.lineplot(data = summary_stats_ner, x='segment_num', y='value', hue='Entities')
    plt.gca().set_xticks(range(1, 6, 1))
    plt.title('Linguistic Drift over time')
    plt.xlabel('Segment')
    plt.ylabel('Linguistic Drift')
    ls_anova = group_df(df_sherlock_segments_ner, 'person_entity_count_norm')

    fvalue_person, pvalue_person = f_oneway(ls_anova[0], ls_anova[1], ls_anova[2], ls_anova[3], ls_anova[4])
    print('Results for ANOVA test for person entities (fvalue, pvalue): {}, {}'.format(fvalue_person, pvalue_person))
    ls_anova = group_df(df_sherlock_segments_ner, 'location_entity_count_norm')

    fvalue_location, pvalue_location = f_oneway(ls_anova[0], ls_anova[1], ls_anova[2], ls_anova[3], ls_anova[4])
    print('Results for ANOVA test for location entities (fvalue, pvalue): {}, {}'.format(fvalue_location, pvalue_location))
    
    col1 = [fvalue_person, fvalue_location, pvalue_person, pvalue_location]
    col2 = ['Person', 'Location', 'Person', 'Location']
    col3 = ['fvalue', 'fvalue', 'pvalue', 'pvalue']

    df_dict = {'value': col1, 'plot_element': col2, 'category_value': col3}

    df_anova_ner = pd.DataFrame(df_dict)

    plt.figure(4)
    sns.set_theme()
    g = sns.barplot(data = df_anova_ner, x='plot_element', y ='value', hue = 'category_value')
    plt.xlabel('NER')
    plt.ylabel('Value')
    g.get_legend().set_title("")
    plt.title('Results of ANOVA tests for NER per segment')

    df_sherlock_segments_ner_vec = df_sherlock_segments_ner[['title', 'person_entity_count_norm', 'location_entity_count_norm']]

    df_sherlock_segments_ner_vec = df_sherlock_segments_ner_vec.groupby('title').agg(lambda x: x.tolist())

    df_sherlock_segments_ner_vec['vector'] = df_sherlock_segments_ner_vec.apply(lambda row: [val for sublist in row.values for val in sublist], axis=1)

    df_sherlock_segments_ner_vec.reset_index(inplace=True)

    df_sherlock_segments_ner_vec = df_sherlock_segments_ner_vec[['title', 'vector']]

    matrix_vec = np.array(df_sherlock_segments_ner_vec['vector'].tolist())
    similarity_matrix = cosine_similarity(matrix_vec)
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))

    sns.set(font_scale=1)
    fig, ax = plt.subplots(figsize=(30, 30))
    plt.figure(5)
    sns.heatmap(similarity_matrix, annot=False, cmap='Blues', mask=mask, ax=ax)
    plt.subplots_adjust(left=0.25, bottom=0.25, right=0.95, top=0.95)
    plt.title('Cosine similarity matrix')
    linkage_matrix = linkage(similarity_matrix, method='ward')

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.figure(6)
    dendrogram(linkage_matrix, labels=range(1,57))
    plt.title('Dendrogram of Cosine Similarity Matrix')
    plt.xlabel('Texts')
    plt.ylabel('Distance')
    plt.show()
    
    cluster_assignments = fcluster(linkage_matrix, t=2, criterion = 'maxclust')

    df_sherlock_segments_ner_vec['cluster'] = cluster_assignments

    df_sherlock_segments_ner = pd.merge(df_sherlock_segments_ner, df_sherlock_segments_ner_vec[['title', 'cluster']], on='title', how='left')

    df_sherlock_pivot = df_sherlock_segments_ner[['title', 'segment_num', 'person_entity_count_norm', 'location_entity_count_norm', 'cluster']]

    df_sherlock_pivot = pd.melt(df_sherlock_pivot, id_vars=['segment_num','title', 'cluster'], var_name='var', value_name= 'val')
    df_sherlock_pivot = df_sherlock_pivot.rename(columns={'cluster' : 'Cluster'})

    label_map = {'person_entity_count_norm' : 'PERSON' , 'location_entity_count_norm' : 'LOCATION'}
    df_sherlock_pivot['Entities'] = df_sherlock_pivot['var'].replace(label_map)

    fg = sns.FacetGrid(df_sherlock_pivot, col='Cluster')

    fg.map_dataframe(sns.lineplot, x='segment_num', y='val', hue='Entities')

    fg.set(xticks = range(1, 6, 1))

    fg.fig.suptitle('NER per Cluster')
    fg.set_axis_labels('Segments' , 'Linguistic Drift')

    fg.add_legend()
    
    df_sherlock_pivot_cluster1 = df_sherlock_pivot.loc[df_sherlock_pivot['Cluster'] == 1]

    df_sherlock_pivot_cluster2 = df_sherlock_pivot.loc[df_sherlock_pivot['Cluster'] == 2]
    
    df_sherlock_pivot = df_sherlock_segments_ner_outliers[['title', 'segment_num', 'person_entity_count_norm', 'location_entity_count_norm']]

    df_sherlock_pivot = pd.melt(df_sherlock_pivot, id_vars=['segment_num','title'], var_name='var', value_name= 'val')

    df_sherlock_pivot = df_sherlock_pivot.rename(columns={'title' : 'Title'})

    label_map = {'person_entity_count_norm' : 'PERSON' , 'location_entity_count_norm' : 'LOCATION'}

    df_sherlock_pivot['Entities'] = df_sherlock_pivot['var'].replace(label_map)

    fg = sns.FacetGrid(df_sherlock_pivot, row='Title')

    fg.map_dataframe(sns.lineplot, x='segment_num', y='val', hue='Entities')
    fg.add_legend(loc='center right', bbox_to_anchor=(1.2, 0.25))
    fg.fig.subplots_adjust(top=0.85)
    fg.set(xticks = range(1, 6, 1))
    fg.fig.suptitle('NER per Title')
    fg.set_axis_labels('Segments' , 'Linguistic Drift')
    fg.set_titles(row_template="{row_name}")
    
    plt.show()
    
    return None