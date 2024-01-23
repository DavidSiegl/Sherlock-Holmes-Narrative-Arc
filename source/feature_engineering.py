import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import nltk
import numpy as np
from scipy.stats import f_oneway
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

matplotlib.use('TkAgg')
from source.functions import count_matches, min_max_scale_column, group_df


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

    plt.figure(9)
    fg = sns.FacetGrid(df_sherlock_pivot, col='Cluster')

    fg.map_dataframe(sns.lineplot, x='segment_num', y='val', hue='Plot Element')
    fg.add_legend()
    fg.fig.subplots_adjust(top=0.7, bottom=-0.25)
    fg.set(xticks = range(1, 6, 1))
    fg.fig.suptitle('Narrative Coherence per Cluster')
    fg.set_axis_labels('Segments' , 'Narrative Coherence')
   
    df_sherlock_pivot_cluster1 = df_sherlock_pivot.loc[df_sherlock_pivot['Cluster'] == 1]

    df_sherlock_pivot_cluster2 = df_sherlock_pivot.loc[df_sherlock_pivot['Cluster'] == 2]

    df_sherlock_pivot_cluster2['title'].unique()
    
    df_sherlock_pivot = df_sherlock_segments_narrative_outliers[['title', 'segment_num', 'staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm']]

    df_sherlock_pivot = pd.melt(df_sherlock_pivot, id_vars=['segment_num','title'], var_name='var', value_name= 'val')

    df_sherlock_pivot = df_sherlock_pivot.rename(columns={'title' : 'Title'})

    label_map = {'staging_count_norm' : 'Staging' , 'plot_progress_count_norm' : 'Plot Progression' , 'cognitive_tension_count_norm' : 'Cognitive Tension'}

    df_sherlock_pivot['Plot Element'] = df_sherlock_pivot['var'].replace(label_map)
    
    plt.figure(10)
    fg = sns.FacetGrid(df_sherlock_pivot, row='Title')

    fg.map_dataframe(sns.lineplot, x='segment_num', y='val', hue='Plot Element')
    fg.add_legend(loc='center right', bbox_to_anchor=(1.2, 0.25))
    fg.fig.subplots_adjust(top=0.9, bottom=-0.25)
    fg.set(xticks = range(1, 6, 1))
    fg.fig.suptitle('Narrative Coherence per Title')
    fg.set_axis_labels('Segments' , 'Narrative Coherence')
    fg.set_titles(row_template="{row_name}")
    
    plt.show()

    return None