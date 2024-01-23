import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import silhouette_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def clustering(df_sherlock_modeling):
    df_sherlock_modeling_cluster = df_sherlock_modeling[['title', 'staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm', 'past_count_norm', 'present_count_norm', 'future_count_norm', 'avg_compound_score', 'person_entity_count_norm', 'location_entity_count_norm', 'period']]

    df_sherlock_modeling_cluster = df_sherlock_modeling_cluster.groupby('title').agg('mean')
    kmeans = KMeans(n_clusters=2, random_state=19)

    kmeans.fit(df_sherlock_modeling_cluster)
    labels = kmeans.labels_

    df_sherlock_modeling_cluster['cluster'] = labels
    silhouette_avg = silhouette_score(df_sherlock_modeling_cluster, kmeans.labels_)

    print("Silhouette Score:", silhouette_avg)
    centroids = kmeans.cluster_centers_
    inter_distances = np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2)

    intra_distances = []
    for i in range(2):
        cluster_points = df_sherlock_modeling_cluster[df_sherlock_modeling_cluster['cluster'] == i]
        intra_dist = np.mean(np.linalg.norm(cluster_points.values[:, np.newaxis] - cluster_points.values, axis=2))
        intra_distances.append(intra_dist)

    min_inter = np.min(inter_distances[np.nonzero(inter_distances)])
    max_intra = np.max(intra_distances)
    dunn_index = min_inter / max_intra

    print("Dunn Index:", dunn_index)
    
    sns.set_theme()
    sns.countplot(data=df_sherlock_modeling_cluster, x='cluster', hue='cluster')
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title = 'Cluster')
    plt.show()
    
    print(df_sherlock_modeling_cluster["cluster"].sort_values())
    
    return df_sherlock_modeling_cluster


def pca(df_sherlock_modeling):
    
    pca = PCA(n_components=2)

    df_sherlock_modeling_pca = df_sherlock_modeling[['title', 'staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm', 'past_count_norm', 'present_count_norm', 'future_count_norm', 'avg_compound_score', 'person_entity_count_norm', 'location_entity_count_norm']].groupby('title').agg('mean')

    df_sherlock_modeling_pca2 = pca.fit_transform(df_sherlock_modeling_pca[[ 'staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm', 'past_count_norm', 'present_count_norm', 'future_count_norm', 'avg_compound_score', 'person_entity_count_norm', 'location_entity_count_norm']])

    df_sherlock_modeling_pca2 = pd.DataFrame(df_sherlock_modeling_pca2, columns=['pca1', 'pca2'])
    
    plt.figure(1)
    sns.set_theme()
    sns.scatterplot(data=df_sherlock_modeling_pca2, x="pca1", y="pca2")
    plt.title('PCA1 vs PCA2')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    pca = PCA(n_components=2)

    df_sherlock_modeling_pca = df_sherlock_modeling[['title', 'period', 'staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm', 'past_count_norm', 'present_count_norm', 'future_count_norm', 'avg_compound_score', 'person_entity_count_norm', 'location_entity_count_norm']].groupby('title').agg('mean')

    df_sherlock_modeling_pca2 = pca.fit_transform(df_sherlock_modeling_pca[[ 'staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm', 'past_count_norm', 'present_count_norm', 'future_count_norm', 'avg_compound_score', 'person_entity_count_norm', 'location_entity_count_norm', 'period']])

    df_sherlock_modeling_pca2 = pd.DataFrame(df_sherlock_modeling_pca2, columns=['pca1', 'pca2'])
    
    plt.figure(2)
    sns.set_theme()
    sns.scatterplot(data=df_sherlock_modeling_pca2, x="pca1", y="pca2")
    plt.title('PCA1 vs PCA2')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
     
    pca = PCA(n_components=1)

    df_sherlock_modeling_pca1 = pca.fit_transform(df_sherlock_modeling_pca[[ 'staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm', 'past_count_norm', 'present_count_norm', 'future_count_norm', 'avg_compound_score', 'person_entity_count_norm', 'location_entity_count_norm', 'period']])

    df_sherlock_modeling_pca1 = pd.DataFrame(df_sherlock_modeling_pca1, columns=[ 'pca'])
    df_sherlock_modeling_pca.reset_index(inplace=True)

    df_sub = df_sherlock_modeling_pca[['title', 'period']]

    df_sherlock_modeling_pca1 = pd.concat([df_sherlock_modeling_pca1, df_sub], axis=1)
    
    plt.figure(3)
    sns.set_theme()
    sns.scatterplot(data=df_sherlock_modeling_pca1, x="period", y="pca")
    plt.title('PCA vs Period')
    plt.xlabel('Period')
    plt.ylabel('PCA')
    plt.show()
    
    return None


def logistic_regression(df_sherlock_modeling):
    
    X_train, X_test, y_train, y_test = train_test_split(df_sherlock_modeling[['past_count_norm', 'future_count_norm', 'avg_compound_score']], df_sherlock_modeling['segment_num'], test_size=0.2, random_state=19)

    logreg = LogisticRegression(multi_class='ovr', solver='liblinear')

    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    results_logreg = classification_report(y_test, y_pred)
    print(results_logreg)
    
    return None


def svm(df_sherlock_modeling):
    
    X_train, X_test, y_train, y_test = train_test_split(df_sherlock_modeling[['past_count_norm', 'future_count_norm', 'avg_compound_score']], df_sherlock_modeling['segment_num'], test_size=0.2, random_state=19)

    svm = SVC(kernel='rbf', decision_function_shape='ovo')

    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    results_svm = classification_report(y_test, y_pred)
    print(results_svm)

    X_train, X_test, y_train, y_test = train_test_split(df_sherlock_modeling[['staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm', 'past_count_norm', 'present_count_norm', 'future_count_norm', 'avg_compound_score', 'person_entity_count_norm', 'location_entity_count_norm']], df_sherlock_modeling['period'], test_size=0.2, random_state=19)

    svm = SVC()

    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    results_svm = classification_report(y_test, y_pred)
    print(results_svm)
    
    return None


def narrative_arc(df_sherlock_modeling):
    
    df_sherlock_modeling_cluster = df_sherlock_modeling[['title', 'staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm', 'past_count_norm', 'present_count_norm', 'future_count_norm', 'avg_compound_score', 'person_entity_count_norm', 'location_entity_count_norm', 'period']]

    df_sherlock_modeling_cluster = df_sherlock_modeling_cluster.groupby('title').agg('mean')
    kmeans = KMeans(n_clusters=2, random_state=19)

    kmeans.fit(df_sherlock_modeling_cluster)
    labels = kmeans.labels_

    df_sherlock_modeling_cluster['cluster'] = labels
    
    df_sherlock_modeling_cluster.reset_index(inplace=True)

    df_sherlock_modeling = pd.merge(df_sherlock_modeling, df_sherlock_modeling_cluster[['title', 'cluster']], on='title', how='left')
    pca = PCA(n_components=1)

    df_sherlock_modeling_pca_arc = pca.fit_transform(df_sherlock_modeling[['staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm', 'past_count_norm', 'present_count_norm', 'future_count_norm', 'avg_compound_score', 'person_entity_count_norm','location_entity_count_norm']])

    df_sherlock_modeling_pca_arc = pd.DataFrame(df_sherlock_modeling_pca_arc, columns=['pca'])
    df_sub = df_sherlock_modeling[['segment_num', 'cluster']]

    df_sherlock_modeling_pca_arc = pd.concat([df_sherlock_modeling_pca_arc, df_sub], axis=1)
    df_sherlock_modeling_pca_arc = df_sherlock_modeling_pca_arc.groupby(['cluster', 'segment_num']).agg('mean')
    
    plt.figure(1)
    sns.set_theme()
    sns.lineplot(df_sherlock_modeling_pca_arc, x='segment_num', y='pca', hue='cluster')
    plt.title('Narrative Arc per Cluster')
    plt.xlabel('Segments')
    plt.legend(title='Cluster')
    plt.gca().set_xticks(range(1, 6, 1))
    plt.ylabel('PCA')
    
    df_sherlock_modeling_pca_arc = pca.fit_transform(df_sherlock_modeling[['staging_count_norm', 'plot_progress_count_norm', 'cognitive_tension_count_norm', 'past_count_norm', 'present_count_norm', 'future_count_norm', 'avg_compound_score', 'person_entity_count_norm','location_entity_count_norm']])

    df_sherlock_modeling_pca_arc = pd.DataFrame(df_sherlock_modeling_pca_arc, columns=['pca'])
    df_sub = df_sherlock_modeling[['title', 'period', 'segment_num']]

    df_sherlock_modeling_pca_arc = pd.concat([df_sherlock_modeling_pca_arc, df_sub], axis=1)
    df_sherlock_modeling_pca_arc = df_sherlock_modeling_pca_arc.rename(columns={'period' : 'Period'})

    fg = sns.FacetGrid(df_sherlock_modeling_pca_arc, col='Period')

    fg.map_dataframe(sns.lineplot, x='segment_num', y='pca')

    fg.fig.subplots_adjust(top=0.85)

    fg.set(xticks = range(1, 6, 1))

    fg.fig.suptitle('Narrative Arc per Period')
    fg.set_axis_labels('Segments', 'PCA')
    plt.show()
    
    return None