import joblib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

df_sherlock_modeling = joblib.load('./data/df_sherlock_modeling.lib')
df_sherlock_modeling['period'] = df_sherlock_modeling['year'].apply(lambda x: 0 if x <= 1893 else 1)

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
    
    return None


def pca(df):
    
    return None