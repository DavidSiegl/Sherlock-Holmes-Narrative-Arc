import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from source.functions import extract_events, compute_lsi_on_subsets, vectorize_lsi_results, compute_difference

def event_extraction(df_sherlock_segments_event):
    df_sherlock_segments_event['events'] = df_sherlock_segments_event['segments_event'].apply(extract_events)
    counts = []
    for row in df_sherlock_segments_event['events']:
        count = len(row)
        counts.append(count)
        
    df_sherlock_segments_event['events_count'] = counts
    summary_stats_event = df_sherlock_segments_event[['segment_num', 'events_count']].groupby('segment_num').agg('sum')
    
    plt.figure(1)
    sns.set_theme()
    sns.lineplot(data = summary_stats_event, x='segment_num', y='events_count')
    plt.gca().set_xticks(range(1, 6, 1))
    plt.title('Number of Events per Segment')
    plt.xlabel('Segments')
    plt.ylabel('Events Count')
    counts = {}

    for row in df_sherlock_segments_event['events']:
        for item in row:
            key_value = item['event']
            if key_value in counts:
                counts[key_value] += 1
            else:
                counts[key_value] = 1

    counts_df = pd.DataFrame.from_dict(counts, orient='index', columns=['Count'])

    grouped_counts = df_sherlock_segments_event.groupby('segment_num').apply(lambda x: x['events'].explode().apply(lambda y: y['event'] if isinstance(y, dict) else None).value_counts().head(5))
    
    plt.figure(2)
    sns.set_theme()
    grouped_counts.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('Segments/Events')
    plt.ylabel('Count')
    plt.title('Most common Events per Segment')
    plt.show()
    
    return None


def lsi(df_sherlock_segments_lsi):
    
    lsi_results = compute_lsi_on_subsets(df_sherlock_segments_lsi, 'segments_lsi', 'segment_num')
    
    category_vectors = vectorize_lsi_results(lsi_results)
    
    differences = compute_difference(category_vectors)
    categories = list(lsi_results.keys())
    average_diff = np.mean(differences, axis=1)

    sns.set_theme()
    sns.lineplot(x=categories, y=average_diff)
    plt.xlabel('Segments')
    plt.ylabel('Difference')
    plt.gca().set_xticks(range(1, 6, 1))
    plt.title('Change of Topics over Narrative Time')
    plt.tight_layout()
    plt.show()

    return None