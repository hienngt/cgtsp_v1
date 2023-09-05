from t1 import Graph
# from model_2 import a
import pandas as pd

tour = [0, 11, 10, 12, 9, 56, 55, 52, 51, 49, 50, 21, 19, 17, 43, 44, 42, 41, 46, 47, 45, 48, 39, 37, 40, 38, 35, 33, 34, 36, 23, 22, 24, 31, 29, 32, 30, 26, 28, 25, 27, 18, 20, 7, 8, 5, 6, 15, 13, 4, 3, 1, 2, 14, 16, 54, 53, 58, 59, 60, 57, 62, 63, 64, 61]

df = pd.DataFrame(tour, columns=['index'])
corr_df = Graph().corr_df

df = df.merge(corr_df, how='left', on='index')[['index', 'cluster_id', 'subcluster_id']]
df['tag'] = df['cluster_id'].astype(str) + df['subcluster_id'].astype(str)

final_tour = []

final_tour.append(df.loc[0]['index'])
last_tag = df.loc[0]['tag']

for i in range(1, len(df), 1):
    current_tag = df.loc[i]['tag']
    if last_tag != current_tag:
        final_tour.append(df.loc[i]['index'])
        last_tag = current_tag


def calculate_fitness(tour, distances):
    """
    Calculate the fitness of a tour.
    """
    fitness = 0
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        city_i = tour[i]
        city_j = tour[j]
        fitness += distances[city_i][city_j]
    return fitness

tour_len = calculate_fitness(final_tour, Graph().distance_df)
