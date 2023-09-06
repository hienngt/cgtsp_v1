from t1 import Graph
# from model_2 import a
import pandas as pd


import csv
with open('my_list.csv', newline='') as file:
    reader = csv.reader(file)
    my_list = list(reader)
print(my_list)

from itertools import chain

tour = [int(i) for i in list(chain(*my_list))]

print(tour)

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
