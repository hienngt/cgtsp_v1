from model_1 import tsp

import random
import pandas as pd
from itertools import chain

distance_df = pd.DataFrame(index=range(tsp.graph.num_nodes), columns=range(tsp.graph.num_nodes))

num_clusters = tsp.graph.num_clusters
num_subclusters = tsp.graph.num_subclusters

corr_df = tsp.graph.corr_df
cluster_df = tsp.graph.cluster_df


def create_population_v2_cluster(pop_size=100, start_point=None):
    """
    Create a population of tours.
    """
    if start_point is None:
        start_point = [0]
    population_lst = []

    for i in range(pop_size):
        tour_cluster = []
        for c in range(2, num_clusters + 1):
            df_inside_cluster = corr_df.query(f'cluster_id == {c}').reset_index(drop=True)
            tour = df_inside_cluster['index'].to_list()
            random.shuffle(tour)
            tour_cluster.append(tour)
        random.shuffle(tour_cluster)

        population_lst.append(start_point + tour_cluster + [start_point + list(chain(*tour_cluster))])

    population = pd.DataFrame(population_lst)
    last_column_index = population.columns[-1]
    population.rename(columns={last_column_index: 'tour'}, inplace=True)
    return population


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


population = create_population_v2_cluster()
distances=tsp.graph.distance_df
population['fitness'] = population['tour'].apply(lambda x: calculate_fitness(x, distances))


def select_parents(population):
    """
    Select two parents from the population using tournament selection.
    """
    tournament_size = 3
    tournament = population.sample(tournament_size)
    idx1 = tournament.loc[tournament['fitness'].idxmin()]

    tournament = population.sample(tournament_size)
    idx2 = tournament.loc[tournament['fitness'].idxmin()]

    return idx1, idx2


def crossover(parent1, parent2):
    """
    Perform crossover between two parents to create a child.
    """
    len_parent = parent1.shape[0]

    start_idx = random.randint(0, len_parent - 3)
    end_idx = random.randint(start_idx + 1, len_parent - 2)

    if start_idx == 0:
        if end_idx == len_parent - 2:
            child = parent1[start_idx:end_idx]
        else:
            child = pd.concat([parent1[start_idx:end_idx], parent1[end_idx:len_parent - 2]])
    else:
        child = pd.concat([parent2[0:start_idx], parent1[start_idx:end_idx], parent2[end_idx:len_parent - 2]])

    return child


