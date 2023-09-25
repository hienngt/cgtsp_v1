import itertools

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

        population_lst.append([start_point] + tour_cluster + [start_point + list(chain(*tour_cluster))])

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


def select_parents(population):
    """
    Select two parents from the population using tournament selection.
    """
    tournament_size = 3
    tournament = population.sample(tournament_size)
    parent1 = tournament.loc[tournament['fitness'].idxmin()]

    tournament = population.sample(tournament_size)
    parent2 = tournament.loc[tournament['fitness'].idxmin()]

    return parent1, parent2


def crossover(parent1, parent2):
    """
    Perform crossover between two parents to create a child.
    """
    len_parent = parent1.shape[0]

    child = [[-1]] * (len(parent1) - 2)

    start_idx = random.randint(0, len_parent - 3)
    end_idx = random.randint(start_idx + 1, len_parent - 2)

    child[start_idx:end_idx] = parent1[start_idx:end_idx]

    for i in range(len_parent - 2):
        if set(parent2[i]) not in [set(tuple(lst)) for lst in child]:
            for j in range(len(child)):
                if child[j] == [-1]:
                    child[j] = parent2[i]
                    break

    # child.append(list(itertools.chain.from_iterable(child)))
    return child


def mutate(child):
    """
    Mutate a tour by swapping two cities.
    """
    random_cluster = random.randint(1, len(child) - 1)
    cluster = child[random_cluster]
    idx1 = random.randint(0, len(cluster) - 1)
    idx2 = random.randint(0, len(cluster) - 1)

    cluster[idx1], cluster[idx2] = cluster[idx2], cluster[idx1]
    child[random_cluster] = cluster
    return child


def smart_mutate(child, mutation_rate):
    """
    Mutate a tour by swapping two cities with a probability determined by the mutation rate.
    """
    for cluster_idx in range(1, len(child)):
        cluster = child[cluster_idx]
        for city_idx in range(len(cluster)):
            if random.random() < mutation_rate:
                swap_idx = random.randint(0, len(cluster) - 1)
                cluster[city_idx], cluster[swap_idx] = cluster[swap_idx], cluster[city_idx]
        child[cluster_idx] = cluster
    return child


def genetic_algorithm(pop_size=100, num_generations=1000):
    """
    Solve the TSP using a genetic algorithm.
    """
    # Create initial population
    population = create_population_v2_cluster(pop_size=pop_size)
    distances=tsp.graph.distance_df
    population['fitness'] = population['tour'].apply(lambda x: calculate_fitness(x, distances))

    min_fit_curr = population['fitness'].min()
    min_fit = [min_fit_curr]

    worst_idx = population['fitness'].idxmax()
    fitness_max = population.loc[worst_idx]['fitness']

    # Iterate over generations
    for gen in range(num_generations):
        # Select parents
        parent1, parent2 = select_parents(population)

        # Crossover to create child
        child = crossover(parent1, parent2)
        # Mutate child
        # child = mutate(child)

        child = smart_mutate(child, mutation_rate=3)
        child.append(list(itertools.chain.from_iterable(child)))
        child_fit = calculate_fitness(child[-1], distances)
        child.append(child_fit)

        if child_fit < fitness_max:
            population.loc[len(population)] = child
            population = population.drop(worst_idx).reset_index(drop=True)

            worst_idx = population['fitness'].idxmax()
            fitness_max = population.loc[worst_idx]['fitness']

        if child_fit < min_fit_curr:
            min_fit.append(child_fit)
            min_fit_curr = child_fit
        else:
            min_fit.append(min_fit_curr)


    # Return best individual

    return population.loc[population['fitness'].idxmin()], min_fit_curr, min_fit

a = genetic_algorithm(100, 10_000)
import csv
with open('my_list.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(a[0]['tour'])

import seaborn as sns

# Plotting the list
sns.lineplot(data=a[2])

# Display the plot
sns.plt.show()