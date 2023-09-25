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


def circular_shift(tour, random_number):
    if random_number in tour:
        shift_index = tour.index(random_number)
        shifted_tour = tour[shift_index:] + tour[:shift_index]
        # print(shifted_tour)
        return shifted_tour

def create_population_v3_subcluster_v3(pop_size=100, start_point=None):
    """
    Create a population of tours.
    """
    if start_point is None:
        start_point = [0]

    population_lst = []
    corr_df_2 = corr_df.query('index != 0').reset_index(drop=True)

    for _ in range(pop_size):
        tour_cluster = []

        for c in range(1, num_clusters + 1):
            df_inside_cluster = corr_df_2[corr_df_2['cluster_id'] == c].reset_index(drop=True)
            subclusters = df_inside_cluster['subcluster_id'].unique()

            if len(subclusters) > 0:
                tour_subcluster = []

                for s in subclusters:
                    df_inside_subcluster = df_inside_cluster[df_inside_cluster['subcluster_id'] == s].reset_index(drop=True)
                    tour = df_inside_subcluster['index'].tolist()
                    random_number = random.choice(tour)
                    tour_subcluster.append(circular_shift(tour, random_number))

                random.shuffle(tour_subcluster)
                tour_cluster.append(tour_subcluster)

        random.shuffle(tour_cluster)
        flattened_tour = list(chain(*list(chain(*tour_cluster))))
        population_lst.append([start_point] + tour_cluster + [start_point + flattened_tour])

    population = pd.DataFrame(population_lst)
    population.rename(columns={population.columns[-1]: 'tour'}, inplace=True)

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

    def tournament_selection(tournament):
        return tournament.loc[tournament['fitness'].idxmin()]

    tournament1 = population.sample(tournament_size)
    parent1 = tournament_selection(tournament1)

    tournament2 = population.sample(tournament_size)
    parent2 = tournament_selection(tournament2)

    return parent1, parent2


def crossover_v2(parent1, parent2):
    """
    Perform crossover between two parents to create a child.
    """
    len_parent = parent1.shape[0]

    child = [[-1]] * (len(parent1) - 2)

    start_idx = random.randint(0, len_parent - 3)
    end_idx = random.randint(start_idx + 1, len_parent - 2)

    child[start_idx:end_idx] = parent1[start_idx:end_idx]

    existing_values = [set(list(chain(*lst))) if lst not in ([-1], [0]) else lst for lst in child]
    for i in range(len_parent - 2):
        if parent2[i] == [0]:
            child[i] = [0]
        elif parent2[i] != [-1]:
            if set(list(chain(*parent2[i]))) not in existing_values:
                for j in range(len(child)):
                    if child[j] == [-1]:
                        child[j] = parent2[i]
                        existing_values.append(set(chain(*parent2[i])))
                        break

    return child

def mutate_v2(child):
    mutate_rate = random.randint(1, len(child) - 1)
    i = 0
    while i < mutate_rate:
        random_cluster = random.randrange(1, len(child) - 1)
        cluster = child[random_cluster]
        random_subcluster = random.randrange(1, len(cluster) - 1)
        subcluster = cluster[random_subcluster]

        if len(subcluster) > 1:
            random_number = random.choice(subcluster)
            child[random_cluster][random_subcluster] = circular_shift(subcluster, random_number)
            i += 1

    return child

from tqdm import tqdm

def genetic_algorithm(pop_size=100, num_generations=1000):
    """
    Solve the TSP using a genetic algorithm.
    """
    # Create initial population
    population = create_population_v3_subcluster_v3(pop_size=pop_size)
    distances = tsp.graph.distance_df
    population['fitness'] = population['tour'].apply(lambda x: calculate_fitness(x, distances))

    min_fit_curr = population['fitness'].min()
    min_fit = [min_fit_curr]

    worst_idx = population['fitness'].idxmax()
    fitness_max = population.loc[worst_idx]['fitness']

    # Iterate over generations
    for gen in tqdm(range(num_generations)):
        # Select parents
        parent1, parent2 = select_parents(population)

        i = 0
        while True:
            # Crossover to create child
            child = crossover_v2(parent1, parent2)
            # Mutate child
            child = mutate_v2(child)

            child.append([0] + list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(child[1:])))))
            child_fit = calculate_fitness(child[-1], distances)
            child.append(child_fit)

            if child_fit < max(parent1['fitness'], parent2['fitness']):
                break
            else:
                i += 1
            if i == 5:
                break

        if child_fit < fitness_max:

            population.loc[len(population)] = child
            population = population.drop(worst_idx).reset_index(drop=True)


            worst_idx = population['fitness'].idxmax()
            fitness_max = population.loc[worst_idx]['fitness']

        if child_fit < min_fit_curr:
            min_fit_curr = child_fit

        min_fit.append(min_fit_curr)

    # Return best individual
    best_idx = population['fitness'].idxmin()
    return population.loc[best_idx], min_fit_curr, min_fit


a = genetic_algorithm(1000, 30_000)
import csv

with open('my_list.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(a[0]['tour'])

import seaborn as sns

# Plotting the list
sns.lineplot(data=a[2])

# Display the plot
sns.plt.show()