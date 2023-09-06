from model_1 import tsp

import random
import pandas as pd
from itertools import chain

distance_df = pd.DataFrame(index=range(tsp.graph.num_nodes), columns=range(tsp.graph.num_nodes))


num_clusters = tsp.graph.num_clusters
num_subclusters = tsp.graph.num_subclusters

corr_df = tsp.graph.corr_df
cluster_df = tsp.graph.cluster_df


def create_population(num_cities, pop_size, start_point=0):
    """
    Create a population of tours.
    """
    population = []
    for i in range(pop_size):
        tour1 = [start_point]
        tour2 = list(range(1, num_cities))
        random.shuffle(tour2)
        population.append(tour1 + tour2)
    return population


def create_population_v2_cluster(pop_size, start_point=None):
    """
    Create a population of tours.
    """
    if start_point is None:
        start_point = [0]
    population = []

    for i in range(pop_size):
        tour_cluster = []
        for c in range(2, num_clusters + 1):
            num_subcluster_in_cluster = cluster_df.loc[c].index.get_level_values('subcluster_id').nunique()
            # print(f'{num_subcluster_in_cluster=}')

            df_inside_cluster = corr_df.query(f'cluster_id == {c}').reset_index(drop=True)
            tour = df_inside_cluster['index'].to_list()
            random.shuffle(tour)
            tour_cluster.append(tour)
        random.shuffle(tour_cluster)

        population.append(start_point + list(chain(*tour_cluster)))
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


def select_parents(population, distances):
    """
    Select two parents from the population using tournament selection.
    """
    tournament_size = 3
    tournament = random.sample(population, tournament_size)
    fitnesses = [calculate_fitness(tour, distances) for tour in tournament]
    idx1 = tournament[fitnesses.index(min(fitnesses))]

    tournament = random.sample(population, tournament_size)
    fitnesses = [calculate_fitness(tour, distances) for tour in tournament]
    idx2 = tournament[fitnesses.index(min(fitnesses))]

    return idx1, idx2


def crossover(parent1, parent2):
    """
    Perform crossover between two parents to create a child.
    """
    child = [-1] * len(parent1)

    start_idx = random.randint(0, len(parent1) - 1)
    end_idx = random.randint(start_idx + 1, len(parent1))

    child[start_idx:end_idx] = parent1[start_idx:end_idx]

    for i in range(len(parent2)):
        if parent2[i] not in child:
            for j in range(len(child)):
                if child[j] == -1:
                    child[j] = parent2[i]
                    break

    return child


def mutate(tour):
    """
    Mutate a tour by swapping two cities.
    """
    idx1 = random.randint(1, len(tour) - 1)
    idx2 = random.randint(1, len(tour) - 1)

    tour[idx1], tour[idx2] = tour[idx2], tour[idx1]


def genetic_algorithm(distances, pop_size=100, num_generations=1000):
    """
    Solve the TSP using a genetic algorithm.
    """
    num_cities = len(distances)

    # Create initial population
    population = create_population_v2_cluster(pop_size)

    fitnesses = [calculate_fitness(tour, distances) for tour in population]
    min_fit = [min(fitnesses)]
    # Iterate over generations
    for gen in range(num_generations):
        # Select parents
        parent1, parent2 = select_parents(population, distances)

        # Crossover to create child
        child = crossover(parent1, parent2)

        # Mutate child
        mutate(child)

        # Replace worst individual with child
        fitnesses = [calculate_fitness(tour, distances) for tour in population]
        worst_idx = fitnesses.index(max(fitnesses))
        population[worst_idx] = child
        min_fit.append(min(fitnesses))

    # Return best individual
    fitnesses = [calculate_fitness(tour, distances) for tour in population]
    best_idx = fitnesses.index(min(fitnesses))

    return population[best_idx], min(fitnesses), min_fit


def test(num_generations = 1000, pop_size=100):
    a = genetic_algorithm(distances=tsp.graph.distance_df, num_generations=num_generations, pop_size=pop_size)

    import csv
    with open('my_list.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(a[0])

    import matplotlib.pyplot as plt

    plt.plot(a[2])
    plt.show()
