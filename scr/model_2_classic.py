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
    idx1_fit = min(fitnesses)
    idx1 = tournament[fitnesses.index(idx1_fit)]

    tournament = random.sample(population, tournament_size)
    fitnesses = [calculate_fitness(tour, distances) for tour in tournament]
    idx2_fit = min(fitnesses)
    idx2 = tournament[fitnesses.index(idx2_fit)]

    return idx1, idx1_fit, idx2, idx2_fit


def multi_point_crossover(parent1, parent2):
    """
    Perform multi-point crossover between two parents to create a child.
    """
    child = [-1] * len(parent1)

    num_points = random.randint(1, min(len(parent1), len(parent2)) - 1)
    crossover_points = sorted(random.sample(range(len(parent1)), num_points))

    for i in range(len(crossover_points) - 1):
        start_idx = crossover_points[i]
        end_idx = crossover_points[i + 1]
        child[start_idx:end_idx] = parent1[start_idx:end_idx]

    # Handle the last segment
    start_idx = crossover_points[-1]
    child[start_idx:] = parent1[start_idx:]

    for i in range(len(parent2)):
        if parent2[i] not in child:
            for j in range(len(child)):
                if child[j] == -1:
                    child[j] = parent2[i]
                    break
    return child


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
    return tour

def smart_mutate(tour, mutation_rate=2):
    """
    Mutate a tour by swapping two cities with a probability determined by the mutation rate.
    """
    mutated_tour = tour.copy()

    for i in range(1, len(mutated_tour)):
        if random.random() < mutation_rate:
            j = random.randint(1, len(mutated_tour) - 1)
            mutated_tour[i], mutated_tour[j] = mutated_tour[j], mutated_tour[i]

    return mutated_tour


def genetic_algorithm(distances, pop_size=100, num_generations=1000):
    """
    Solve the TSP using a genetic algorithm.
    """
    num_cities = len(distances)

    # Create initial population
    population = create_population_v2_cluster(pop_size)

    fitnesses = [calculate_fitness(tour, distances) for tour in population]
    min_fit_curr = min(fitnesses)
    min_fit = [min_fit_curr]

    # Iterate over generations
    for gen in range(num_generations):
        # Select parents
        parent1, parent1_fit, parent2, parent2_fit = select_parents(population, distances)

        # Crossover to create child
        # child = multi_point_crossover(parent1, parent2)
        child = crossover(parent1, parent2)
        # Mutate child
        # child = smart_mutate(tour=child)
        child = mutate(tour=child)
        # child, child_fit = two_opt(child, distances)
        child_fit = calculate_fitness(child, distances)

        # if child_fit < min(parent1_fit, parent2_fit):
            # Replace worst individual with child
        worst_idx = fitnesses.index(max(fitnesses))
        population[worst_idx] = child
        fitnesses[worst_idx] = child_fit
        if child_fit < min_fit_curr:
            min_fit.append(child_fit)
            min_fit_curr = child_fit
        else:
            min_fit.append(min_fit_curr)

    # Return best individual
    best_idx = fitnesses.index(min(fitnesses))

    return population[best_idx], min(fitnesses), min_fit


def test(num_generations=15_000, pop_size=1_00):
    a = genetic_algorithm(distances=tsp.graph.distance_df, num_generations=num_generations, pop_size=pop_size)

    import csv
    with open('my_list.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(a[0])
    import matplotlib.pyplot as plt

    plt.plot(a[2])
    plt.show()
    return a

a = test()
# a = test()
#
# import matplotlib.pyplot as plt
#
# plt.plot(a[2])
# plt.show()
