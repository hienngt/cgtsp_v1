from t1 import TSP
from model_1 import tsp

import numpy as np
import random

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
    population = create_population(num_cities, pop_size)

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

    # Return best individual
    fitnesses = [calculate_fitness(tour, distances) for tour in population]
    best_idx = fitnesses.index(min(fitnesses))

    return population[best_idx], min(fitnesses)

a = genetic_algorithm(distances=tsp.graph.distance_df)