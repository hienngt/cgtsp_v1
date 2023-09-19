import random
import numpy as np
from deap import base, creator, tools
from model_1 import tsp
# Ma trận khoảng cách giữa các thành phố
distance_matrix = tsp.graph.distance_df

# Số lượng thành phố
num_cities = distance_matrix.shape[0]

# Khởi tạo các công cụ DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Định nghĩa hàm mục tiêu
def tsp_distance(individual):
    distance = 0
    for i in range(len(individual) - 1):
        city1 = individual[i]
        city2 = individual[i + 1]
        distance += distance_matrix[city1][city2]
    distance += distance_matrix[individual[-1]][individual[0]]  # Khoảng cách từ thành phố cuối cùng đến thành phố đầu tiên
    return distance,

# Đăng ký các công cụ cho quần thể và cá thể
toolbox.register("indices", random.sample, range(num_cities), num_cities)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", tsp_distance)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(0)

    # Khởi tạo quần thể
    population = toolbox.population(n=100)

    # Evaluate fitness
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Tiến hóa quần thể
    for generation in range(5_000):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        fits = [ind.fitness.values[0] for ind in population]

        best_index = fits.index(min(fits))
        best_individual = population[best_index]

        print("Generation:", generation, "Best distance:", best_individual.fitness.values[0])

    best_individual = tools.selBest(population, k=1)[0]

    best_order = [city for city in best_individual]

    print("Best order:", best_order)

if __name__ == "__main__":
    main()