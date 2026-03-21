import random
import numpy as np
from deap import base, creator, tools

from fuzzy_system.fuzzy_layer import fuzzy_transform


# fitness function
def evaluate(individual, features, labels):

    mean1, sigma1, mean2, sigma2 = individual

    params = [mean1, sigma1, mean2, sigma2]

    fuzzy_features = fuzzy_transform(features, params)

    score = np.mean(fuzzy_features)

    return (score,)


def run_ga(features, labels):

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.uniform, 0, 1)

    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        4
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate,
        features=features,
        labels=labels
    )

    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)

    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=20)

    generations = 10

    for gen in range(generations):

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(child1, child2)

        for mutant in offspring:
            toolbox.mutate(mutant)

        fits = toolbox.map(toolbox.evaluate, offspring)

        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        population[:] = offspring

        print(f"Generation {gen} completed")

    best_ind = tools.selBest(population, 1)[0]

    return best_ind