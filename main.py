# This is a sample Python script.
import numpy as np

from Constraints import ReverseRadiusConstraint
from FitnessFunctions import ConstraintPenaltyFitness, SimpleFitnessFunction
from MultimodalBinaryGenetic import MultimodalBinaryGenetic, BinaryCrossoverType


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# The function that we want to find it's maximum
from SimpleBinaryGeneticAlgorithm import SimpleBinaryGenetic


def trigonometric_exponential_function(x: []):
    value = (1 + np.cos(2 * np.pi * x[0] * x[1])) * np.exp(-(np.abs(x[0]) + np.abs(x[1])) / 2)
    return value


def gaussian_function(x: []):
    value = 1-np.exp(-0.5*(np.linalg.norm(x-2.5)/2.5)**2)
    return value


def run_multimodal():
    # Instantiating simple binary genetic algorithm object
    sbg = MultimodalBinaryGenetic(
        n_chromosome=126,
        m_gene=2,
        genes_lengths=np.array([8, 8]),
        genes_intervals=np.array([[-4, 2], [-1.5, 1]]),
        target_function=trigonometric_exponential_function,
        use_linear_ranking=False,
        use_sigma_limited=False,
        use_tournament_selection=False,
        crossover_type=BinaryCrossoverType.SimpleCrossOver
    )

    # Beginning execution of the algorithm
    sbg.begin_learning(2000)
    print('aaaa')


def run_simple():
    # Instantiating simple binary genetic algorithm object

    fitness_func = ConstraintPenaltyFitness(SimpleFitnessFunction(gaussian_function, is_maximization=False),
                                           [ReverseRadiusConstraint(5)])
    # fitness_func = SimpleFitnessFunction(gaussian_function, is_maximization=False)

    sbg = SimpleBinaryGenetic(
        n_chromosome=126,
        m_gene=2,
        fitness_function=fitness_func,
        genes_lengths=np.array([8, 8]),
        genes_intervals=np.array([[-4, 2], [-1.5, 1]]),
        target_function=gaussian_function,
        is_maximization=False,
        use_linear_ranking=False,
        use_sigma_limited=False,
        use_tournament_selection=False,
        crossover_type=BinaryCrossoverType.SimpleCrossOver
    )

    # Beginning execution of the algorithm
    sbg.begin_learning(2000)


if __name__ == '__main__':
    # run_multimodal()
    run_simple()