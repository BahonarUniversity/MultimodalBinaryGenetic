from abc import ABC, abstractmethod
from typing import List

import numpy as np

from Constraints import Constraint


class FitnessFunction(ABC):
    def __init__(self):
        self.fitness: np.ndarray = None

    @abstractmethod
    def calculate(self, decoded_chromosomes: np.ndarray, objectives_count: int = 1):
        pass

    def get_max_fitness(self, count: int = 1):
        return self.fitness[np.argpartition(self.fitness, -count)[-count:]]

    def get_mean(self):
        return np.mean(self.fitness)


class SimpleFitnessFunction(FitnessFunction):

    def __init__(self, target_function, is_maximization: bool = True):
        super().__init__()
        self.target_function = target_function
        self.is_maximization = is_maximization

    def calculate(self, decoded_chromosomes: np.ndarray, objectives_count: int = 1):
        self.fitness = np.ndarray((decoded_chromosomes.shape[0],))
        self.target_function = self.target_function
        # print('calculated')
        function_values = []
        for i in range(decoded_chromosomes.shape[0]):
            f_value = self.target_function(decoded_chromosomes[i])
            function_values.append(f_value)

        self.fitness = np.array(function_values)
        if self.is_maximization:
            if min(function_values) < 0:
                self.fitness += min(function_values)
        else:
            self.fitness = max(function_values) - self.fitness

        return self.fitness


class ConstraintPenaltyFitness(FitnessFunction):

    def __init__(self, fitness_function: FitnessFunction, constraints: List[Constraint], k_factor: float = 2,
                 lambda_term: float = 0.01, update_interval: int = 10, beta_1: float = 0.6, beta_2: float = 0.7):
        super().__init__()
        self.fitness_function = fitness_function
        self.constraints = constraints
        self.k_factor = k_factor
        self.lambda_term = lambda_term
        self.update_interval = update_interval
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self._iterator = 0
        self._case1_iterator = 0
        self._case2_iterator = 0

    def calculate(self, decoded_chromosomes: np.ndarray, objectives_count: int = 1):
        self.fitness = self.fitness_function.calculate(decoded_chromosomes)
        best_chromosomes = self.fitness_function.get_max_fitness(objectives_count)
        self.__update_iterators(best_chromosomes)
        if self._iterator < self.update_interval:
            return self.fitness
        self.__reset__iterators()
        self.__update_lambda()
        for i in range(len(self.fitness)):
            penalty_term = self.fitness[i] + self.__penalty_term(decoded_chromosomes[i])
            self.fitness[i] = penalty_term
        return self.fitness

    def __penalty_term(self, chromosome):
        penalty_sum = 0
        for i in range(len(self.constraints)):
            penalty_sum += self.constraints[i].calculate_distance(chromosome)**self.k_factor
        return penalty_sum * self.lambda_term

    def __update_iterators(self, best_chromosomes):
        is_feasible = True
        for i in range(len(self.constraints)):
            for j in range(len(best_chromosomes)):
                if not self.constraints[i].is_feasible(best_chromosomes[j]):
                    is_feasible = False
                    break;
            if not is_feasible:
                break
        if is_feasible:
            self._iterator += 1
            self._case1_iterator += 1
            self._case2_iterator = 0
        else:
            self._iterator += 1
            self._case2_iterator += 1
            self._case1_iterator = 0

    def __reset__iterators(self):
        self._iterator = 0
        self._case2_iterator = 0
        self._case1_iterator = 0

    def __update_lambda(self):
        if self._case1_iterator == self._iterator:
            self.lambda_term = self.lambda_term / self.beta_1
        elif self._case2_iterator == self._iterator:
            self.lambda_term = self.lambda_term / self.beta_2
        return self.lambda_term


