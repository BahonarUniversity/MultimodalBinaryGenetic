from abc import ABC, abstractmethod

import numpy as np


class Constraint(ABC):

    @abstractmethod
    def is_feasible(self, chromosome: []):
        pass

    @abstractmethod
    def calculate_distance(self, chromosome: []):
        pass


class ReverseRadiusConstraint(Constraint):
    def __init__(self, radius: float = 5):
        self.radius = radius

    def is_feasible(self, chromosome: []):
        return np.linalg.norm(chromosome) > self.radius

    def calculate_distance(self, chromosome: []):
        if self.is_feasible(chromosome):
            return 0
        else:
            return np.linalg.norm(chromosome) - self.radius
