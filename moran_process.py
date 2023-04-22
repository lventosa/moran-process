"""
Moran process class definition.
"""

from typing import List, Tuple

import numpy as np


class MoranProcess():
    """ 
    This class allows us to build a Moran process, which is a discrete-time 
    stochastic process typically used to model evolutionary dynamics in 
    populations. 
    
    At each step this process will increase by one, decrease by one, or remain 
    at the same value between 0 and the number of states, i.e. individuals (n). 
    The process ends when its value reaches zero or n.

    :param population_size: number of individuals in the population.
    :type population_size: int

    :param initial_state: initial state of the process.
    :type initial_state: int

    :param seed: seed for the random number generator.
    :type seed: int

    :param simulation_type: type of simulation to be performed. Either 
        simulation_type = 'selection' or simulation_type = 'neutral_drift'.
    :type simulation_type: str
    """

    def __init__(
        self, population_size: int, seed: int, 
        initial_state: float, simulation_type: str
    ):
        super().__init__()
        self.population_size = population_size
        self.initial_state = initial_state

        if simulation_type == 'selection':
            self.f_a, self.f_b = self.group_fitness()
        else: 
            self.f_a = None
            self.f_b = None
        self.probs = self.transition_probabilities()

        self.rng = np.random 
        self.set_seed(self.rng, seed)

    def set_seed(self, rng: np.random, seed: int) -> None:
        """
        This function sets the seed for the random number generator.

        :param rng: random number generator.
        :type rng: np.random

        :param seed: seed for the random number generator.
        :type seed: int
        """
        rng.seed(seed)

    def group_fitness(self) -> Tuple[np.ndarray, np.ndarray]: 
        """
        This function generates random fitness values for each of the population
        groups.

        :return: fitness values for each group.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        f_a: np.ndarray = np.random.randint(low=1, high=10, size=self.population_size)
        f_b: np.ndarray = np.random.randint(low=1, high=10, size=self.population_size)
        return f_a, f_b
        

    def transition_probabilities(self) -> List[float]:
        """
        This function generates transition probabilities for state n, the 
        current state.

        :return: list of transition probabilities.
        :rtype: List[float]
        """
        probabilities = []
        n = self.population_size # to simplify notation in the formulas below

        if np.any(self.f_a) and np.any(self.f_b): # simulation_type = 'selection'
            for k in range(1, n): 
                p_down = self.f_b[k]*(-k)/(self.f_a[k]*k + self.f_b[k]*(n-k)) * k/n
                p_up = self.f_a[k]*k/(self.f_a[k]*k + self.f_b[k]*(n-k)) * (n-k)/n
                p_steady = 1 - (p_down + p_up)
                probabilities.append([p_down, p_steady, p_up])

        else: # simulation_type = 'neutral_drift'
            for k in range(1, n): 
                p_down = (n-k)/n * k/n
                p_up = k/n * (n-k)/n
                p_steady = 1 - (p_down + p_up)
                probabilities.append([p_down, p_steady, p_up])            

        return probabilities

    def simulate_moran_process(self) -> np.ndarray:
        """
        This function generates a realization of the Moran process. The 
        generation process continues until absorption occurs.

        :return: a realization of the Moran process (group A counts).
        :rtype: np.array
        """
        a_counts = [self.initial_state]
        new_state = self.initial_state
        increments = [-1, 0, 1]
        
        while True:
            if new_state in [0, self.population_size]: 
                break
            new_state += self.rng.choice(
                increments, p=self.probs[new_state-1]
            )
            a_counts.append(new_state)

        return np.array(a_counts)
    