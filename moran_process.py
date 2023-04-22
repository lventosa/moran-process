"""
Moran process class definition.
"""

from typing import List, Tuple

import numpy as np


class MoranProcess():
    """ 
    This class allows us to build a neutral-drift Moran process, which is a 
    discrete-time stochastic process typically used to model evolutionary 
    dynamics in populations. 
    
    At each step this process will increase by one, decrease by one, or remain 
    at the same value between 0 and the number of states, i.e. individuals (n). 
    The process ends when its value reaches zero or n.

    :param population_size: number of individuals in the population.
    :type population_size: int

    :param initial_state: initial state of the process.
    :type initial_state: int

    :param seed: seed for the random number generator.
    :type seed: int
    """

    def __init__(
        self, population_size: int, initial_state: float, seed: int, 
    ):
        super().__init__()
        self.population_size = population_size
        self.initial_state = initial_state

        self.probs = self.transition_probabilities()

        self.rng = np.random 
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """
        This function sets the seed for the random number generator.

        :param seed: seed for the random number generator.
        :type seed: int
        """
        self.rng.seed(seed)        

    def transition_probabilities(self) -> List[float]:
        """
        This function generates transition probabilities for state n, the 
        current state.

        :return: list of transition probabilities.
        :rtype: List[float]
        """
        probabilities = []
        n = self.population_size # to simplify notation in the formulas below

        for i in range(1, n): 
            p_down = (n-i)/n * i/n
            p_up = i/n * (n-i)/n
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
    