"""
Script to generate a Moran process.
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class MoranProcess():
    """ 
    This class allows us to build a Moran process.

    A **neutral drift Moran process** is a discrete-time stochastic process
    typically used to model populations. 
    
    At each step this process will increase by one, decrease by one, or remain 
    at the same value between 0 and the number of states (n). 
    The process ends when its value reaches zero or a set maximum value.

    :param max: maximum possible value for the process.
    :type max: int

    :param seed: seed for the random number generator.
    :type seed: int

    :param simulation_type: type of simulation to be performed. Either 
        simulation_type = 'selection' or simulation_type = 'neutral_drift'.
    :type simulation_type: str
    """

    def __init__(self, max: int, seed: int, simulation_type: str):
        super().__init__()
        self.max = max
        if simulation_type == 'selection':
            self.f_a, self.f_b = self.group_fitness(max)
        else: 
            self.f_a = None
            self.f_b = None
        self.probs = self.transition_probabilities(max)
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

    def group_fitness(self, n: int) -> Tuple[np.ndarray, np.ndarray]: 
        """
        This function generates random fitness values for each of the population
        groups.

        :param n: length of the array.
        :type n: int

        :return: fitness values for each group.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        f_a: np.ndarray = np.random.randint(low=1, high=10, size=n)
        f_b: np.ndarray = np.random.randint(low=1, high=10, size=n)
        return f_a, f_b
        

    def transition_probabilities(self, n: int) -> List[float]:
        """
        This function generates transition probabilities for state n, the 
        current state.

        :param n: current state.
        :type n: int 

        :return: list of transition probabilities.
        :rtype: List[float]
        """
        probabilities = []

        if np.any(self.f_a) and np.any(self.f_b): # simulation_type == 'selection'
            for k in range(1, n): 
                p_down = self.f_b[k]*(n-k)/(self.f_a[k]*k + self.f_b[k]*(n-k)) * k/n
                p_up = self.f_a[k]*k/(self.f_a[k]*k + self.f_b[k]*(n-k)) * (n-k)/n
                p_steady = 1 - (p_down + p_up)
                probabilities.append([p_down, p_steady, p_up])

        else: # simulation_type == 'neutral_drift'
            for k in range(1, n): 
                p_down = (n-k)/n * k/n
                p_up = k/n * (n-k)/n
                p_steady = 1 - (p_down + p_up)
                probabilities.append([p_down, p_steady, p_up])            

        return probabilities

    def simulate_moran_process(self, n, initial_state) -> np.ndarray:
        """
        This function generates a realization of the Moran process. The 
        generation process continues until absorption occurs (state 0 or maximum)
        or the length of the process reaches length n.

        :param n: length of the process.
        :type n: int

        :param initial_state: initial state of the process.
        :type initial_state: int

        :return: a realization of the Moran process.
        :rtype: np.array
        """
        s = [initial_state]
        increments = [-1, 0, 1]
        for k in range(n-1):
            if initial_state in [0, self.max]: 
                break
            initial_state += self.rng.choice(increments, p=self.probs[initial_state - 1])
            s.append(initial_state)

        return np.array(s)
    

if __name__ == '__main__':

    # valor maxim: 100; 100 individul total; a l'inici 20 de tipus A
    moran_process = MoranProcess(max=100, seed=123, simulation_type='neutral_drift')
    neutral_drift_array = moran_process.simulate_moran_process(n=100, initial_state=20)
    print(f'Moran process with neutral drift simulation:\n {neutral_drift_array}\n')
    # plt.plot(moran_process.simulate_moran_process(n=100, initial_state=20))

    moran_process = MoranProcess(max=100, seed=123, simulation_type='selection')
    selection_array = moran_process.simulate_moran_process(n=100, initial_state=20)
    print(f'Moran process with selection simulation:\n {selection_array}\n')
    # plt.plot(moran_process.simulate_moran_process(n=100, initial_state=20))


    # TODO: tuning de max, n i initial_state
    # TODO: fer grafica