"""
Moran process simulation.
"""

import matplotlib.pyplot as plt
import numpy as np

from moran_process import MoranProcess


INITIAL_STATE = 5
POPULATION_SIZE = 10


def get_other_group_population_counts(
    a_array: np.ndarray, population_size: int
) -> np.ndarray:
    """
    Given the array showing the population counts of group A and the total
    population size, this function computes the population counts of group B.

    :param a_array: array showing the population counts of group A
    :type a_array: np.ndarray

    :param population_size: total population size
    :type population_size: int

    :return: array showing the population counts of group B
    :rtype: np.ndarray
    """
    b_array = np.array([population_size for i in range(len(a_array))])
    for index in range(len(b_array)): 
        b_array[index] = b_array[index] - a_array[index]
    return b_array


def fixation_probability(num_repetitions: int, initial_state: int) -> float:
    """
    Computes the probability of fixation of a mutation in a population.

    :param num_repetitions: number of repetitions of the simulation
    :type num_repetitions: int

    :param initial_state: initial population count of group A
    :type initial_state: int

    :return: probability of fixation
    :rtype: float
    """
    fixation_count = 0

    for rep in range(num_repetitions):
        moran_process = MoranProcess(
            population_size=POPULATION_SIZE, 
            seed=rep, 
            initial_state=initial_state,
        )
        neutral_drift_array_a = moran_process.simulate_moran_process()

        if neutral_drift_array_a[-1] == POPULATION_SIZE:
            fixation_count += 1

    return fixation_count/num_repetitions


if __name__ == '__main__':
    moran_process = MoranProcess(
        population_size=POPULATION_SIZE, 
        seed=123, 
        initial_state=INITIAL_STATE,
    )
    neutral_drift_array_a = moran_process.simulate_moran_process()
    neutral_drift_array_b = get_other_group_population_counts(
        a_array=neutral_drift_array_a, 
        population_size=POPULATION_SIZE
    )

    plt.plot(np.column_stack((neutral_drift_array_a, neutral_drift_array_b)))
    plt.legend(loc='upper left', labels=('A', 'B'))
    plt.show()

    probabilities = [
        fixation_probability(num_repetitions=500, initial_state=i) for i in range(1, POPULATION_SIZE)
    ]
    plt.scatter(range(1, POPULATION_SIZE), probabilities, label='Experimental probability')
    plt.plot(
        range(1, POPULATION_SIZE), [i/POPULATION_SIZE for i in range(1, POPULATION_SIZE)], 
        label='Theoretical probability', linestyle='dashed'
    )
    plt.xlabel("$i$")
    plt.ylabel("$i/N$")
    plt.legend(loc='upper left')
    plt.show()
