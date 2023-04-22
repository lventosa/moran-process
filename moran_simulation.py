"""
Moran process simulation.
"""

import matplotlib.pyplot as plt
import numpy as np

from moran_process import MoranProcess


INITIAL_STATE = 5
POPULATION_SIZE = 10
SEED = 123


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


if __name__ == '__main__':
    moran_process = MoranProcess(
        population_size=POPULATION_SIZE, 
        seed=SEED, 
        initial_state=INITIAL_STATE,
        simulation_type='neutral_drift'
    )
    neutral_drift_array_a = moran_process.simulate_moran_process()
    neutral_drift_array_b = get_other_group_population_counts(
        a_array=neutral_drift_array_a, 
        population_size=POPULATION_SIZE
    )
    print(f'Moran process with neutral drift simulation (group A):\n {neutral_drift_array_a}\n')
    plt.plot(np.column_stack((neutral_drift_array_a, neutral_drift_array_b)))
    plt.legend(loc='upper left', labels=('A', 'B'))
    plt.show()

    moran_process = MoranProcess(
        population_size=POPULATION_SIZE,
        seed=SEED, 
        initial_state=INITIAL_STATE,
        simulation_type='selection'
    )
    selection_array_a = moran_process.simulate_moran_process()
    selection_array_b = get_other_group_population_counts(
        a_array=selection_array_a, 
        population_size=POPULATION_SIZE
    )
    print(f'Moran process with selection simulation (group A):\n {selection_array_a}\n')
    plt.plot(np.column_stack((neutral_drift_array_a, neutral_drift_array_b)))
    plt.legend(loc='upper left', labels=('A', 'B'))
    plt.show()

    # TODO: separar fitness aleatòria i fitness constant i determinista
