"""
Program to generate a Moran process.
"""

import numpy as np


class MoranProcess():
    """ 
    This class allows us to buil a Moran process.

    A **neutral drift Moran process** is a discrete-time stochastic process
    typically used to model populations. 
    
    At each step this process will increase by one, decrease by one, or remain 
    at the same value between the value of 0 and the number of states (n). 
    The process ends when its value reaches zero or the maximum valued state.

    :param maximum: the maximum possible value for the process.
    :type maximum: int
    """

    def __init__(self, maximum):
        super().__init__()
        self.maximum = maximum
        self.p = self._probabilities(maximum)
        self.rng = np.random # random number generator

    def _probabilities(self, n):
        """ # TODO: review docstring
        Generate the transition probabilities for state :math:`n`.
        :param int n: the current state for which to generate transition
            probabilities.
        """
        probabilities = []
        for k in range(1, n):
            p_down = (n-k)/n * k/n
            p_up = k/n * (n-k)/n
            p_same = 1 - (p_down + p_up)
            probabilities.append([p_down, p_same, p_up])

        return probabilities

    def _sample_moran_process(self, n, start):
        """ # TODO: review docstring
        Generate a realization of the Moran process.
        Generate a Moran process until absorption occurs (state 0 or n) or
        length of process reaches length :math:`maximum`.
        """
        if not isinstance(start, int):
            raise TypeError(f'Initial state must be a positive integer.')
        if start < 0 or start > self.maximum:
            raise ValueError(f'Initial state must be between 0 and {self.maximum}')

        if not isinstance(n, int):
            raise TypeError(f'Sample length must be positive integer.')
        if n < 1:
            raise ValueError(f'Sample length must be at least 1.')

        print(f'Moran process with {self.maximum} states' )
        s = [start]
        increments = [-1, 0, 1]
        for k in range(n-1):
            if start in [0, self.maximum]:
                break
            start += self.rng.choice(increments, p=self.p[start - 1])
            s.append(start)

        return np.array(s)

    def sample(self, n, start):
        """
        # TODO: review docstring
        Generate a realization of the Moran process.
        Generate a Moran process until absorption occurs (state 0
        or :py:attr:`maximum`) or length of process reaches length :math:`n`.
        :param int n: the maximum number of steps to generate assuming
            absorption does not occur.
        :param int start: the initial state of the process.
        """
        return self._sample_moran_process(n, start)
    

if __name__ == '__main__':
    moran_process = MoranProcess(maximum=100)
    moran_process.sample(10000, 2)

    # TODO: experimentar amb diferents valors de maximum, n i start (mirar teoria)
    # TODO: print array i fer grafica