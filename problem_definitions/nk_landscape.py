# feda_project/problem_definitions/nk_landscape.py
"""Defines the NK-Landscape optimization problem."""

import numpy as np
from..utils.debugging import print_debug # Assuming utils is a sibling directory

class NKLandscapeProblem:
    """
    Represents an NK-Landscape problem instance.

    The NK-Landscape is a tunable rugged fitness landscape model used in
    evolutionary computation and complex systems research. 'N' is the number
    of genes (or items), and 'K' is the number of other genes that influence
    the fitness contribution of each gene.

    Attributes:
        num_items (int): The number of items/genes (N).
        num_genes (int): Alias for num_items (N).
        N (int): The number of genes.
        K (int): The number of epistatic interactions for each gene.
    """
    def __init__(self, N: int, K: int, landscape_seed: int = None):
        """Initializes the NKLandscapeProblem.

        Args:
            N (int): The number of genes in the landscape.
            K (int): The number of other genes that influence each gene's
                     fitness contribution. Must be less than N if N > 0.
            landscape_seed (int, optional): Seed for the random number generator
                                           to ensure reproducible landscapes.
                                           Defaults to None.

        Raises:
            ValueError: If K >= N (and N > 0), or if N or K are negative.
        """
        if K >= N and N > 0:
            raise ValueError("K must be less than N for this NK Landscape implementation if N > 0.")
        if N < 0 or K < 0:
            raise ValueError("N and K must be non-negative.")

        self.num_items = N
        self.num_genes = N  # Alias for consistency with optimizer
        self.N = N
        self.K = K
        self._interaction_tables = {}
        self._influencers = {}
        rng_landscape = np.random.RandomState(landscape_seed)

        if N > 0:
            # Define influencers for each gene (circularly for simplicity)
            for i in range(N):
                self._influencers[i] = [(i + j + 1) % N for j in range(K)]

            # Create random interaction tables for each gene
            for i in range(N):
                # Each gene i and its K influencers form K+1 bits
                # There are 2^(K+1) possible contexts for these K+1 bits
                num_contexts = 2**(self.K + 1)
                self._interaction_tables[i] = rng_landscape.rand(num_contexts)

    def _get_context_index(self, gene_i_value: int, influencer_values: list[int]) -> int:
        """Calculates the index into the interaction table for a gene.

        The index is determined by the binary pattern formed by the gene's own
        value and the values of its influencing genes.

        Args:
            gene_i_value (int): The binary value (0 or 1) of the gene itself.
            influencer_values (list[int]): A list of binary values (0 or 1) of
                                           the influencing genes.

        Returns:
            int: The calculated index for the interaction table.
        """
        context_bits = [int(gene_i_value)] + [int(val) for val in influencer_values]
        index = 0
        for bit_idx, bit_val in enumerate(context_bits):
            index += bit_val * (2**(len(context_bits) - 1 - bit_idx))
        return index

    def fitness(self, individual: np.ndarray) -> float:
        """Calculates the fitness of an individual solution.

        The fitness is the average of the contributions of each gene, where
        each gene's contribution is determined by its value and the values
        of its K influencers, looked up in pre-generated interaction tables.

        Args:
            individual (np.ndarray): A 1D NumPy array of binary values (0s and 1s)
                                     representing the solution. Its length must
                                     equal self.N.

        Returns:
            float: The calculated fitness of the individual. Returns 0.0 if N is 0
                   and the individual is empty. Returns -np.inf for invalid
                   individuals.
        """
        if self.N == 0:
            return 0.0 if individual.size == 0 else -np.inf

        if not isinstance(individual, np.ndarray) or individual.ndim!= 1 or individual.size!= self.N:
            print_debug(f"NKLandscapeProblem Warning: Invalid individual. Shape: {individual.shape if isinstance(individual, np.ndarray) else 'N/A'}, Expected: ({self.N},)")
            return -np.inf # Invalid individual

        total_fitness_contribution = 0.0
        for i in range(self.N):
            gene_i_value = individual[i]
            influencer_indices = self._influencers[i]
            influencer_values = [individual[j] for j in influencer_indices]
            context_idx = self._get_context_index(gene_i_value, influencer_values)
            total_fitness_contribution += self._interaction_tables[i][context_idx]

        # Average fitness contribution
        return total_fitness_contribution / self.N if self.N > 0 else 0.0

    def get_optimal_fitness(self) -> float:
        """Returns the theoretical maximum possible fitness for an NK-Landscape.

        In this implementation, since interaction table values are drawn from
        Unif(0,1), the maximum contribution per gene is 1.0.

        Returns:
            float: The theoretical maximum fitness (1.0).
        """
        return 1.0 # Max contribution per gene is 1, so average max is 1.

    def __repr__(self) -> str:
        """Provides a string representation of the NKLandscapeProblem instance.

        Returns:
            str: A string representation (e.g., "NKLandscapeProblem(N=20, K=3)").
        """
        return f"NKLandscapeProblem(N={self.N}, K={self.K})"