import pytest
import numpy as np
from problem_definitions.nk_landscape import NKLandscapeProblem

class TestNKLandscapeProblem:
    """Tests for the NKLandscapeProblem class."""

    def test_initialization_valid(self):
        """Test successful initialization."""
        problem = NKLandscapeProblem(N=10, K=2, landscape_seed=42)
        assert problem.N == 10
        assert problem.K == 2
        assert problem.num_genes == 10

    def test_initialization_invalid_k(self):
        """Test that K >= N raises a ValueError."""
        with pytest.raises(ValueError, match="K must be less than N"):
            NKLandscapeProblem(N=5, K=5)

    def test_fitness_calculation(self):
        """Test that fitness calculation returns a float."""
        problem = NKLandscapeProblem(N=20, K=4, landscape_seed=123)
        individual = np.random.randint(2, size=20)
        fitness = problem.fitness(individual)
        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0

    def test_reproducible_landscape(self):
        """Test that the same seed produces the same fitness landscape."""
        problem1 = NKLandscapeProblem(N=10, K=3, landscape_seed=99)
        problem2 = NKLandscapeProblem(N=10, K=3, landscape_seed=99)
        individual = np.ones(10, dtype=int)
        assert problem1.fitness(individual) == problem2.fitness(individual)

    def test_fitness_empty_individual(self):
        """Test fitness for N=0 case."""
        problem = NKLandscapeProblem(N=0, K=0)
        individual = np.array([])
        assert problem.fitness(individual) == 0.0