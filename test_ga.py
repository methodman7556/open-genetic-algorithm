import numpy
import pytest

from ga import GeneticAlgorithm


@pytest.fixture
def prng():
    yield numpy.random.default_rng(12345)


def test_ga(prng):
    def fitness_func(population):
        equation_inputs = [4, -2, 3.5, 5, -11, -4.7]
        # Calculating the fitness value of each solution in the current population.
        # The fitness function calculates the sum of products between each input and its corresponding weight.
        return numpy.sum(population * equation_inputs, axis=1)

    ga = GeneticAlgorithm(
        chromosomes=8,
        fitness_func=fitness_func,
        generations=5,
        genes=6,
        num_parents_mating=4,
        population=None,
        prng=prng,
    )
    ga.run()

    assert ga.report() == 7
