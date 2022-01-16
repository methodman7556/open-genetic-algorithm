import numpy

from ga import GeneticAlgorithm

"""
The y=target is to maximize this equation ASAP:
    y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)
    What are the best values for the 6 weights w1 to w6?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""

# Inputs of the equation.
equation_inputs = [4, -2, 3.5, 5, -11, -4.7]


def fitness_func(population):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calculates the sum of products between each input and its corresponding weight.
    return numpy.sum(population * equation_inputs, axis=1)


# Number of chromosomes in the population
chromosomes = 8
# Number of genes in each chromosome
genes = 6
# Number of parents to select for mating
mating_parents = 4

# PCG64 https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html
prng = numpy.random.default_rng(seed=12345)

ga = GeneticAlgorithm(
    fitness_func=fitness_func,
    generations=5,
    num_parents_mating=mating_parents,
    prng=prng,
    chromosomes=chromosomes,
    genes=genes,
)

ga.run()
ga.report()
