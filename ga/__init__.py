import numpy
from numpy import ndarray


class GeneticAlgorithm:
    population: ndarray
    """Set of chromosomes"""

    def __init__(
        self,
        fitness_func,
        generations,
        genes,
        num_parents_mating,
        chromosomes,
        population=None,
        prng=numpy.random.default_rng(),
    ):
        self.chromosomes = chromosomes
        self.genes = genes
        self.fitness_func = fitness_func
        self.generations = generations
        self.num_parents = num_parents_mating
        self.pop_size = (self.chromosomes, self.genes)
        self.prng = prng
        self.population = (
            self.initialise_population() if population is None else population
        )

    def initialise_population(self) -> ndarray:
        """Initialises population for usage by the genetic algorithm.

        The population has shape of (chromosomes, genes). Chromosomes represent the individuals making up the population.
        Each chromosome's gene is populated by drawing a sample from a uniform distribution.

        """
        return self.prng.uniform(low=-4.0, high=4.0, size=self.pop_size)

    def selection(self, fitnesses):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next
        # generation.
        parents = numpy.empty((self.num_parents, self.genes))
        for parent_num in range(self.num_parents):
            max_fitness_idx = numpy.where(fitnesses == numpy.max(fitnesses))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = self.population[max_fitness_idx, :]
            fitnesses[max_fitness_idx] = -99999999999
        return parents

    def crossover(self, parents, offspring_size):
        offspring = numpy.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually it is at the center.
        crossover_point = numpy.uint8(offspring_size[1] / 2)

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k + 1) % parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def mutation(self, offspring_crossover):
        # Mutation changes a single gene in each offspring randomly.
        for idx in range(offspring_crossover.shape[0]):
            # The random value to be added to the gene.
            random_value = self.prng.uniform(low=-1.0, high=1.0, size=1)
            offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
        return offspring_crossover

    def run(self):
        for generation in range(self.generations):
            print("Generation : ", generation)
            # Measuring the fitness of each chromosome in the population.
            chromosomes_fitness = self.fitness_func(self.population)

            # Selecting the best parents in the population for mating.
            parents = self.selection(chromosomes_fitness)

            # Generating next generation using crossover.
            offspring_crossover = self.crossover(
                parents,
                offspring_size=(self.chromosomes - parents.shape[0], self.genes),
            )

            # Adding some variations to the offspring using mutation.
            offspring_mutation = self.mutation(offspring_crossover)

            # Creating the new population based on the parents and offspring.
            self.population[0 : parents.shape[0], :] = parents
            self.population[parents.shape[0] :, :] = offspring_mutation

            # The best result in the current iteration.
            print(
                "Best result : ",
                numpy.max(self.fitness_func(self.population)),
            )

    def report(self):
        # Getting the best solution after iterating finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        fitness = self.fitness_func(self.population)
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = numpy.where(fitness == numpy.max(fitness))

        print("Best solution : ", self.population[best_match_idx, :])
        print("Best solution fitness : ", fitness[best_match_idx])

        return best_match_idx[0][0]
