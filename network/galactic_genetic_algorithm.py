import numpy as np
import random
import operator

class GalacticGeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, galaxy_type):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.galaxy_type = galaxy_type
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {
                'distance': random.uniform(0, 100),
                'velocity': random.uniform(0, 200),
                'fitness': 0
            }
            population.append(individual)
        return population

    def calculate_fitness(self, individual):
        if self.galaxy_type == 'Spiral':
            fitness = individual['distance'] * individual['velocity']
        elif self.galaxy_type == 'Elliptical':
            fitness = individual['distance'] + individual['velocity']
        else:
            fitness = individual['distance'] - individual['velocity']
        return fitness

    def evaluate_population(self):
        for individual in self.population:
            individual['fitness'] = self.calculate_fitness(individual)

    def select_parents(self):
        parents = []
        for _ in range(self.population_size // 2):
            parent1 = random.choice(self.population)
            parent2 = random.choice(self.population)
            if parent1['fitness'] > parent2['fitness']:
                parents.append(parent1)
            else:
                parents.append(parent2)
        return parents

    def crossover(self, parent1, parent2):
        child = {
            'distance': (parent1['distance'] + parent2['distance']) / 2,
            'velocity': (parent1['velocity'] + parent2['velocity']) / 2,
            'fitness': 0
        }
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            individual['distance'] += random.uniform(-10, 10)
            individual['velocity'] += random.uniform(-20, 20)

    def next_generation(self):
        parents = self.select_parents()
        children = []
        for _ in range(self.population_size // 2):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            children.append(child)
        self.population = children + parents

    def run(self):
        for _ in range(self.generations):
            self.evaluate_population()
            self.next_generation()
        self.evaluate_population()
        return max(self.population, key=operator.itemgetter('fitness'))

def main():
    gga = GalacticGeneticAlgorithm(population_size=100, generations=100, mutation_rate=0.1, galaxy_type='Spiral')
    best_individual = gga.run()
    print("Best Individual:")
    print("Distance:", best_individual['distance'])
    print("Velocity:", best_individual['velocity'])
    print("Fitness:", best_individual['fitness'])

if __name__ == "__main__":
    main()
