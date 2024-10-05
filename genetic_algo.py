import numpy as np
import operator
import random
import pandas as pd
import collections

class KnapSack:
    def __init__(self, conf_path, criteria, enable_crossover, 
    enable_mutation, srate, mutation_rate, save_output,
    early_stop, tournament_size=2):

        self.conf = conf_path
        self.selection_criteria = criteria
        self.enable_crossover = enable_crossover
        self.enable_mutation = enable_mutation
        self.population = None
        self.capacity = None
        self.stop = None
        self.current_gen = None
        self.item_choices = None
        self.pop_size = None
        self.best_parents = []
        self.selection_rate = srate
        self.tournament_size = tournament_size
        self.initial_population_size = None
        self.mutation_rate = mutation_rate
        self.save_output = save_output

        self.best_generation_value = -1
        self.best_generation = []
        self.best_generation_idx = None
        self.best_gene_count = None
        
        self.fitness_tracker = collections.deque(maxlen=10)
        self.early_stop = early_stop
    

    def get_initial_population(self):

        # code from the pdf

        np.random.seed(1470)

        with open (self.conf, 'r') as file:
            lines = file.readlines()

        pop_size, n, self.stop, self.capacity = map(int, [lines[i].strip() for i in range (4)])

        self.item_choices = [tuple(map(int ,line.strip().split())) for line in lines [4:]]

        self.current_gen = 0

        self.population = np.random.choice([0, 1], size=(pop_size ,n), p=[0.9, 0.1])

        self.initial_population_size = pop_size


    def print_config(self):

        print("Capacity: ", self.capacity)
        print("Population: ", self.population)
        print("Stop: ", self.stop)


    def evaluate_fitness(self, chromosome):

        total_value = 0
        total_weight = 0

        for idx, gene in enumerate(chromosome):

            if gene:
                total_weight += self.item_choices[idx][0]
                total_value += self.item_choices[idx][1]

        if total_weight > self.capacity:
                return 0, 0
            
        return total_value, total_weight


    def evaluate_population_fitness(self):

        best_pop = int(self.initial_population_size * self.selection_rate)
        bests = []
        
        for idx, chromosome in enumerate(self.population):

            fitness, cur_capacity = self.evaluate_fitness(chromosome)
            
            if (fitness > 0):
                bests.append((fitness, chromosome))
        
        bests.sort(key=operator.itemgetter(0), reverse=True)

        self.bests_parents = bests[:best_pop]

        self.best_parents = [x[1] for x in self.bests_parents]

    
    def get_fittest_individual(self, selected_parents):
        
        bests = []

        for idx, chromosome in enumerate(selected_parents):

            fitness, cur_capacity = self.evaluate_fitness(chromosome)
            bests.append((fitness, chromosome))

        bests.sort(key=operator.itemgetter(0), reverse=True)

        return bests[0]
        

    def tournament_selection(self, min_parent_count):

        selected_parents = []

        while len(selected_parents) < min_parent_count:
            participating_parents = random.sample(self.best_parents, self.tournament_size)

            best_parent = participating_parents[0]
            best_fitness, best_capacity = self.evaluate_fitness(best_parent)

            for cur_parent in participating_parents[1:]:
                cur_fitness, cur_capacity = self.evaluate_fitness(cur_parent)
                if best_fitness < cur_fitness:
                    best_fitness = cur_fitness
                    best_parent = cur_parent
            
            selected_parents.append(best_parent)

        return selected_parents


    def roulette_selection(self, min_parent_count):
        
        total_fitness = 0

        for cur_parent in self.best_parents:
            cur_fitness, cur_capacity = self.evaluate_fitness(cur_parent)
            total_fitness += cur_fitness

        selected_parents = []

        while len(selected_parents) < min_parent_count:

            rand_fitness = random.randint(0, total_fitness)

            total = 0

            for cur_parent in self.best_parents:
                cur_fitness, cur_capacity = self.evaluate_fitness(cur_parent)
                total += cur_fitness
                if (total > rand_fitness):
                    selected_parents.append(cur_parent)
                    break

        return selected_parents

    
    def selection(self):

        min_parent_count = int(self.selection_rate * len(self.best_parents))
        selected_parents = []

        if (self.selection_criteria == 0):
            # Tournament Selection
            selected_parents = self.tournament_selection(min_parent_count)
        else:
            # Roulette Selection
            selected_parents = self.roulette_selection(min_parent_count)

        return selected_parents


    def crossover(self, c1, c2):
        
        threshold = random.randint(1, len(c1)-1)

        tmp1 = c1[threshold:]
        tmp2 = c2[threshold:]
        c1 = c1[:threshold]
        c2 = c2[:threshold]

        c1 = np.append(c1, tmp2)
        c2 = np.append(c2, tmp1)
        
        return c1, c2


    def mutation(self, child):

        random_vals = np.random.rand(len(child))

        for i, (gene, rand_val) in enumerate(zip(child, random_vals)):
            if rand_val < self.mutation_rate:
                if gene == 1:
                    child[i] = 0
                else:
                    child[i] = 1
        
        return child


    def calculate_evaluation_report(self, selected_parents):

        population_fitness = 0
        population_capacity = 0

        for idx, chromosome in enumerate(selected_parents):
            fitness, cur_capacity = self.evaluate_fitness(chromosome)

            population_fitness += fitness
            population_capacity += cur_capacity
            
        avg_fitness = population_fitness / len(selected_parents)
        avg_capacity = population_capacity / len(selected_parents)

        return (avg_fitness, avg_capacity)

            
    def run(self):

        self.get_initial_population()

        df = pd.DataFrame(columns=['Generation', 'Avg_Population_Fitness', 'Avg_Population_Capacity', 'Active_Gene_Count', 'Best_Fitness_Score'])

        offset = ""
        
        if self.selection_criteria == 0:
            offset += "_tournament"
        else:
            offset += "_roulette"

        if self.enable_crossover:
            offset += "_crossover"
        
        if self.enable_mutation:
            offset += "_mutation"
            offset += "_" + str(self.mutation_rate)

        while self.current_gen <= self.stop:

            self.evaluate_population_fitness()

            selected_parents = self.selection()

            self.current_gen += 1

            avg_fitness, avg_capacity = self.calculate_evaluation_report(selected_parents)

            fittest_individual = self.get_fittest_individual(selected_parents)

            best_fitness = fittest_individual[0]
            active_gene_count = np.sum(fittest_individual[1] == 1)

            print("Generation: {0}, Avg Fitness: {1}, Avg Capacity: {2}, Best Fitness: {3}, Best Active Gene Count: {4}".format(self.current_gen, avg_fitness, 
            avg_capacity, best_fitness, active_gene_count))

            if avg_fitness > self.best_generation_value:

                self.best_generation_value = avg_fitness
                self.best_generation = selected_parents
                self.best_generation_idx = self.current_gen

            self.fitness_tracker.append(avg_fitness)

            if ((self.early_stop) and (self.current_gen > len(self.fitness_tracker))):
                if self.fitness_tracker.count(self.fitness_tracker[0]) == len(self.fitness_tracker):
                    break    
            
            df.loc[len(df)] = [self.current_gen, avg_fitness, avg_capacity, best_fitness, active_gene_count]

            new_parents = []

            if self.enable_crossover:
                
                pop = len(selected_parents)-1

                sample = random.sample(range(pop), pop)
                
                for i in range(0, pop):
                    r1 = []
                    r2 = []

                    if i < pop-1:
                        r1 = selected_parents[i]
                        r2 = selected_parents[i+1]
                    else:
                        r1 = selected_parents[i]
                        r2 = selected_parents[0]
                    
                    new_child1, new_child2 = self.crossover(r1, r2)

                    new_parents.append(new_child1)
                    new_parents.append(new_child2)

            if self.enable_mutation:

                for i in range(len(new_parents)):
                    new_parents[i] = self.mutation(new_parents[i])

            if self.enable_crossover:
                self.population = new_parents
            elif self.enable_mutation:
                self.population = new_parents
            else:
                self.population = selected_parents

        if self.save_output:

            output_file_name = 'output/selection' + offset + "_" + str(self.current_gen) + '_generation' + "_" + self.conf.split('/')[-1].split('.')[0] + '.csv'
            df.to_csv(output_file_name, index=False)

        print("*"*105)
        print("Best Generation : {0}, Fitness : {1}".format(self.best_generation_idx, 
        self.best_generation_value))
        print("*"*105)

        





	




                    


        
