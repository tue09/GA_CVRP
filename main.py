import copy
import math
import random
import numpy
import heapq
import time
import collections

file_path = "A-n32-k5.vrp.txt"

global data
global number_of_cities
global number_of_trucks
global truck_capacity
global distance_matrix
global city
global city_demand
def read_data(path):
    global data
    global number_of_cities
    global number_of_trucks
    global truck_capacity
    global distance_matrix
    global city
    global city_demand
    city_demand = []
    f = open(path)
    data = f.readlines()
    number_of_cities = int(data[3].split()[2])
    number_of_trucks = int(data[0][len(data[0]) - 2])
    truck_capacity = int(data[5].split()[2])
    distance_matrix = [0] * number_of_cities
    for i in range(number_of_cities):
        distance_matrix[i] = [0] * number_of_cities
    city = []
    for i in range(7, 7 + number_of_cities ):
        city.append([])
        line = data[i].split()
        for j in range(1, len(line)):
            city[i - 7].append(float(line[j]))
    for i in range(number_of_cities):
        for j in range(number_of_cities):
            distance_matrix[i][j] = distance(city[i], city[j])
    for i in range(7 + number_of_cities + 1, 7 + number_of_cities + 1 + number_of_cities):
        line = data[i].split()
        city_demand.append(float(line[1]))
    f.close()
    distance_matrix = numpy.array(distance_matrix)
def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
read_data(file_path)
def fitness_function(chromosome):
    temp = copy.copy(chromosome)
    temp.insert(0, 0)
    temp.append(0)
    sum = 0
    for i in range(len(temp) - 1):
        sum = sum + distance_matrix[temp[i]][temp[i + 1]]
    array = [0] * number_of_trucks
    index = 0
    for i in range(0, len(chromosome)):
        if chromosome[i] != 0:
            array[index] = array[index] + city_demand[chromosome[i]]
        else:
            index = index + 1
            array[index] = array[index] + city_demand[chromosome[i]]
    ratio = [0]*number_of_trucks
    for i in range(number_of_trucks):
        if array[i] <= truck_capacity: ratio[i] = 1
        else: ratio[i] = array[i] / 100
    for i in range(number_of_trucks):
        sum = sum * ratio[i]
    return sum

#individual = [[trip], fitness, [index_of_0]]
def number_of_0(chromosome):
    number = 0
    for i in chromosome:
        if i == 0:
            number = number + 1
    return number
def index_of_0(chromosome):
    set0 = []
    for i in range(len(chromosome)):
        if chromosome[i] == 0:
            set0.append(i)
    return set0
def len_tour(route):
    temp = copy.copy(route)
    temp.insert(0,0)
    temp.append(0)
    global number_of_trucks
    index = []
    array = []
    for i in range(number_of_trucks):
        array.append([])
    for i in range(len(temp)):
        if temp[i] == 0: index.append(i)
    for i in range(number_of_trucks):
        array[i] = index[i+1] - index[i] - 1
    return array
def different(chromosome1, chromosome2):
    if round(chromosome1[1], 5) != round(chromosome2[1], 5): return True
    else:
        x1 = sorted(len_tour(chromosome1[0]))
        x2 = sorted(len_tour(chromosome2[0]))
        if x1 != x2: return True
        else:
            array1 = []
            for i in range(number_of_trucks):
                array1.append([])
            index1 = 0
            for i in range(len(chromosome1[0])):
                if i == len(chromosome1[0]) - 1: break
                if chromosome1[0][i] != 0: array1[index1].append(chromosome1[0][i])
                else: index1 = index1 + 1
            array2 = []
            for i in range(number_of_trucks):
                array2.append([])
            index2 = 0
            for i in range(len(chromosome2[0])):
                if i == len(chromosome2[0]) - 1: break
                if chromosome2[0][i] != 0: array2[index2].append(chromosome2[0][i])
                else: index2 = index2 + 1
            hashed1 = [hash(tuple(sorted(sub))) for sub in array1]
            hashed2 = [hash(tuple(sorted(sub))) for sub in array2]
            x = collections.Counter(hashed1)
            y = collections.Counter(hashed2)
            if x != y: return True
            else: return False

def exist(population, individual):
    for i in population:
        if different(i, individual) == False: return True
    return False

def polar_angles():
    relative_point = []
    for i in range(number_of_cities):
        relative_point.append([i, []])
    for i in range(number_of_cities):
        relative_point[i][1] = [city[i][0] - city[0][0], city[i][1] - city[0][1]]
    polar_angles = []
    polar_angles.append([0, 0])
    for i in range(1, number_of_cities):
        if relative_point[i][1][0] == 0 and relative_point[i][1][0] == 0 > 0: polar_angles.append( [i, math.pi / 2 ] )
        elif relative_point[i][1][0] == 0 and relative_point[i][1][0] == 0 < 0: polar_angles.append([i, -math.pi / 2])
        elif relative_point[i][1][1] >= 0 and relative_point[i][1][0] > 0:
            polar_angles.append( [i, math.atan(relative_point[i][1][1] / relative_point[i][1][0])] )
        elif relative_point[i][1][1] < 0:
            polar_angles.append( [i, math.pi + math.atan(relative_point[i][1][1] / relative_point[i][1][0])] )
        else:
            polar_angles.append( [i, 2*math.pi + math.atan(relative_point[i][1][1] / relative_point[i][1][0])] )
    polar_angles = sorted(polar_angles, key = lambda x: x[1])
    return polar_angles

def nearest_city(city, route):
    if len(route) == 1: return route[0]
    else:
        min = max(distance_matrix[city][route[0]], distance_matrix[city][route[1]])
        index = 0
        for i in route:
            if distance_matrix[city][i] <= min and i != city:
                min = distance_matrix[city][i]
                index = i
    return index

def sorted_route(route):
    fake = copy.copy(route)
    array = []
    route.insert(0, 0)
    point = 0
    for i in range(len(route) - 1):
        point = nearest_city(point, fake)
        array.append(point)
        fake.remove(point)
    return array
def initialize_population(size):
    global index
    population = []
    route = []
    x = polar_angles()
    capacity = 0
    set = []
    for i in range(number_of_trucks):
        set.append([])
    index = 0
    for i in range(1, len(x)):
        if capacity + city_demand[x[i][0]] <= truck_capacity:
            set[index].append(x[i][0])
            capacity = capacity + city_demand[x[i][0]]
        else:
            if index == number_of_trucks - 1: break
            index = index + 1
            set[index].append(x[i][0])
            capacity = city_demand[x[i][0]]
    for i in range(len(set)):
        set[i] = sorted_route(set[i])
    for i in range(len(set)):
        route.extend(set[i])
        route.append(0)
    route.pop(len(route) - 1)
    first_individual = [route, fitness_function(route), index_of_0(route)]
    population.append(first_individual)
    individual = copy.copy(route)
    for i in range(size - 1):
        while exist(population, [individual, fitness_function(individual), index_of_0(individual)]) == True:
            random.shuffle(individual)
        population.append(copy.deepcopy([individual, fitness_function(individual), index_of_0(individual)]))
    return population

def feasible_solution(solution):
    global index
    array = [0]* number_of_trucks
    index = 0
    for i in range(1, len(solution[0])):
        if solution[0][i] != 0:
            array[index] = array[index] + city_demand[solution[0][i]]
        else:
            index = index + 1
            array[index] = array[index] + city_demand[solution[0][i]]
    for i in range(len(array)):
        if array[i] > truck_capacity: return False
    return True

def roulette_wheel_selection(population):
    total_fitness = 0
    for i in range(len(population)):
        total_fitness = total_fitness + population[i][1]
    probabilities = []
    for i in range(len(population)):
        probabilities.append((total_fitness - population[i][1]))
    probabilities = [p / sum(probabilities) for p in probabilities]
    r = random.random()
    cumulative_probability = 0
    for i in range(len(population)):
        cumulative_probability += probabilities[i]
        if r < cumulative_probability:
            return population[i]
def tournament_selection(population, Tournament_size):
    set1 = random.choices(population, k = Tournament_size)
    set1.sort(key = lambda x: x[1])
    parent1 = set1[0]
    set2 = random.choices(population, k = Tournament_size)
    set2.sort(key = lambda x: x[1])
    parent2 = set2[0]
    return parent1, parent2

def one_point_crossover(parent1, parent2, crossover_rate):
    random_number = random.random()
    if random_number <= crossover_rate:
        child1 = [0]*2
        child2 = [0]*2
        point = random.randint(0,len(parent1[0]))
        child1[0] = parent1[0][0:point]
        for k in parent2[0]:
            if (k in child1[0]) == False and k != 0:
                child1[0].append(k)
            elif k == 0 and number_of_0(child1[0]) != number_of_trucks - 1:
                child1[0].append(k)
        child1[1] = fitness_function(child1[0])
        child2[0] = parent2[0][0:point]
        for k in parent1[0]:
            if (k in child2[0]) == False and k != 0:
                child2[0].append(k)
            elif k == 0 and number_of_0(child2[0]) != number_of_trucks - 1:
                child2[0].append(k)
        child2[1] = fitness_function(child2[0])

        child1.append(index_of_0(child1[0]))
        child2.append(index_of_0(child2[0]))
    else:
        child1 = parent1
        child2 = parent2
    return child1, child2

def order_crossover(parent1, parent2, crossover_rate):
    global x1
    global x2
    global off_spring1
    global off_spring2
    random_number = random.random()
    if random_number <= crossover_rate:
        point1 = random.randint(0, len(parent1) - 2)
        point2 = random.randint(point1, len(parent2) - 1)
        p1 = parent1[0][point1 : point2]
        p2 = parent2[0][point1 : point2]
        off_spring1, off_spring2 = [[], 0], [[], 0]
        x1, x2 = [], []
        for i in range(point2, len(parent1[0])):
            x1.append(parent1[0][i])
        for i in range(0, point2):
            x1.append(parent1[0][i])
        for i in x1:
            if (i in p2) == True and i != 0: x1.remove(i)
        for i in range(point2, len(parent2[0])):
            x2.append(parent2[0][i])
        for i in range(0, point2):
            x2.append(parent2[0][i])
        for i in x2:
            if (i in p1) == True and i != 0: x2.remove(i)
        off_spring1[0] = x2[len(x2) - len(parent1[0]) + point2 - 1 : len(x2)] + p1 + x2[0 : len(x2) - len(parent1[0]) + point2 - 1]
        off_spring2[0] = x1[len(x1) - len(parent1[0]) + point2 - 1 : len(x1)] + p2 + x1[0 : len(x1) - len(parent1[0]) + point2 - 1]
        if number_of_0(off_spring1[0]) == number_of_trucks: off_spring1[0].remove(0)
        if number_of_0(off_spring2[0]) == number_of_trucks: off_spring2[0].remove(0)
        off_spring1 = [off_spring1[0], fitness_function(off_spring1[0]), index_of_0(off_spring1[0])]
        off_spring2 = [off_spring2[0], fitness_function(off_spring2[0]), index_of_0(off_spring2[0])]
    else:
        off_spring1 = parent1
        off_spring2 = parent2
    return off_spring1, off_spring2

def crossover(population, parent1, parent2, crossover_rate):
    child1_one, child2_one = one_point_crossover(parent1, parent2, crossover_rate)
    child1_order, child2_order = order_crossover(parent1, parent2, crossover_rate)
    array1 = [child1_one, child1_order]
    array2 = [child2_one, child2_order]
    array1 = sorted(array1, key = lambda x: x[1])
    array2 = sorted(array2, key = lambda x: x[1])
    child1 = array1[0]
    if exist(population, child1) == True:
        child1 = array1[1]
        if exist(population, child1) == True:
            child1 = parent1
    child2 = array2[0]
    if exist(population, child2) == True:
        child2 = array2[1]
        if exist(population, child2) == True:
            child2 = parent2
    return child1, child2

def swap_mutation(chromosome, mutation_rate):
    random_number = random.random()
    if random_number <= mutation_rate:
        child = copy.deepcopy(chromosome)
        point1 = random.randint(0,len(chromosome))
        point2 = random.randint(0,len(chromosome))
        child[0][point1], child[0][point2] = child[0][point2], child[0][point1]
        child[1] = fitness_function(child[0])
        child = [child[0], fitness_function(child[0]), index_of_0(child[0])]
        return child
    return chromosome

def inversion_mutation(chromosome, mutation_rate):
    random_number = random.random()
    if random_number <= mutation_rate:
        child = copy.deepcopy(chromosome)
        point1 = random.randint(0, len(child) - 2)
        point2 = random.randint(point1 + 1, len(child) - 1)
        b = child[0][point1 : point2 + 1]
        b.reverse()
        child[0][point1 : point2 + 1] = b
        child = [child[0], fitness_function(child[0]), index_of_0(child[0])]
        return child
    return chromosome

def nearest_city1(city):
    min = max(distance_matrix[city][0], distance_matrix[city][1])
    index = 0
    for i in range(number_of_cities):
        if distance_matrix[city][i] <= min and i != city:
            min = distance_matrix[city][i]
            index = i
    return index

def IRGIBNNM_mutation(chrmosome, mutation_rate):
    random_number = random.random()
    if random_number <= mutation_rate:
        child = copy.deepcopy(chrmosome)
        child = inversion_mutation(child, mutation_rate)

        point = random.randint(0, len(child))
        near = nearest_city1(point)
        array = []
        for i in range(0, number_of_cities):
            array.append(i)
        array.remove(point)
        array.remove(near)
        random_element = random.choice(array)
        child[0][point], child[0][random_element] = child[0][random_element], child[0][point]
        child = [child[0], fitness_function(child[0]), index_of_0(child[0])]
        return child
    return chrmosome

def mutation(population, child, mutation_rate):
    child_swap = swap_mutation(child, mutation_rate)
    child_inversion = inversion_mutation(child, mutation_rate)
    child_IRGIBNNM = IRGIBNNM_mutation(child, mutation_rate)
    array = [child_swap, child_inversion, child_IRGIBNNM]
    array = sorted(array, key = lambda x: x[1])
    child1 = array[0]
    if exist(population, child1) == True:
        child1 = array[1]
        if exist(population, child1) == True:
            child1 = array[2]
            if exist(population, child1) == True:
                child1 = child
    return child1

def Genetic_Algorithm(current_population, tournament_size, crossover_rate, mutation_rate, number_iteration):
    for i in range(number_iteration):
        print(i)
        feasible = []
        for element in current_population:
            if feasible_solution(element) == True: feasible.append(element)
        best = sorted(feasible, key = lambda  x: x[1])[: 2]
        new_population = []
        new_population.extend(best)
        for j in range(int(len(current_population)/2) - 1):
            #Crossover:
            if i <= number_iteration/2: parent1, parent2 = tournament_selection(current_population, tournament_size)
            else: parent1, parent2 = roulette_wheel_selection(current_population), roulette_wheel_selection(current_population)
            child1, child2 = crossover(new_population, parent1, parent2, crossover_rate)
            #Mutation:
            child1, child2 = mutation(new_population, child1, mutation_rate), mutation(new_population, child2, mutation_rate)
            new_population.append(child1)
            new_population.append(child2)
        tick = int(len(current_population) * 80/100)
        new_population1 = heapq.nsmallest(tick, list(new_population), key = lambda x: x[1])
        current_population1 = heapq.nsmallest(min(50, number_of_cities) - len(new_population1), list(current_population), key = lambda x: x[1])
        current_population = current_population1 + new_population1
        length = len(current_population)
        if length != min(50, number_of_cities):
            """current_population = current_population + new_population[:min(50, number_of_cities) - length]"""
            temp = initialize_population(min(50, number_of_cities) - length + 1)
            temp.pop(0)
            current_population = current_population1 + temp
    feasible = []
    for element in current_population:
        if feasible_solution(element) == True: feasible.append(element)
    best = min(feasible, key=lambda x: x[1])
    return best

solution = Genetic_Algorithm(initialize_population(min(50, number_of_cities)), 4, 0.95, 1/number_of_cities, 500)
print(solution)

#individual = [[trip], fitness, [index_of_0]]







