import copy
import math
import random
import numpy
import heapq
import time
import itertools
import collections
import TABU

global file_path
global data
global number_of_cities
global number_of_trucks
global truck_capacity
global distance_matrix
global city
global city_demand
file_path = "A-n32-k5.vrp.txt"
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
    if path == "A-n80-k10.vrp.txt":
        number_of_trucks = 10
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
    return sum
def penalty(solution, amount_of_pen, number_of_appearances):
    array = [0] * number_of_trucks
    index = 0
    for i in range(len(solution[0])):
        if solution[0][i] != 0: array[index] = array[index] + city_demand[solution[0][i]]
        else:
            index = index + 1
    sum = 0
    for i in range(len(array)):
        if array[i] > truck_capacity: sum = sum + pow(array[i] - truck_capacity, 1 )
    eval = pow(amount_of_pen, 1 / 2) * sum
    return eval * number_of_appearances
def feasible_solution(solution):
    array = [0]* number_of_trucks
    index = 0
    for i in range(0, len(solution[0])):
        if solution[0][i] != 0:
            array[index] = array[index] + city_demand[solution[0][i]]
        else:
            index = index + 1
            array[index] = array[index] + city_demand[solution[0][i]]
    for i in range(len(array)):
        if array[i] > truck_capacity: return False
    return True
#individual = [[trip], fitness, number_of_appearances]
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
    x1 = sorted(len_tour(chromosome1[0]))
    x2 = sorted(len_tour(chromosome2[0]))
    if x1 != x2:
        return True
    else:
        array1 = []
        for i in range(number_of_trucks):
            array1.append([])
        index1 = 0
        for i in range(len(chromosome1[0])):
            if chromosome1[0][i] != 0: array1[index1].append(chromosome1[0][i])
            else: index1 = index1 + 1
        array2 = []
        for i in range(number_of_trucks):
            array2.append([])
        index2 = 0
        for i in range(len(chromosome2[0])):
            if chromosome2[0][i] != 0: array2[index2].append(chromosome2[0][i])
            else: index2 = index2 + 1
        array1 = [sorted(element) for element in array1]
        array2 = [sorted(element) for element in array2]
        hashed1 = [hash(tuple(sorted(sub))) for sub in array1]
        hashed2 = [hash(tuple(sorted(sub))) for sub in array2]
        x = collections.Counter(hashed1)
        y = collections.Counter(hashed2)
        if x != y:
            return True
        else:
            return False
def exist(population, individual):
    for element in population:
        if different(element, individual) == False:
            return True, element
    return False, individual
def polar_angles():
    relative_point = []
    for i in range(number_of_cities):
        relative_point.append([i, []])
    for i in range(number_of_cities):
        relative_point[i][1] = [city[i][0] - city[0][0], city[i][1] - city[0][1]]
    polar_angles = []
    polar_angles.append([0, 0])
    for i in range(1, number_of_cities):
        if relative_point[i][1][0] == 0 and relative_point[i][1][1] > 0: polar_angles.append( [i, math.pi / 2 ] )
        elif relative_point[i][1][0] == 0 and relative_point[i][1][1] < 0: polar_angles.append([i, -math.pi / 2])
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
def initialize_population(size, amount_of_pen):
    population = []
    route = []
    x = polar_angles()
    x.remove([0, 0])
    set = []
    for i in range(number_of_trucks):
        set.append([])
    index = 0
    capacity = 0
    for i in range(0, len(x)):
        if capacity + city_demand[x[i][0]] <= truck_capacity:
            set[index].append(x[i][0])
            capacity = capacity + city_demand[x[i][0]]
        else:
            if index < number_of_trucks - 1: index = index + 1
            set[index].append(x[i][0])
            capacity = city_demand[x[i][0]]
    for i in range(len(set)):
        set[i] = sorted_route(set[i])
    for i in range(len(set)):
        route.extend(set[i])
        route.append(0)
    route.pop(len(route) - 1)
    first_individual = [route, fitness_function(route), 1]
    population.append(first_individual)
    individual = copy.copy(route)
    for i in range(size - 1):
        while exist(population, [individual, fitness_function(individual), 1])[0] == True:
            random.shuffle(individual)
        population.append(copy.deepcopy([individual, fitness_function(individual) + penalty([individual, 0], amount_of_pen, 1), 1]))
    return population
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

def one_point_crossover(parent1, parent2, crossover_rate, amount_of_pen):
    random_number = random.random()
    if random_number <= crossover_rate:
        child1 = [[], 0, 1]
        child2 = [[], 0, 1]
        point = random.randint(0,len(parent1[0]))
        child1[0] = parent1[0][0:point]
        for k in parent2[0]:
            if (k in child1[0]) == False and k != 0:
                child1[0].append(k)
            elif k == 0 and number_of_0(child1[0]) != number_of_trucks - 1:
                child1[0].append(k)
        child1[1] = fitness_function(child1[0]) + penalty(child1, amount_of_pen, 1)
        child2[0] = parent2[0][0:point]
        for k in parent1[0]:
            if (k in child2[0]) == False and k != 0:
                child2[0].append(k)
            elif k == 0 and number_of_0(child2[0]) != number_of_trucks - 1:
                child2[0].append(k)
        child2[1] = fitness_function(child2[0]) + penalty(child2, amount_of_pen, 1)
    else:
        child1 = parent1
        child2 = parent2
    return child1, child2
def two_point_crossover(parent1, parent2, crossover_rate, amount_of_pen):
    random_number = random.random()
    if random_number <= crossover_rate:
        child1 = [[-1] * len(parent1[0]), 0, 1]
        child2 = [[-1] * len(parent2[0]), 0, 1]
        point1 = random.randint(0, len(parent1[0]) - 2)
        point2 = random.randint(point1, len(parent1[0]) - 1)
        child1[0][:point1] = parent1[0][:point1]
        child1[0][point2:] = parent1[0][point2:]
        fake1 = []
        for k in range(len(parent2[0])):
            if (parent2[0][k] in child1[0]) == False and parent2[0][k] != 0:
                fake1.append(parent2[0][k])
            elif parent2[0][k] == 0 and number_of_0(fake1) + number_of_0(child1[0]) != number_of_trucks - 1:
                fake1.append(0)
        child1[0][point1:point2] = fake1
        child1[1] = fitness_function(child1[0]) + penalty(child1, amount_of_pen, 1)
        child2[0][:point1] = parent2[0][:point1]
        child2[0][point2:] = parent2[0][point2:]
        fake2 = []
        for k in range(len(parent1[0])):
            if (parent1[0][k] in child2[0]) == False and parent1[0][k] != 0:
                fake2.append(parent1[0][k])
            elif parent1[0][k] == 0 and number_of_0(fake2) + number_of_0(child2[0]) != number_of_trucks - 1:
                fake2.append(0)
        child2[0][point1:point2] = fake2
        child2[1] = fitness_function(child2[0]) + penalty(child2, amount_of_pen, 1)
    else:
        child1 = parent1
        child2 = parent2
    return child1, child2

def crossover(population, parent1, parent2, crossover_rate, change0_rate, amount_of_pen):
    child1_one, child2_one = one_point_crossover(parent1, parent2, crossover_rate, amount_of_pen)
    child1_one, child2_one = change_0_position(child1_one, change0_rate, amount_of_pen), change_0_position(child2_one, change0_rate, amount_of_pen)
    child1_two, child2_two = two_point_crossover(parent1, parent2, crossover_rate, amount_of_pen)
    child1_two, child2_two = change_0_position(child1_two, change0_rate, amount_of_pen), change_0_position(child2_two, change0_rate, amount_of_pen)
    temp = exist(population, child1_one)
    if temp[0] == True:
        child1_one[2] = temp[1][2]
        child1_one[1] = fitness_function(child1_one[0]) + penalty(child1_one, amount_of_pen, child1_one[2])
    temp = exist(population, child1_two)
    if temp[0] == True:
        child1_two[2] = temp[1][2]
        child1_two[1] = fitness_function(child1_two[0]) + penalty(child1_two, amount_of_pen, child1_two[2])

    temp = exist(population, child2_one)
    if temp[0] == True:
        child2_one[2] = temp[1][2]
        child2_one[1] = fitness_function(child2_one[0]) + penalty(child2_one, amount_of_pen, child2_one[2])
    temp = exist(population, child2_two)
    if temp[0] == True:
        child2_two[2] = temp[1][2]
        child2_two[1] = fitness_function(child2_two[0]) + penalty(child2_two, amount_of_pen, child2_two[2])

    array1 = [child1_one, child1_two]
    array2 = [child2_one, child2_two]
    array1 = sorted(array1, key=lambda x: x[1])
    array2 = sorted(array2, key=lambda x: x[1])
    child1 = array1[0]
    child2 = array2[0]
    if exist(population, child1)[0] == True:
        child1 = array1[1]
        if exist(population, child1)[0] == True:
            child1 = array1[0]
    if exist(population, child2)[0] == True:
        child2 = array2[1]
        if exist(population, child2)[0] == True:
            child2 = array2[0]
    for element in population:
        if different(element, child1) == False:
            element = copy.deepcopy(child1)
    for element in population:
        if different(element, child2) == False:
            element = copy.deepcopy(child2)
    return child1, child2

def swap_mutation(chromosome, mutation_rate, amount_of_pen):
    random_number = random.random()
    if random_number <= mutation_rate:
        child = copy.deepcopy(chromosome)
        point1 = random.randint(0,len(chromosome))
        point2 = random.randint(0,len(chromosome))
        child[0][point1], child[0][point2] = child[0][point2], child[0][point1]
        child = [child[0], fitness_function(child[0]) + penalty(child, amount_of_pen, 1), 1]
        return child
    return chromosome

def inversion_mutation(chromosome, mutation_rate, amount_of_pen):
    random_number = random.random()
    if random_number <= mutation_rate:
        child = copy.deepcopy(chromosome)
        point1 = random.randint(0, len(child) - 2)
        point2 = random.randint(point1 + 1, len(child) - 1)
        b = child[0][point1 : point2 + 1]
        b.reverse()
        child[0][point1 : point2 + 1] = b
        child = [child[0], fitness_function(child[0]) + penalty(child, amount_of_pen, 1), 1]
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
def IRGIBNNM_mutation(chrmosome, mutation_rate, amount_of_pen):
    random_number = random.random()
    if random_number <= mutation_rate:
        child = copy.deepcopy(chrmosome)
        child = inversion_mutation(child, mutation_rate, amount_of_pen)

        point = random.randint(0, len(child))
        near = nearest_city1(point)
        array = []
        for i in range(0, number_of_cities):
            array.append(i)
        array.remove(point)
        array.remove(near)
        random_element = random.choice(array)
        child[0][point], child[0][random_element] = child[0][random_element], child[0][point]
        child = [child[0], fitness_function(child[0]) + penalty(child, amount_of_pen, 1), 1]
        return child
    return chrmosome
def swap_truck_position(solution):
    point1 = random.randint(0, number_of_trucks - 2)
    point2 = random.randint(point1, number_of_trucks - 1)
    array = []
    for i in range(number_of_trucks):
        array.append([])
    index = 0
    for i in range(0, len(solution[0])):
        if solution[0][i] != 0:
            array[index].append(solution[0][i])
        else:
            index = index + 1
    array[point1], array[point2] = array[point2], array[point1]
    new = []
    for i in range(len(array)):
        new.extend(array[i])
        new.append(0)
    new.pop(len(new) - 1)
    new = [new, solution[1], solution[2]]
    return new
def change_0_position(parent, change0_rate, amount_of_pen):
    for i in range(number_of_trucks):
        parent = swap_truck_position(parent)
    random_number = random.random()
    if random_number <= change0_rate:
        child = [[], 0, 1]
        child[0] = [x for x in parent[0] if x != 0]
        array1 = []
        for i in range(number_of_trucks): array1.append([])
        index = 0
        sum = 0
        for i in range(len(child[0])):
           if sum + city_demand[child[0][i]] <= truck_capacity:
               array1[index].append(child[0][i])
               sum = sum + city_demand[child[0][i]]
           else:
               index = index + 1
               if index == number_of_trucks: return parent
               array1[index].append(child[0][i])
               sum = city_demand[child[0][i]]
        child = [[], 0]
        for i in range(len(array1)):
            child[0].extend(array1[i])
            child[0].append(0)
        child[0].pop(len(child[0]) - 1)
        child = [child[0], fitness_function(child[0]) + penalty(child, amount_of_pen, 1), 1]
        return child
    return parent

def mutation(population, child, mutation_rate, change0_rate, amount_of_pen):
    child_swap = swap_mutation(child, mutation_rate, amount_of_pen)
    child_swap = change_0_position(child_swap, change0_rate, amount_of_pen)
    child_inversion = inversion_mutation(child, mutation_rate, amount_of_pen)
    child_inversion = change_0_position(child_inversion, change0_rate, amount_of_pen)
    child_IRGIBNNM = IRGIBNNM_mutation(child, mutation_rate, amount_of_pen)
    child_IRGIBNNM = change_0_position(child_IRGIBNNM, change0_rate, amount_of_pen)
    temp = exist(population, child_swap)
    if temp[0] == True:
        child_swap[2] = temp[1][2]
        child_swap[1] = fitness_function(child_swap[0]) + penalty(child_swap, amount_of_pen, child_swap[2])
    temp = exist(population, child_inversion)
    if temp[0] == True:
        child_inversion[2] = temp[1][2]
        child_inversion[1] = fitness_function(child_inversion[0]) + penalty(child_inversion, amount_of_pen, child_inversion[2])
    temp = exist(population, child_IRGIBNNM)
    if temp[0] == True:
        child_IRGIBNNM[2] = temp[1][2]
        child_IRGIBNNM[1] = fitness_function(child_IRGIBNNM[0]) + penalty(child_IRGIBNNM, amount_of_pen, child_IRGIBNNM[2])
    array = [child_swap, child_inversion, child_IRGIBNNM]
    array = sorted(array, key=lambda x: x[1])
    child1 = array[0]
    if exist(population, child1)[0] == True:
        child1 = array[1]
        if exist(population, child1)[0] == True:
            child1 = array[2]
            if exist(population, child1)[0] == True:
                child1 = array[0]
    for element in population:
        if different(element, child1) == False:
            element = copy.deepcopy(child1)
    return child1


def Genetic_Algorithm(current_population, tournament_size, crossover_rate, mutation_rate, number_iteration):
    best = [[], pow(10, number_of_trucks)]
    feasible = []
    for element in current_population:
        if feasible_solution(element) == True: feasible.append(element)
    if feasible != []:
        best = min(feasible, key=lambda x: x[1])
    for i in range(number_iteration):
        print(i)
        print(best[1])
        new_population = []
        number = 0
        for element in current_population:
            if feasible_solution(element) == True: number = number + 1
        amount_of_pen = ( min(50, number_of_cities) - number )
        change0_rate = mutation_rate
        change0_rate = 2 / number_of_trucks
        for j in range(len(current_population)):
            for k in range(20):
                current_population[j] = swap_truck_position(current_population[j])
        for j in range(int(len(current_population)/2)):
            #Crossover:
            if i <= number_iteration/2: parent1, parent2 = tournament_selection(current_population, tournament_size)
            else: parent1, parent2 = roulette_wheel_selection(current_population), roulette_wheel_selection(current_population)
            child1, child2 = crossover(new_population, parent1, parent2, crossover_rate, change0_rate, amount_of_pen)
            #Mutation:
            child1, child2 = mutation(new_population, child1, mutation_rate, change0_rate, amount_of_pen), mutation(new_population, child2, mutation_rate, change0_rate, amount_of_pen)
            new_population.append(child1)
            new_population.append(child2)
        tick = int(len(current_population) * 70/100)
        new_population = sorted(new_population, key = lambda x: x[1])
        new_population1 = []
        for k in new_population:
            if exist(new_population1, k)[0] == False: new_population1.append(k)
            if len(new_population1) == tick: break
        length = len(new_population1)
        current_population = sorted(current_population, key = lambda x: x[1])
        for k in current_population:
            if exist(new_population1, k)[0] == False: new_population1.append(k)
            if len(new_population1) == min(50, number_of_cities): break
        length = len(new_population1)
        temp = []
        if length != min(50, number_of_cities):
            temp = initialize_population(min(50, number_of_cities) - length + 1, amount_of_pen)
            temp.pop(0)
        new_population1 = new_population1 + temp
        random.shuffle(new_population1)
        tick = int(len(new_population1) * 5 / 100)
        arr = random.choices(new_population1, k=tick)
        for i in range(len(arr)):
            arr[i][0] = list(TABU.Tabu_search_for_CVRP(file_path, 40, arr[i]))
        new_population1[:tick] = arr
        for j in range(len(new_population1)):
            temp = exist(current_population, new_population1[j])
            if temp[0] == True:
                new_population1[j][2] == temp[1][2] + 1
                new_population1[j][1] = fitness_function(new_population1[j][0]) + penalty(new_population1[j], amount_of_pen, new_population1[j][2])
        current_population = new_population1
        feasible = []
        for element in current_population:
            if feasible_solution(element) == True: feasible.append(element)
        if feasible != []:
            temp = min(feasible, key=lambda x: x[1])
            if temp[1] <= best[1]: best = copy.deepcopy(temp)
            if i >= number_iteration : break
    return best
start=time.time()
k = 1
array = []
for i in range(k):
    print(i)
    solution = Genetic_Algorithm(initialize_population(min(50, number_of_cities), 1), 4, 0.95, 1/min(50, number_of_cities), 70)
    array.append(solution)
end=time.time()
time=end-start
print ("Compute time:{0}".format(time) + "[sec]")
best = min(array, key = lambda x: x[1])
worst = max(array, key = lambda x: x[1])
sum = 0
for i in range(k):
    sum = sum + array[i][1]
average = sum/k

temp = 0
for i in range(len(array)):
    temp = temp + pow(array[i][1] - average, 2)
'''STD = math.sqrt(temp/(k - 1))'''
