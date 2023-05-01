import copy
import numpy
import math
import random
import time
INT_MAX= 100000000000

def read_data(path):
    global data
    global number_of_cities
    global number_of_trucks
    global truck_capacity
    global distance_matrix
    global city
    global city_demand
    global tabu_tenure
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
    tabu_tenure = int(math.sqrt(number_of_cities))
    distance_matrix = numpy.array(distance_matrix)
    f.close()

def Calculate_distance_of_truck(route):
    distance_of_truck=[]
    for i in range(number_of_trucks):
        fitness = 0
        num = len(route[i])
        if num ==0:
            distance_of_truck.append(0)
        else:
            for j in range(len(route[i])+1):
                if j == 0:
                    fitness += distance_matrix[0][route[i][0]]
                elif j == num:
                    fitness += distance_matrix[route[i][j-1]][0]
                else:
                    fitness += distance_matrix[route[i][j-1]][route[i][j]]
            distance_of_truck.append(fitness)
    return distance_of_truck

def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def CheckIndividualValid(route,ratio_infeasible):
    Penalty=[]
#    Check = True
    for i in range(len(route)):
        Total_Demand=0
        for j in range(len(route[i])):
            Total_Demand += city_demand[route[i][j]]
        if Total_Demand > truck_capacity:
#            Check =False
            Penalty.append(float(Total_Demand/truck_capacity) * ratio_infeasible)
        else:
            Penalty.append(1)
    return Penalty

def Transform_add(route,city_changed,truck_changed):
    min = INT_MAX
    index=0
    num = len(route[truck_changed])
    route1 = route.copy()
    route1[truck_changed] = route[truck_changed].copy()
    if num == 0:
        index = 0
        min = distance_matrix[0][city_changed] + distance_matrix[city_changed][0]
    else:
        for i in range(len(route[truck_changed])+1):
            if i == 0:
                if distance_matrix[0][city_changed] + distance_matrix[city_changed][route[truck_changed][0]] \
                        - distance_matrix[0][route[truck_changed][0]] < min:
                    min = distance_matrix[0][city_changed] + distance_matrix[city_changed][route[truck_changed][0]] \
                        - distance_matrix[0][route[truck_changed][0]]
                    index = i
            elif i == num:
                if distance_matrix[route[truck_changed][num-1]][city_changed] + distance_matrix[city_changed][0] \
                        - distance_matrix[route[truck_changed][num-1]][0] < min:
                    min = distance_matrix[route[truck_changed][num-1]][city_changed] + distance_matrix[city_changed][0] \
                        - distance_matrix[route[truck_changed][num-1]][0]
                    index =i
            else:
                if distance_matrix[route[truck_changed][i-1]][city_changed] + distance_matrix[city_changed][route[truck_changed][i]] \
                        - distance_matrix[route[truck_changed][i-1]][route[truck_changed][i]] < min :
                    min = distance_matrix[route[truck_changed][i-1]][city_changed] + distance_matrix[city_changed][route[truck_changed][i]] \
                        - distance_matrix[route[truck_changed][i-1]][route[truck_changed][i]]
                    index = i
    route1[truck_changed].insert(index,city_changed)
    return route1, min

def Transform_remove(route,num_city_changed,truck_changed):
    num = len(route[truck_changed])-1
    route1 = route.copy()
    route1[truck_changed] = route[truck_changed].copy()
    if num == 0:
        change = - distance_matrix[0][route[truck_changed][num_city_changed]] - distance_matrix[route[truck_changed][num_city_changed]][0]
    else:
        if num_city_changed == 0:
            change = distance_matrix[0][route[truck_changed][num_city_changed+1]] \
            - distance_matrix[0][route[truck_changed][num_city_changed]] \
                     - distance_matrix[route[truck_changed][num_city_changed]][route[truck_changed][num_city_changed+1]]

        elif num_city_changed == num:
            change = distance_matrix[route[truck_changed][num_city_changed-1]][0] \
            - distance_matrix[route[truck_changed][num_city_changed-1]][route[truck_changed][num]] \
                     - distance_matrix[route[truck_changed][num]][0]
        else:
            change = distance_matrix[route[truck_changed][num_city_changed - 1]][route[truck_changed][num_city_changed + 1]] \
                     - distance_matrix[route[truck_changed][num_city_changed - 1]][route[truck_changed][num_city_changed]] - \
                     distance_matrix[route[truck_changed][num_city_changed]][route[truck_changed][num_city_changed+1]]

    route1[truck_changed].pop(num_city_changed)
    return route1, change


def Neighbourhood_swap(route, distance_of_truck):
    Neighbourhood=[]

    for i in range(len(route)):
        for j in range(len(route[i])):
            for k in range(len(route)):
                if i == k:
                    continue
                else:
                    route1, change1 = Transform_remove(route, j, i)

                    route1, change2 = Transform_add(route1, route[i][j], k)

                    Penalty = CheckIndividualValid(route1, ratio_infeasible)
                    new_fitness = 0
                    for m in range(len(distance_of_truck)):
                        if m == i:
                            new_fitness += ( change1 + distance_of_truck[m]) * Penalty[m]
                        elif m == k:
                            new_fitness += ( change2 + distance_of_truck[m]) * Penalty[m]
                        else:
                            new_fitness += distance_of_truck[m] * Penalty[m]
                    arr = [i, route[i][j], k, route1, new_fitness]
                    Neighbourhood.append(arr)
    return Neighbourhood

def Tabu_search_for_CVRP(filepath, Stopping_Condition, first_individual):
    global best_sol
    global best_fitness
    global best_fitness_valid
    global best_sol_valid
    global Tabu_Structure
    global ratio_infeasible
    global ratio_infeasible_for_best
    ratio_infeasible = 1
    ratio_infeasible_for_best = 1
    num_ite_best = 0
    num_ite_current = 0

    read_data(filepath)
    Tabu_Structure = []
    for i in range(number_of_cities):
        row = [tabu_tenure * (-1)] * number_of_trucks
        Tabu_Structure.append(row)
    current_sol = []
    index = 0
    for i in range(number_of_trucks):
        current_sol.append([])
    for i in range(len(first_individual[0])):
        if first_individual[0][i] != 0 : current_sol[index].append(first_individual[0][i])
        else:
            index = index + 1
    distance_of_truck = Calculate_distance_of_truck(current_sol)
    best_sol = copy.deepcopy(current_sol)
    best_sol_valid = copy.deepcopy(current_sol)
    best_fitness = sum(distance_of_truck)
    best_fitness_valid = best_fitness
    for i in range(Stopping_Condition):
        current_neighbourhood = Neighbourhood_swap(current_sol,distance_of_truck)
        index = -1;
        min = INT_MAX
        for j in range(len(current_neighbourhood)):
            cfnode = current_neighbourhood[j][4]
            '''cfnode1 = sum(Calculate_distance_of_truck(current_neighbourhood[j][2]))
            print(CheckIndividualValid(current_neighbourhood[j][2]))
            print(current_neighbourhood[j][0],", ",current_neighbourhood[j][1],", ",cfnode1,", ",cfnode,": ")
            print(current_neighbourhood[j][2])
            if cfnode1 == cfnode: print("true")
            else: print("False")'''
            if cfnode < best_fitness:
                min = cfnode
                index = j
                best_fitness = cfnode
                best_sol = current_neighbourhood[j][3]

            elif cfnode < min and Tabu_Structure[current_neighbourhood[j][1]][current_neighbourhood[j][0]] + tabu_tenure <= i:

                min = cfnode
                index = j
        current_sol = current_neighbourhood[index][3]
        Tabu_Structure[current_neighbourhood[index][1]][current_neighbourhood[index][2]] = i
        if CheckIndividualValid(current_sol,1) == [1]*number_of_trucks:
            num_ite_current = i
        if i - num_ite_current > 4:
            ratio_infeasible = float((i-num_ite_current)/4)
        else:
            ratio_infeasible = 1
        if CheckIndividualValid(best_sol,1) == [1] * number_of_trucks:
            num_ite_best = i
        if i - num_ite_best > 3:
            Penalty = CheckIndividualValid(best_sol,float((i - num_ite_best)/3))
            best_fitness = 0
            for m in range(number_of_trucks):
                best_fitness += distance_of_truck[m] * Penalty[m]
        if CheckIndividualValid(best_sol, 1) == [1] * number_of_trucks:
            if sum(Calculate_distance_of_truck(best_sol)) < sum(Calculate_distance_of_truck(best_sol_valid)):
                best_sol_valid = copy.deepcopy(best_sol)
        distance_of_truck = Calculate_distance_of_truck(current_sol)
    best_fitness_valid = sum(Calculate_distance_of_truck(best_sol_valid))
    solution = []
    for i in range(len(best_sol_valid)):
        solution.extend(best_sol_valid[i])
        solution.append(0)
    solution.pop(len(solution) - 1)
    return solution


"""start=time.time()
Stopping_Condition = 1000
solution = Tabu_search_for_CVRP("A-n60-k9.vrp.txt", Stopping_Condition)
print(solution)
end=time.time()
time=end-start
print("Compute time:{0}".format(time) + "[sec]")"""

'''
a=[[1,3,4,31],[5,6,7,8,9,10,25,26,27,28,29,30],[11,12,13,14,15],[16,17,18,19,20,21,22,23],[1,2]]


route = [[1,3,4,31],[5,6,7,8,9,10,25,26,27,28,29,30],[11,12,13,14,15],[16,17,18,19,20,21,22,23],[1,2]]
print(tabu_tenure)
penalty = CheckIndividualValid(route)

print(route)
route2, change = Transform_remove(route,1,4)
print(route2)
dis = Calculate_distance_of_truck(route)
nei = Neighbourhood_swap(route,Calculate_distance_of_truck(route))
print(nei)
'''



