# KNAPSACK CODE


#import some standard python packages that will be useful
import array
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# import deap packages required
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import pandas as pd

# THIS FUNCTION READS THE DATA FILE CONTAINING THE INFORMATION RE EACH PLAYER

# read data
data = (pd.read_csv("clean-data.csv")
        .reset_index(drop=True))

num_players = len(data.index)

print("num possible players is %s" % (num_players))

# HELPFUL DATA 
# these can be used for calculating points and costs and are also used in the constraint_checking function
points = data['Points'] 
cost = data['Cost']
budget = 100
    

# create lists with all elements initialised to 0
gk = np.zeros(num_players)
mid = np.zeros(num_players)
defe = np.zeros(num_players)
stri = np.zeros(num_players)

for i in range(num_players):
    if data['Position'][i] == 'GK':
        gk[i] = 1
    elif data['Position'][i] == 'DEF':
        defe[i] = 1
    elif data['Position'][i] == 'MID':
        mid[i] = 1
    elif data['Position'][i] == 'STR':
        stri[i]=1
  
# check the constraints
# the function MUST be passed a list of length num_players in which each bit is set to 0 or 1
popCount = 0

# this returns a single individual: this function has the probability pInit of initialsing as feasible:
# if it is set to 0, initialisation is all random. If it is 1, initialistion is all feasible
def myInitialisationFunction(icls, size):
   
    # first create an individual with all bits set to 0
    ind = icls(np.zeros(num_players))
    gkCount = 0
    defeCount = 0
    midCount = 0
    striCount = 0
    initProb = 8
    while gkCount == 0 or defeCount < 3 or midCount < 3 or striCount < 1:
        for i in range(num_players):

            if sum(ind) < 8:
                j = random.randint(0, 523)
                if j < initProb:
                    if (data['Position'][i] == 'GK' and gkCount == 0):
                        gkCount+=1
                        ind[i] = 1
                    elif (data['Position'][i] == 'DEF' and defeCount < 3):
                        defeCount+=1
                        ind[i] = 1
                    elif (data['Position'][i] == 'MID' and midCount < 3):
                        midCount+=1
                        ind[i] = 1
                    elif (data['Position'][i] == 'STR' and striCount < 1):
                        striCount+=1
                        ind[i] = 1
            
    print(sum(ind), gkCount + defeCount + midCount + striCount, gkCount, defeCount, midCount, striCount)
    while sum(ind)<11:
        for i in range(num_players):

            if sum(ind) < 11:
                j = random.randint(0, 523)
                if j < initProb:
                    if (data['Position'][i] == 'GK' and gkCount == 0):
                        gkCount+=1
                        ind[i] = 1
                    elif (data['Position'][i] == 'DEF' and defeCount < 5):
                        defeCount+=1
                        ind[i] = 1
                    elif (data['Position'][i] == 'MID' and midCount < 5):
                        midCount+=1
                        ind[i] = 1
                    elif (data['Position'][i] == 'STR' and striCount < 3):
                        striCount+=1
                        ind[i] = 1
            
            
                
    print(sum(ind), gkCount + defeCount + midCount + striCount, gkCount, defeCount, midCount, striCount)
    print("--------------")
        
    # if u < pInit:
    #     current_weight=0
    
    #     # create a random permutation of the numbers 0 to NBR_ITEMS-1
    #     # now loop over this shuffled list from start to finish: add the item if it does not break the weight constraint
    
    #     item_indices = list(range(0, size))
    #     random.shuffle(item_indices)
    
    #     for pos in range(size):
    #         item = item_indices[pos]
    #         new_weight = weights[item]
    #         if current_weight + new_weight <= MAX_WEIGHT:
    #             ind[item]=1
    #             current_weight += new_weight
    # else:
    #     for i in range(size):
    #          ind[i] = random.randint(0, 1)
   
    #print("ind weight is %s" %(current_weight))
    return ind

# define the fitness class and creare an individual class
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# create a toolbox
toolbox = base.Toolbox()

# USE THIS LINE IF YOU WANT TO USE THE CUSTOM INIT FUNCTION
toolbox.register("individual", myInitialisationFunction, creator.Individual, num_players)

#  a population consist of a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalFootie1(individual):
    
    iCost = 0.0
    value = 0.0
    playerCountE = 0
    gkCountE = 0
    defeCountE = 0
    midCountE = 0
    striCountE = 0
    
    for item in range(num_players):
        if (individual[item]==1):
            playerCountE +=1
            iCost += cost[item]
            value += points[item]
            
            if data['Position'][item] == 'GK':
                gkCountE+= 1
            elif data['Position'][item] == 'DEF':
                defeCountE+= 1
            elif data['Position'][item] == 'MID':
                midCountE+= 1
            elif data['Position'][item] == 'STR':
                striCountE+=1
                
    if  (iCost > budget or playerCountE != 11 or gkCountE > 1 or defeCountE > 5 or midCountE > 5 or striCountE > 3):
        return 0,          
    return  value,

def check_constraints(individual):
     
    broken_constraints = 0

    # exactly 11 players
    c1 = np.sum(individual)
    if  c1 != 11:
        broken_constraints+=1
        print("total players is %s " %(c1))
        
    
    #need cost <= 100"
    c2 = np.sum(np.multiply(cost, individual)) 
    if c2 > 100:
        broken_constraints+=1
        print("cost is %s " %(c2))
    
    # need only 1 GK
    c3 = np.sum(np.multiply(gk, individual))
    if  c3 != 1:
        broken_constraints+=1
        print("goalies is %s " %(c3))
    
    # need less than 3-5 DEF"
    c4 = np.sum(np.multiply(defe,individual))
    if  c4 > 5 or c4 < 3:
        broken_constraints+=1
        print("DEFE is %s " %(c4))
            
    #need 3- 5 MID
    c5 = np.sum(np.multiply(mid,individual))
    if  c5 > 5 or c5 < 3: 
        broken_constraints+=1
        print("MID is %s " %(c5))
        
    # need 1 -1 3 STR"
    c6 = np.sum(np.multiply(stri,individual))
    if c6 > 3 or c6 < 1: 
        broken_constraints+=1
        print("STR is %s " %(c6))
        
    # get indices of players selected
    selectedPlayers = [idx for idx, element in enumerate(individual) if element==1]
    
    totalpoints = np.sum(np.multiply(points, individual))
        
        
    print("total broken constraints: %s" %(broken_constraints))
    print("total points: %s" %(totalpoints))
    print("total cost is %s" %(c2))
    print("selected players are %s" %(selectedPlayers))
    
    return broken_constraints, totalpoints

# register all operators we need with the toolbox
toolbox.register("constraints", check_constraints)
toolbox.register("evaluate", evalFootie1)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=2)


def main():
    
    # choose a population size: e.g. 200
    population = 15000
    pop = toolbox.population(n=population)
    
    # keep track of the single best solution found
    hof = tools.HallOfFame(1)
 
    # create a statistics object: we can log what ever statistics we want using this. We use the numpy Python library
    # to calculate the stats and label them with convenient labels
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # run the algorithm: we need to tell it what parameters to use
    # cxpb = crossover probability; mutpb = mutation probability; ngen = number of iterations
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.05, ngen=80, 
                                   stats=stats, halloffame=hof, verbose=True)
    print (population)
    return pop, log, hof


##############################
# run the main function 
pop, log, hof = main()

##############################
print("-------")
check_constraints(hof[0])
print("-------")

best = hof[0].fitness.values[0]   # best fitness found is stored at index 0 in the hof list


# look in the logbook to see what generation this was found at

max = log.select("max")  # max fitness per generation stored in log

for i in range(150):  # set to ngen
        fit = max[i]
        if fit == best:
            break        
        
print("max fitness found is %s at generation %s" % (best, i))
 
# code for plotting

gen = log.select("gen")
fit_max = log.select("max")
fit_min = log.select("min")
fit_avg = log.select("avg")

fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, fit_max, "b-", label="max Fitness", color="r")
line2 = ax1.plot(gen, fit_min, "b-", label="min Fitness", color="b")
line3 = ax1.plot(gen , fit_avg, "b-", label="avg Fitness", color="g")
ax1.set_xlabel("Generations")
ax1.set_ylabel("Fitness", color="b")
for tl in ax1.get_yticklabels():
    tl.set_color("b")
ax1.set_ylim(0,2500)
    
lns = line1+line2+line3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right")