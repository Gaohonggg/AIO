import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(0)

def load_data():
    data = np.genfromtxt("advertising.csv",dtype=None,delimiter=",",skip_header=1)
    X = data[:,:3]
    y = data[:,3]
    extra = np.ones((len(X),1))
    X = np.concatenate((extra,X),axis=1)
    return X, y

X,y = load_data()

# Tạo 1 cá thể
def generate_random_value(bound = 10):
    return (random.random() - 0.5)*bound

def create_individual(n=4, bound=10):
    individual = [generate_random_value() for _ in range(n)]
    return individual

# individual = create_individual()
# individual = [[4.097462559682401,4.827854760376531,3.1021723599658957,4.021659504395827]]

# Tính loss theo cá thể đó
def compute_loss(individual):
    theta = np.array(individual)
    y_hat = X.dot( theta )
    loss = (y_hat - y)**2
    loss = np.mean(loss)
    return loss

# Đánh giá mật độ tốt( fitness càng lớn càng tốt )
def compute_fitness(individual):
    loss = compute_loss(individual)
    fitness_value = 1/(loss +1)
    return fitness_value

# Lai tạo2 cá thể( Trao đổi gen )
def crossover(individual1, individual2, crossover_rate=0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()

    for i in range( len(individual1) ):
        if random.random() < crossover_rate:
            individual1_new[i] = individual2[i]
            individual2_new[i] = individual1[i]

    return individual1_new, individual2_new

# Đột biến cá thể
def mutate(individual, mutation_rate=0.05):
    individual_m = individual.copy()

    for i in range( len(individual) ):
        if random.random() < mutation_rate:
            individual_m[i] = (random.random() - 0.5)*10

    return individual_m

# Khởi tạo quần thể
def initializePopulation(m):
    population = [ create_individual() for _ in range(m) ]
    return population

# Chọn lọc tự nhiên
def selection(sorted_old_population, m=100):
    index1 = random.randint(0, m-1)
    while True:
        index2 = random.randint(0, m-1)
        if index2 != index1:
            break

    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]

    return individual_s

# Tạo ra quần thể mới
def create_new_population(old_population, elitism=2, gen=1):
    m = len(old_population)
    sorted_population = sorted(old_population, key=compute_fitness)

    if gen%1 == 0:
        print("Best loss: ",compute_loss(sorted_population[m-1])," with chromsome: ",sorted_population[m - 1])

    new_population = []
    while len(new_population) < m-elitism:
        individual1 = selection(sorted_population, m)
        individual2 = selection(sorted_population, m)
        print("cuc cu cuc cu")
        individual_1, individual_2 = crossover(individual1,individual2)

        individual_1 = mutate(individual_1)
        individual_2 = mutate(individual_2)

        new_population.append( individual_1 )
        new_population.append( individual_2 )
    print("--------------------------------------------------------------------")
    for ind in sorted_population[m-elitism:]:
        new_population.append(ind.copy())

    return new_population, compute_loss(sorted_population[m-1])

def run_GA():
    n_generations = 100
    m = 600
    population = initializePopulation(m)
    losses_list = []
    for i in range(n_generations):
        population, losses = create_new_population(population,2,i)
        losses_list.append( losses )
    return  losses_list

def visualize_loss(losses_list):
    plt.plot(losses_list, c='green')
    plt.xlabel('Generations')
    plt.ylabel('losses')
    plt.show()

losses_list = run_GA()
visualize_loss( losses_list )





































