
from GASolution.GA import Ga
import matplotlib.pyplot as plt
def main():

    ga = Ga(n=200, k=10, t=10, mr=0.05, ms=0.1, g=300)
    best_chromosome, best_fitness_history, avg_fitness_history = ga.run()

    plt.plot(best_fitness_history, label='best')
    plt.plot(avg_fitness_history, label='average')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig('fitness_history.png')
    plt.show()



if __name__ == '__main__':
    main()