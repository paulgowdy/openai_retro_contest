import matplotlib.pyplot as plt
import numpy as np


def visualize_dist(list_of_numbers):

    plt.figure()
    plt.hist(list_of_numbers)
    plt.show()


def rand5():

    # Returns a uniform random integer between 1 and 5 inclusive

    return int(np.random.randint(1,6))

## Below is one solution to the problem: Given rand5() write rand7()

def rand7():

    # NOT UNIFORM, will favor lower indices because max() returns the first max in case of ties
    # Although ties are very unlikely after 200 rolls...
    '''
    rolls = [0] * 7

    for i in range(200):

        rolls = [x + rand5() for x in rolls]

    return rolls.index(max(rolls)) + 1
    '''

    #UNIFORM!

    possible_max = range(7)

    while len(possible_max) != 1:

        current_max = 0

        for r in possible_max:

            roll = rand5()

            if roll > current_max:

                current_max = roll
                possible_max = [r]

            elif roll == current_max:

                possible_max.append(r)

            else:

                pass

    return possible_max[0] + 1


a = []

for i in range(210000):

    a.append(rand7())

visualize_dist(a)
