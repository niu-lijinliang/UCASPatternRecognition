import numpy as np
import matplotlib.pyplot as plt


# generate n ints from U(xl, xu)
def generateUniformRandomInt(xl, xu, n):
    x = np.random.random_integers(xl, xu, n)
    return x


if __name__ == '__main__':
    y = []
    x = generateUniformRandomInt(-100, 100, 2)
    xl, xu = np.min(x), np.max(x)
    n = generateUniformRandomInt(1, 1000, 1)
    plt.figure(figsize=(16, 4))

    for i in 4, 5, 6:
        for j in range(np.power(10, i)):
            x = generateUniformRandomInt(xl, xu, n)
            average = np.mean(x)
            y.append(average)

        mu = np.round(np.mean(y), 2)
        sigma = np.round(np.std(y), 2)

        loc = 130 + i - 3
        plt.subplot(loc)
        plt.title('10^' + str(i) +'points ''μ=' + str(mu) + ' σ=' + str(sigma))
        plt.hist(y, bins=100)
    plt.show()