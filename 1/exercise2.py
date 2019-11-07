import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


# generate n samples from N(mu, sigma)
def generateNormSamples(d, mu, sigma, n):
    s = None
    if d == mu.shape[0]:
        s = np.random.multivariate_normal(mu, sigma, n)
    else:
        print('维数应与均值维度相同！')
        exit()
    return s


def CalZfor2DNorm(mu, sigma):
    N = 50
    x = y = np.linspace(-4, 4, N)
    mesh = np.meshgrid(x, y)
    z = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            vec = np.array([mesh[0][i][j], mesh[1][i][j]])
            p = multivariate_normal.pdf(vec, mu, sigma)
            z[i][j] = p

    return mesh, z


def drawNormAndPlane(mu1, sigma1, mu2, sigma2):
    mesh1, z1 = CalZfor2DNorm(mu1, sigma1)
    mesh2, z2 = CalZfor2DNorm(mu2, sigma2)

    w = np.matmul(np.linalg.inv(sigma), (mu1 - mu2))
    x0 = (mu1 + mu2) / 2

    N = 50
    x = np.linspace(-0.1, 0.1, N)
    y = np.linspace(-4, 4, N)
    mesh = np.meshgrid(x, y)
    z = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            vec = np.array([mesh[0][i][j], mesh[1][i][j]])
            p = np.matmul(w.T, vec-x0)
            z[i][j] = p

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(mesh1[0], mesh1[1], z1, cmap=plt.get_cmap('rainbow'))
    ax.plot_surface(mesh2[0], mesh2[1], z2, cmap=plt.get_cmap('rainbow'))
    ax.plot_surface(mesh[0], mesh[1], z)
    plt.show()


if __name__ == '__main__':
    n = 100
    sigma = np.array([[1, 0], [0, 1]])
    mu1 = np.array([1, 0])
    mu2 = np.array([-1, 0])

    drawNormAndPlane(mu1, sigma, mu2, sigma)

    fig = plt.figure(figsize=(16, 9))
    x1, y1 = generateNormSamples(2, mu1, sigma, int(n / 2)).T
    plt.plot(x1, y1, 'x')

    x2, y2 = generateNormSamples(2, mu2, sigma, int(n / 2)).T
    plt.plot(x2, y2, 'x')

    # 通过之前的图可知决策面二维投影为x=0
    x = np.zeros(5)
    y = np.linspace(-5, 5, 5)
    plt.plot(x, y)

    error1_2 = 0
    error2_1 = 0

    for i in x1:
        if i < 0:
            error1_2 += 1

    for i in x2:
        if i > 0:
            error2_1 += 1

    empiricalError = (error1_2 + error2_1) / n
    plt.title('empirical error = ' + str(empiricalError))

    plt.show()