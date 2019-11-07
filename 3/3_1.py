import numpy as np
import matplotlib.pyplot as plt

omega1 = np.array([[0.1, 6.8, -3.5, 2.0, 4.1, 3.1, -0.8, 0.9, 5.0, 3.9],
                   [1.1, 7.1, -4.1, 2.7, 2.8, 5.0, -1.3, 1.2, 6.4, 4.0]])
omega2 = np.array([[7.1, -1.4, 4.5, 6.3, 4.2, 1.4, 2.4, 2.5, 8.4, 4.1],
                   [4.2, -4.3, 0.0, 1.6, 1.9, -3.2, -4.0, -6.1, 3.7, -2.2]])
omega3 = np.array([[-3.0, 0.5, 2.9, -0.1, -4.0, -1.3, -3.4, -4.1, -5.1, 1.9],
                   [-2.9, 8.7, 2.1, 5.2, 2.2, 3.7, 6.2, 3.4, 1.6, 5.1]])
a0 = np.zeros(3)
eta = 1
theta = 0.01


def show(omega1, omega2, a, iteration):
    plt.title('Batch Perception, iteration=' + str(iteration))
    plt.xlabel('X1')
    plt.ylabel('X2')

    # 画决策面
    x = np.linspace(-5, 9, 500)
    y = -(a[0] / a[1]) * x - a[2] / a[1]
    plt.plot(x, y, label='Decision Boundary')

    # 画样本散点
    plt.scatter(omega1[0], omega1[1], s=20, c="r",
                marker='o', label='Catagory 1')
    plt.scatter(omega2[0], omega2[1], s=20, c="b",
                marker='+', label='Catagory 2')
    plt.legend()
    plt.show()


def sum_wrong(a, sample):
    n, d = sample.shape
    wrong = np.zeros(d)
    for i in range(n):
        if np.dot(a, sample[i]) <= 0:
            wrong = wrong + sample[i]
    return wrong


def batch_perception(sample):
    iteration = 0
    a = a0
    while(True):
        a = a + eta * sum_wrong(a, sample)
        iteration += 1
        if np.linalg.norm((eta * sum_wrong(a, sample)), 1) < theta:
            break
    return a, iteration


if __name__ == "__main__":
    # 规范化增广样本
    sample1 = np.row_stack((omega1, [1] * omega1.shape[1])).T
    sample2 = np.row_stack((-omega2, [-1] * omega2.shape[1])).T
    sample3 = np.row_stack((omega3, [1] * omega3.shape[1])).T

    # 分类1和2
    a, iteration = batch_perception(np.row_stack((sample1, sample2)))
    show(omega1, omega2, a, iteration)

    # 分类3和2
    a, iteration = batch_perception(np.row_stack((sample3, sample2)))
    show(omega3, omega2, a, iteration)
