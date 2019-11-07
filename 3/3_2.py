import numpy as np
import matplotlib.pyplot as plt

omega1 = np.array([[0.1, 6.8, -3.5, 2.0, 4.1, 3.1, -0.8, 0.9, 5.0, 3.9],
                   [1.1, 7.1, -4.1, 2.7, 2.8, 5.0, -1.3, 1.2, 6.4, 4.0]])
omega2 = np.array([[7.1, -1.4, 4.5, 6.3, 4.2, 1.4, 2.4, 2.5, 8.4, 4.1],
                   [4.2, -4.3, 0.0, 1.6, 1.9, -3.2, -4.0, -6.1, 3.7, -2.2]])
omega3 = np.array([[-3.0, 0.5, 2.9, -0.1, -4.0, -1.3, -3.4, -4.1, -5.1, 1.9],
                   [-2.9, 8.7, 2.1, 5.2, 2.2, 3.7, 6.2, 3.4, 1.6, 5.1]])
omega4 = np.array([[-2.0, -8.9, -4.2, -8.5, -6.7, -0.5, -5.3, -8.7, -7.1, -8.0],
                   [-8.4, 0.2, -7.7, -3.2, -4.0, -9.2, -6.7, -6.4, -9.7, -6.3]])
a0 = np.array([0, 0, 0])
eta = 0.8
theta = 0.01
b_min = 0.001 * np.ones(20)
k_max = 2000


def show(omega1, omega2, a, iteration):
    if iteration == k_max:
        plt.title('Ho-Kashyap, No solution found')
    else:
        plt.title('Ho-Kashyap, iteration=' + str(iteration))
    plt.xlabel('X1')
    plt.ylabel('X2')

    # 画决策面
    x = np.linspace(-9, 9, 500)
    y = -(a[0] / a[1]) * x - a[2] / a[1]
    plt.plot(x, y, label='Decision Boundary')

    # 画样本散点
    plt.scatter(omega1[0], omega1[1], s=20, c="r",
                marker='o', label='Catagory 1')
    plt.scatter(omega2[0], omega2[1], s=20, c="b",
                marker='+', label='Catagory 2')
    plt.legend()
    plt.show()


def Ho_Kashyap(sample, b):
    iteration = 0
    a = a0
    while iteration < k_max:
        iteration += 1
        e = np.matmul(sample, a) - b
        e_pseudo_inverse = 0.5 * (e + np.abs(e))
        b = b + 2 * eta *e_pseudo_inverse
        a = np.matmul(np.linalg.pinv(sample), b)
        if (np.abs(e) <= b_min).all():
            break
    return a, iteration


if __name__ == "__main__":
    sample1 = np.row_stack((omega1, [1] * omega1.shape[1])).T
    sample3 = np.row_stack((-omega3, [-1] * omega3.shape[1])).T

    sample = np.row_stack((sample1, sample3))
    b = np.ones(sample.shape[0])
    a, iteration = Ho_Kashyap(sample, b)
    show(omega1, omega3, a, iteration)

    sample2 = np.row_stack((omega2, [1] * omega2.shape[1])).T
    sample4 = np.row_stack((-omega4, [-1] * omega4.shape[1])).T

    sample = np.row_stack((sample2, sample4))
    b = np.ones(sample.shape[0])
    a, iteration = Ho_Kashyap(sample, b)
    show(omega2, omega4, a, iteration)