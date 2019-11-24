import numpy as np
import matplotlib.pyplot as plt

omega1 = np.array([[1.58, 2.32, -5.8], [0.67, 1.58, -4.78], [1.04, 1.01, -3.63],
                   [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
                   [1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [0.45, 1.33, -4.38],
                   [-0.76, 0.84, -1.96]])
omega2 = np.array([[0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16],
                   [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
                   [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14],
                   [0.46, 1.49, 0.68]])
omega3 = np.array([[-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [1.55, 0.99, 2.69],
                   [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
                   [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [0.25, 0.68, -0.99],
                   [0.66, -0.45, 0.08]])
CLASS_LABELS = [0, 1, 2]
CLASS_NUMBER = 3
max_epoch = 1000


def drawError(errors):
    plt.title('Error-Iteration')
    plt.plot([[i] for i in range(len(errors))], errors, ms=1)
    plt.show()


def multilayer_perceptron(x, labels, batch_size=1, hidden_node=8, eta=0.1):
    max_iteration = int(x.shape[0] // batch_size)

    # 随机初始化权重
    w1 = 0.01 * np.random.randn(hidden_node, x.shape[1])
    w2 = 0.01 * np.random.randn(CLASS_NUMBER, hidden_node)

    square_error = 0
    errors = []
    for epoch in range(max_epoch):
        for iteration in range(max_iteration):
            net1 = w1.dot(x.T).T
            y = (np.exp(net1) - np.exp(0 - net1)) / \
                (np.exp(net1) + np.exp(0 - net1))

            net2 = w2.dot(y.T).T
            z = 1 / (1+np.exp(0-net2))
            error = labels - z
            square_error = 0.5 * np.sum(np.power(error, 2))

            errors.append(square_error)

            # BP
            df2 = z * (1-z)
            delta2 = error * df2
            dw2 = (y.T.dot(delta2)).T

            df1 = 1 / np.power((np.exp(net1) + np.exp(0-net1))/2, 2)
            delta1 = (delta2.dot(w2)) * df1
            dw1 = (x.T .dot(delta1)).T

            w2 += eta * dw2
            w1 += eta * dw1

    pred = np.argmax(z, axis=1)
    label = np.argmax(labels, axis=1)
    acc = np.sum(pred == label) / x.shape[0]

    drawError(errors)

    return square_error, acc


if __name__ == "__main__":
    # 增广样本
    x = np.vstack((omega1, omega2, omega3))
    x = np.column_stack((x, [1] * 30))

    # 建立标签
    labels_train = np.repeat(CLASS_LABELS, 10)
    labels = np.eye(3)[labels_train]

    # 批量训练
    batch_size = x.shape[0] / CLASS_NUMBER
    b_error, b_res = multilayer_perceptron(
        x, labels, batch_size)

    # 单样本训练
    s_error, s_res = multilayer_perceptron(x, labels, 1)
