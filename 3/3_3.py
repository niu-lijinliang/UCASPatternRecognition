import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

CATAGORY_NUMBER = 4
CATAGORY_LABELS = [0, 1, 2, 3]
TRAIN_NUMBER = 8
TEST_NUMBER = 2
omega1 = np.array([[0.1, 6.8, -3.5, 2.0, 4.1, 3.1, -0.8, 0.9, 5.0, 3.9],
                   [1.1, 7.1, -4.1, 2.7, 2.8, 5.0, -1.3, 1.2, 6.4, 4.0]])
omega2 = np.array([[7.1, -1.4, 4.5, 6.3, 4.2, 1.4, 2.4, 2.5, 8.4, 4.1],
                   [4.2, -4.3, 0.0, 1.6, 1.9, -3.2, -4.0, -6.1, 3.7, -2.2]])
omega3 = np.array([[-3.0, 0.5, 2.9, -0.1, -4.0, -1.3, -3.4, -4.1, -5.1, 1.9],
                   [-2.9, 8.7, 2.1, 5.2, 2.2, 3.7, 6.2, 3.4, 1.6, 5.1]])
omega4 = np.array([[-2.0, -8.9, -4.2, -8.5, -6.7, -0.5, -5.3, -8.7, -7.1, -8.0],
                   [-8.4, 0.2, -7.7, -3.2, -4.0, -9.2, -6.7, -6.4, -9.7, -6.3]])
lamda = 0.01


def show(w, accuracy):
    plt.title('MSE, accuracy=' + str(accuracy))

    # 绘制决策区域
    x, y = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))
    X_ = np.concatenate((x.reshape(1, -1), y.reshape(1, -1),
                         np.ones((1, 1000*1000))), axis=0)
    pred = np.dot(w.T, X_).argmax(0).reshape(1000, 1000)
    cm = mpl.colors.ListedColormap(
        ['#FFA0A0', '#fff2a0', '#A0FFA0', '#A0DCFF'])
    plt.pcolormesh(x, y, pred, cmap=cm)

    plt.scatter(omega1[0, 0:TRAIN_NUMBER], omega1[1, 0:TRAIN_NUMBER],
                c='r', marker='o', label=r'${\omega}_1$' + '-train')
    plt.scatter(omega1[0, TRAIN_NUMBER:], omega1[1, TRAIN_NUMBER:],
                c='r', marker='x', label=r'${\omega}_1$' + '-test')
    plt.scatter(omega2[0, 0:TRAIN_NUMBER], omega2[1, 0:TRAIN_NUMBER],
                c='y', marker='o', label=r'${\omega}_2$' + '-train')
    plt.scatter(omega2[0, TRAIN_NUMBER:], omega2[1, TRAIN_NUMBER:],
                c='y', marker='x', label=r'${\omega}_2$' + '-test')
    plt.scatter(omega3[0, 0:TRAIN_NUMBER], omega3[1, 0:TRAIN_NUMBER],
                c='g', marker='o', label=r'${\omega}_3$' + '-train')
    plt.scatter(omega3[0, TRAIN_NUMBER:], omega3[1, TRAIN_NUMBER:],
                c='g', marker='x', label=r'${\omega}_3$' + '-test')
    plt.scatter(omega4[0, 0:TRAIN_NUMBER], omega4[1, 0:TRAIN_NUMBER],
                c='b', marker='o', label=r'${\omega}_4$' + '-train')
    plt.scatter(omega4[0, TRAIN_NUMBER:], omega4[1, TRAIN_NUMBER:],
                c='b', marker='x', label=r'${\omega}_4$' + '_test')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 训练
    X_train = np.hstack((omega1[:, 0:TRAIN_NUMBER], omega2[:, 0:TRAIN_NUMBER],
                         omega3[:, 0:TRAIN_NUMBER], omega4[:, 0:TRAIN_NUMBER]))
    X_train = np.row_stack((X_train, [1] * CATAGORY_NUMBER * TRAIN_NUMBER))

    labels_train = np.repeat(CATAGORY_LABELS, TRAIN_NUMBER)
    Y = np.eye(CATAGORY_NUMBER)[labels_train].T
    w = np.linalg.pinv(np.matmul(X_train, X_train.T) + lamda *
                       np.identity(X_train.shape[0])).dot(X_train).dot(Y.T)

    # 测试
    X_test = np.hstack(
        (omega1[:, TRAIN_NUMBER:], omega2[:, TRAIN_NUMBER:],
         omega3[:, TRAIN_NUMBER:], omega4[:, TRAIN_NUMBER:]))
    X_test = np.row_stack((X_test, [1] * CATAGORY_NUMBER * TEST_NUMBER))

    labels_test = np.repeat(CATAGORY_LABELS, TEST_NUMBER)
    predict_labels = w.T.dot(X_test).argmax(0)
    accuracy = (labels_test == predict_labels).sum() / (CATAGORY_NUMBER * TEST_NUMBER)

    show(w, accuracy)