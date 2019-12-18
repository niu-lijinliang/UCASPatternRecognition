import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.datasets import fetch_openml

NUMBER1, NUMBER2 = '3', '8'

if __name__ == "__main__":
    # 加载数据
    mnist_data = fetch_openml('mnist_784')
    index = np.isin(mnist_data['target'], [NUMBER1, NUMBER2])
    x, y = mnist_data['data'][index], mnist_data['target'][index]
    # 生成数据集，训练集占0.6
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.6)

    # 训练SVM分类器
    classfifier = svm.SVC().fit(x_train, y_train)
    # SVM预测
    y_predict = classfifier.predict(x_test)
    score = metrics.classification_report(y_test, y_predict)
    print(score)
