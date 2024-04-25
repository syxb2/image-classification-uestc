import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

# 这里是绘制混淆矩阵函数的定义
def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
) -> None:
    # This function prints and plots the confusion matrix.
    # Normalization can be applied by setting `normalize=True`.
    # Input
    # - cm : 计算出的混淆矩阵的值
    # - classes : 混淆矩阵中每一行每一列对应的列
    # - normalize : True:显示百分比, False:显示个数
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # 变量转换为float类型，除以每一个行向量相加得数,[:, np.newaxis]指选取的数据增加一个维度
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
        print(cm)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)  # 将数据显示为图像
    plt.title(title)  # 设置标题
    plt.colorbar()  # 设置legend
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".1f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            size="xx-small",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("test.jpg", dpi=300)

    return


def result_evalution(truelable_lst, prelable_lst) -> None:
    print(
        "该训练模型的预测结果的F1值："
        + str(f1_score(truelable_lst, prelable_lst, average="macro"))
    )
    print(
        "该训练模型的预测结果的精确率："
        + str(precision_score(truelable_lst, prelable_lst, average="macro"))
    )
    print(
        "该训练模型的预测结果的准确率："
        + str(accuracy_score(truelable_lst, prelable_lst))
    )
    print(
        "该训练模型的预测结果的召回率："
        + str(recall_score(truelable_lst, prelable_lst, average="macro"))
    )

    return
