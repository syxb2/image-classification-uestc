import os
import paddlehub as hub
import matplotlib.pyplot as plt
import numpy as np

modulexcp = hub.Module(name="xception71_imagenet")
# moduleres = hub.Module(name="resnet_v2_50_imagenet")
modulevgg = hub.Module(name="vgg19_imagenet")


# identify the function used to iterate through all the files in the folder
def listdir(
    path, pathlist
):  # pathlist: input the list which is used to store the paths
    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        if os.path.isdir(file_path):
            listdir(file_path, pathlist)
        else:
            pathlist.append(file_path)


bassetpath = "/home/aistudio/test-dog10/1-basset-test"
beaglepath = "/home/aistudio/test-dog10/2-beagle-test"
houndpath = "/home/aistudio/test-dog10/3-gray hound-test"
shepherdpath = "/home/aistudio/test-dog10/4-german shepherd-test"
chnauzerpath = "/home/aistudio/test-dog10/5-schnauzer-test"
spanielpath = "/home/aistudio/test-dog10/6-springer spaniel-test"
labradorpath = "/home/aistudio/test-dog10/7-labrador-test"
cockerpath = "/home/aistudio/test-dog10/8-cocker-test"
sheepdogpath = "/home/aistudio/test-dog10/9-old english sheepdog-test"
shetlandpath = "/home/aistudio/test-dog10/10-shetland-test"
allpath = "/home/aistudio/test-dog10"

bassetlist = []
beaglelist = []
houndlist = []
shepherdlist = []
chnauzerlist = []
spaniellist = []
labradorlist = []
cockerlist = []
sheepdoglist = []
shetlandlist = []

petpath = [
    bassetpath,
    beaglepath,
    houndpath,
    shepherdpath,
    chnauzerpath,
    spanielpath,
    labradorpath,
    cockerpath,
    sheepdogpath,
    shetlandpath,
]
petlist = [
    bassetlist,
    beaglelist,
    houndlist,
    shepherdlist,
    chnauzerlist,
    spaniellist,
    labradorlist,
    cockerlist,
    sheepdoglist,
    shetlandlist,
]

# Identify the dog with one word (identify the folder tags):
pett = [
    "basset",
    "beagle",
    "gray",
    "german",
    "schnauzer",
    "springer",
    "labrador",
    "cocker",
    "sheepdog",
    "shetland",
]

# give value to '<pet>list'
for path, pet in zip(petpath, petlist):
    listdir(path, pet)
# print(bassetlist)

# 'path: animal' in 'dic'
dic = dict()
for petli, pet in zip(petlist, pett):  # petlist is a list
    for petl in petli:
        dic[petl] = [pet]
# print(dic)

# 'pathlist' has all-pic's path
pathlist = []
listdir(allpath, pathlist)
# print(pathlist)

# ------------------------------------------------------CONDUCT CLASSIFICATION--------------------------------------------------------- #

# conduct classification with xcption17:
xcor = 0
xtot = 0
for i in pathlist:
    xtot += 1
    input_dict = {"image": [i]}
    xresults = modulexcp.classification(data=input_dict)
    print("xcption17's predicted class: %s" % (list(xresults[0][0].keys()))[0])
    l = list((xresults[0][0]).keys())
    print("real class: %s" % (dic[i][0]))  # dic[i][0] is the real class of path i.
    # l[0].lower() is the result of the prediction.
    if l[0].lower().find(dic[i][0]) != -1:
        xcor += 1
        print("True")
    else:
        print("False")
    print(" ")

xres = xcor / xtot
print(
    "/*------------------------------------RESULT----------------------------------------*/"
)
print("xcption17's correctness is : %lf" % (xres / xtot))
print(" ")

# conduct classification with resnet50:
rcor = 0
rtot = 0
for i in pathlist:
    rtot += 1
    input_dict = {"image": [i]}
    rresults = moduleres.classification(data=input_dict)
    print("resnet50's predicted class: %s" % (list(rresults[0][0].keys()))[0])
    l = list((rresults[0][0]).keys())
    print("real class: %s" % (dic[i][0]))  # dic[i][0] is the real class of path i.
    # l[0].lower() is the result of the prediction.
    if l[0].lower().find(dic[i][0]) != -1:
        rcor += 1
        print("True")
    else:
        print("False")
    print(" ")

rres = rcor / rtot
print(
    "/*------------------------------------RESULT----------------------------------------*/"
)
print("resnet50's correctness is : %lf" % (rres))
print(" ")

# conduct classification with vgg19:
vcor = 0
vtot = 0
for i in pathlist:
    vtot += 1
    input_dict = {"image": [i]}
    vresults = modulevgg.classification(data=input_dict)
    print("vgg19's predicted class: %s" % (list(vresults[0][0].keys()))[0])
    l = list((vresults[0][0]).keys())
    print("real class: %s" % (dic[i][0]))  # dic[i][0] is the real class of path i.
    # l[0].lower() is the result of the prediction.
    if l[0].lower().find(dic[i][0]) != -1:
        vcor += 1
        print("True")
    else:
        print("False")
    print(" ")

vres = vcor / vtot
print(
    "/*------------------------------------RESULT----------------------------------------*/"
)
print("vgg19's correctness is : %lf" % (vres))
print(" ")

# print chart
x = ["xcption17", "resnet50", "vgg19"]
y = [xres, rres, vres]

plt.title("modules correctness")
plt.xlabel("module")
plt.ylabel("correctness")
plt.bar(x, y)
plt.show()

# ---------------------------------------------------CALCULATE THE METRICS（指标）-------------------------------------------------------------- #

"""
calculate x's Accuracy and Precise and Recall and F1
TP = 0
FP = 0
FN = 0
TN = 0

for i in pathlist:
    input_dict = {"image": [i]}
    xresults = modulexcp.classification(data=input_dict)
    # print("xcption17's predicted class: %s" % (list(xresults[0][0].keys()))[0])
    l = list((xresults[0][0]).keys())
    # print("real class: %s" % (dic[i][0]))
    # if l[0].lower().find(dic[i][0]) != -1:
    if l[0].lower().find("basset") != -1:
        # print("True")
        TP += 1
    elif l[0].lower().find("basset") == -1 and dic[i][0] == "basset":
        # print("False")
        FP += 1
    elif l[0].lower().find("basset") != -1 and dic[i][0] != "basset":
        FN += 1
    else:
        TN += 1

print("For basset:")
print("A = %f" % ((TP + TN) / (TP + FP + TN + FN)))
print("P = %f" % (TP / (TP + FP)))
print("R = %f" % (TP / (TP + FN)))
"""

xmatx = np.zeros((11, 10))

numdic = {
    "basset": 0,
    "beagle": 1,
    "gray": 2,
    "german": 3,
    "schnauzer": 4,
    "springer": 5,
    "labrador": 6,
    "cocker": 7,
    "sheepdog": 8,
    "land": 9,
}

for i in pathlist:
    input_dict = {"image": [i]}
    xresults = modulexcp.classification(data=input_dict)
    l = list((xresults[0][0]).keys())
    num = -1
    # l[0].lower() is the result of the prediction.
    if l[0].lower().find("basset") != -1:
        num = 0
    elif l[0].lower().find("beagle") != -1:
        num = 1
    elif l[0].lower().find("gray") != -1:
        num = 2
    elif l[0].lower().find("german") != -1:
        num = 3
    elif l[0].lower().find("schnauzer") != -1:
        num = 4
    elif l[0].lower().find("springer") != -1:
        num = 5
    elif l[0].lower().find("labrador") != -1:
        num = 6
    elif l[0].lower().find("cocker") != -1:
        num = 7
    elif l[0].lower().find("sheepdog") != -1:
        num = 8
    elif l[0].lower().find("land") != -1:
        num = 9
    else:
        num = 10
    xmatx[num][numdic[dic[i][0]]] += 1  # dic[i][0] is the real class of path i.

print(xmatx)

# ------------------------------------------------------------DRAW MATRIX---------------------------------------------------------------- #

import itertools
import numpy as np
import matplotlib.pyplot as plt


# 这里是绘制混淆矩阵函数的定义
def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
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


attack_types = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
plot_confusion_matrix(
    xmatx, classes=attack_types, normalize=True, title="Normalized confusion matrix"
)
