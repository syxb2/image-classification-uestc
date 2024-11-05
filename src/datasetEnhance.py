import math
import os
import numpy as np
import random
from PIL import Image


def OneMultplt(A, B, i=0, j=0):
    out = 0
    for n in range(len(A[0])):
        out += A[i][n] * B[n][j]
    return out


def DotMatrix(A, B):
    if len(A[0]) == len(B):
        res = [[0] * len(B[0]) for i in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                res[i][j] = OneMultplt(A, B, i, j)
        return res

    else:
        return "输入矩阵有误！"


class Img:
    def __init__(self, image, rows, cols, center=[0, 0]):
        self.src = image
        self.rows = rows
        self.cols = cols
        self.center = center

    def Move(self, delta_x, delta_y, img):  # 平移
        self.transform = np.array([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])
        img.Process()
        img2 = Image.fromarray(img.dst)
        return img2

    def Zoom(self, factor, img):  # 缩放
        self.transform = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        img.Process()
        img2 = Image.fromarray(img.dst)
        return img2

    def Horizontal(self, img):  # 水平镜像
        # '''水平镜像
        # 镜像的这两个函数，因为原始图像读进来后是height×width×3,和我们本身思路width×height×3相反
        # 所以造成了此处水平镜像和垂直镜像的tranform做了对调，同学们可以考虑如何把它们调换回来？'''
        self.transform = np.array([[1, 0, 0], [0, -1, self.cols - 1], [0, 0, 1]])
        img.Process()
        img2 = Image.fromarray(img.dst)
        return img2

    # 垂直镜像，transform用的是PPT上所述的水平镜像的transfrom，理由如上所述。
    def Vertically(self, img):  # 垂直镜像
        self.transform = np.array([[-1, 0, self.rows - 1], [0, 1, 0], [0, 0, 1]])
        img.Process()
        img2 = Image.fromarray(img.dst)
        return img2

    def Rotate(self, beta, img):  # 旋转
        self.transform = np.array(
            [
                [math.cos(beta), -math.sin(beta), 0],
                [math.sin(beta), math.cos(beta), 0],
                [0, 0, 1],
            ]
        )
        img.Process()
        img2 = Image.fromarray(img.dst)
        return img2

    def Process(self):
        # 初始化定义目标图像，具有3通道RBG值，一定要注意dst和src的通道值是否对应.
        self.dst = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)

        # 提供for循环，遍历图像中的每个像素点，然后使用矩阵乘法，找到变换后的坐标位置
        for i in range(self.rows):
            for j in range(self.cols):
                src_pos = np.array([[i - self.center[0]], [j - self.center[1]], [1]])
                [[x], [y], [z]] = DotMatrix(self.transform, src_pos)
                x = int(x) + self.center[0]
                y = int(y) + self.center[1]
                if x >= self.rows or y >= self.cols or x < 0 or y < 0:
                    self.dst[i][j] = 255
                else:

                    self.dst[i][j] = self.src[x][y]


def zengqiangjuzhen(path):
    list = os.listdir(path)
    for food_class in list:
        images = os.listdir(path + food_class)
        for i in images:
            imgv = Image.open("train/" + food_class + "/" + i)
            imgv = np.array(imgv)
            rows = len(imgv)
            cols = len(imgv[0])
            img = Img(imgv, rows, cols, [0, 0])

            a = random.randint(0, int(rows / 2))
            b = random.randint(0, int(cols / 2))
            img2 = img.Move(a, b, img)

            c = random.random()
            img3 = img.Zoom(c, img)

            img4 = img.Vertically(img)

            img5 = img.Horizontal(img)

            d = random.uniform(0, 45)
            img6 = img.Rotate(math.radians(d), img)

            img2.save(os.path.join(path, food_class, "2_" + i))
            img3.save(os.path.join(path, food_class, "3_" + i))
            img4.save(os.path.join(path, food_class, "4_" + i))
            img5.save(os.path.join(path, food_class, "5_" + i))
            img6.save(os.path.join(path, food_class, "6_" + i))


zengqiangjuzhen("/home/aistudio/test/")

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.
